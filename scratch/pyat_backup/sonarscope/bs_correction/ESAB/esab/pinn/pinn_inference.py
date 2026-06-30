import torch
import numpy as np
import netCDF4
import os
import sys
import time
import xarray as xr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pinn_model import PINNESABTransformer
from pinn_esab_forward import DifferentiableESAB


def export_yolo_features(nc_path, out_nc_path, weights_path="pinn_esab_finetuned.pth",
                         freq_hz=300000.0, gamma=10/3):
    """
    Rapid inference using the PINN to extract 5 feature channels.
    Channels: [z, mu_db, s1, delta2, rmse_physics]
    - s1: 归一化粗糙度 (频率无关, 参考频率 150kHz)
    - rmse: 仅在原始有效角度上计算，不受 NaN 填充影响
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting inference using {device}")

    # 1. Load Data
    t0 = time.time()
    ds = netCDF4.Dataset(nc_path, "r")
    inc_grp = ds.groups["by_incidence_angle"] if "by_incidence_angle" in ds.groups else ds

    bs_var = None
    for v in ['mean_bs', 'raw_mean_bs']:
        if v in inc_grp.variables and len(inc_grp.variables[v].shape) == 2:
            bs_var = v
            break

    if not bs_var:
        ds.close()
        raise ValueError("Could not find BS matrix.")

    bs_data = inc_grp.variables[bs_var][:]
    if hasattr(bs_data, 'filled'):
        bs_data = bs_data.filled(np.nan)
    bs_matrix = np.array(bs_data, dtype=float)

    ang_var = 'angle' if 'angle' in inc_grp.variables else 'incidence_angle'
    ang_data = inc_grp.variables[ang_var][:]
    if hasattr(ang_data, 'filled'):
        ang_data = ang_data.filled(np.nan)
    angles_deg = np.array(ang_data, dtype=float)

    ping_times = ds.variables['ping_time'][:] if 'ping_time' in ds.variables else np.arange(bs_matrix.shape[0])

    ds.close()

    # 保存初始 NaN mask 用于后续 RMSE 计算
    nan_mask_orig = np.isnan(bs_matrix)
    bs_matrix[nan_mask_orig] = -40.0

    if np.min(angles_deg) < 0:
        # 折叠正负角: port/starboard 对称平均
        center_idx = len(angles_deg) // 2
        # NaN mask 也折叠 (any side NaN -> folded NaN)
        nan_folded = nan_mask_orig[:, center_idx:] | nan_mask_orig[:, center_idx-1::-1]
        bs_folded = (bs_matrix[:, center_idx:] + bs_matrix[:, center_idx-1::-1]) / 2.0
        angles_pos = angles_deg[center_idx:]
    else:
        nan_folded = nan_mask_orig.copy()
        bs_folded = bs_matrix.copy()
        angles_pos = angles_deg.copy()

    # 插值到标准 0-70 deg 角度网格
    from scipy import interpolate
    angles_standard = np.arange(0, 71, 1.0)
    if len(angles_pos) != len(angles_standard) or not np.allclose(angles_pos, angles_standard):
        print(f"Interpolating from {len(angles_pos)} angles to {len(angles_standard)} angles (0-70 deg)...")
        f_interp = interpolate.interp1d(angles_pos, bs_folded, axis=1,
                                         kind='linear', bounds_error=False,
                                         fill_value='extrapolate')
        bs_interp = f_interp(angles_standard)
        bs_interp[:, angles_standard < angles_pos.min()] = bs_folded[:, 0:1]
        bs_interp[:, angles_standard > angles_pos.max()] = bs_folded[:, -1:]
        # NaN mask 也插值 (最近邻，保持 NaN 区域)
        f_nan = interpolate.interp1d(angles_pos, nan_folded.astype(float), axis=1,
                                      kind='nearest', bounds_error=False, fill_value=1)
        nan_interp = f_nan(angles_standard) > 0.5
    else:
        bs_interp = bs_folded
        nan_interp = nan_folded

    bs_matrix = bs_interp
    angles_deg = angles_standard
    # valid_mask: True = 该角度有有效数据
    valid_mask = ~nan_interp

    t1 = time.time()
    print(f"Data loading took {t1 - t0:.3f} seconds for {bs_matrix.shape[0]} pings.")

    # 2. Prepare Tensors
    bs_norm = (bs_matrix + 20.0) / 20.0
    theta_norm = angles_deg / 70.0

    theta_norm_expanded = np.tile(theta_norm, (bs_matrix.shape[0], 1))
    x_numpy = np.stack([bs_norm, theta_norm_expanded], axis=1)

    freq_norm_val = (freq_hz - 35000.0) / 415000.0
    freq_numpy = np.full((bs_matrix.shape[0], 1), freq_norm_val)

    x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to(device)
    f_tensor = torch.tensor(freq_numpy, dtype=torch.float32).to(device)
    valid_mask_t = torch.tensor(valid_mask, dtype=torch.bool).to(device)

    # 3. Inference
    model = PINNESABTransformer(seq_len=len(angles_deg)).to(device)
    model.eval()

    physics_model = DifferentiableESAB(gamma=gamma).to(device)
    physics_model.eval()

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print(f"Warning: {weights_path} not found. Using untrained weights.")

    t2 = time.time()
    print("Running PINN Inference...")

    batch_size = 2048
    # 预计算每 ping 的有效角度数 (用于 RMSE 归一化)
    n_valid_total = valid_mask_t.sum(dim=1)

    results = {
        'z': np.zeros(bs_matrix.shape[0]),
        'mu': np.zeros(bs_matrix.shape[0]),
        's1': np.zeros(bs_matrix.shape[0]),
        'delta2': np.zeros(bs_matrix.shape[0]),
        'rmse': np.full(bs_matrix.shape[0], np.nan),
        'n_valid_angles': np.zeros(bs_matrix.shape[0], dtype=np.int32)
    }

    with torch.no_grad():
        for i in range(0, bs_matrix.shape[0], batch_size):
            x_batch = x_tensor[i:i+batch_size]
            f_batch = f_tensor[i:i+batch_size]
            mask_batch = valid_mask_t[i:i+batch_size]
            n_valid_batch = n_valid_total[i:i+batch_size]

            y_pred = model(x_batch, f_batch)

            z_pred = y_pred[:, 0:1] * 13.0 + 1.0
            mu_pred = y_pred[:, 1:2] * 30.0 - 15.0
            s1_pred = y_pred[:, 2:3] * 54.5 + 0.5    # s1 in [0.5, 55.0]
            d2_pred = y_pred[:, 3:4] * 74.0 + 1.0    # delta2 in [1, 75]

            z_pred = torch.clamp(z_pred, 1.0, 14.0)
            mu_pred = torch.clamp(mu_pred, -15.0, 15.0)
            s1_pred = torch.clamp(s1_pred, 0.5, 55.0)
            d2_pred = torch.clamp(d2_pred, 1.0, 75.0)

            # Save parameters
            bs_idx = slice(i, i + x_batch.shape[0])
            results['z'][bs_idx] = z_pred.squeeze().cpu().numpy()
            results['mu'][bs_idx] = mu_pred.squeeze().cpu().numpy()
            results['s1'][bs_idx] = s1_pred.squeeze().cpu().numpy()
            results['delta2'][bs_idx] = d2_pred.squeeze().cpu().numpy()
            results['n_valid_angles'][bs_idx] = n_valid_batch.cpu().numpy()

            # Compute RMSE only on valid (non-NaN) angles
            freq_hz_batch = f_batch * 415000.0 + 35000.0
            theta_deg_batch = x_batch[:, 1, :] * 70.0

            bs_recon = physics_model(theta_deg_batch, freq_hz_batch, z_pred,
                                      s1_pred, d2_pred, mu_pred)
            bs_real = x_batch[:, 0, :] * 20.0 - 20.0

            bs_recon = torch.nan_to_num(bs_recon, nan=-30.0, posinf=10.0, neginf=-60.0)

            # 只在有效角度上计算 MSE
            diff_sq = (bs_recon - bs_real) ** 2
            # 将无效角度的 diff 设为 0 (不参与求和)
            diff_sq_masked = diff_sq * mask_batch.float()
            mse = diff_sq_masked.sum(dim=1) / (n_valid_batch.float() + 1e-8)
            rmse = torch.sqrt(mse)
            # 如果没有有效角度，rmse 保持为 NaN
            rmse[n_valid_batch < 5] = float('nan')
            results['rmse'][bs_idx] = rmse.cpu().numpy()

    t3 = time.time()
    print(f"Inference complete! Processed {bs_matrix.shape[0]} pings in {t3 - t2:.3f} seconds.")
    print(f"Speed: {bs_matrix.shape[0] / (t3 - t2):.1f} pings/sec")

    # 统计有效角度分布
    n_valid_arr = np.array(results['n_valid_angles'])
    print(f"Valid angles per ping: median={np.median(n_valid_arr):.0f}, "
          f"min={n_valid_arr.min():.0f}, max={n_valid_arr.max():.0f}")

    # 4. Save to Xarray
    ds_out = xr.Dataset(
        {
            "z": (["ping_time"], results['z']),
            "mu_db": (["ping_time"], results['mu']),
            "s1": (["ping_time"], results['s1']),
            "delta2": (["ping_time"], results['delta2']),
            "physics_rmse": (["ping_time"], results['rmse']),
            "n_valid_angles": (["ping_time"], results['n_valid_angles'])
        },
        coords={"ping_time": ping_times[:bs_matrix.shape[0]]}
    )

    ds_out.attrs['description'] = "PINN Extracted ESAB Physical Features (RMSE on valid angles only)"
    ds_out.to_netcdf(out_nc_path)
    print(f"Successfully saved features to {out_nc_path}")


if __name__ == "__main__":
    import sys
    default_input = r"D:\Software\xsf\merged_all.bsar.nc"
    default_output = r"D:\Software\xsf\merged_all_pinn_features.nc"

    nc_path = sys.argv[1] if len(sys.argv) > 1 else default_input
    out_path = sys.argv[2] if len(sys.argv) > 2 else default_output
    freq = float(sys.argv[3]) if len(sys.argv) > 3 else 300000.0

    export_yolo_features(
        nc_path=nc_path,
        out_nc_path=out_path,
        freq_hz=freq
    )
