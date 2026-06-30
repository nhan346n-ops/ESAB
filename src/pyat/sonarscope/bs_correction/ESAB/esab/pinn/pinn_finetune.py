import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import netCDF4
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pinn_model import PINNESABTransformer
from pinn_trainer import PINNLoss


def finetune_real_data(nc_path, pretrained_weights="pinn_esab_pretrained.pth", freq_hz=300000.0, gamma=10/3):
    """
    在真实数据上微调 PINN (纯自监督: 仅使用物理损失).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Fine-tuning on real data using {device}")

    # 1. 加载真实数据
    print(f"Loading {nc_path}...")
    ds = netCDF4.Dataset(nc_path, "r")

    if "by_incidence_angle" in ds.groups:
        inc_grp = ds.groups["by_incidence_angle"]
    else:
        inc_grp = ds

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

    ds.close()

    # Handle NaNs
    valid_pings = ~np.isnan(bs_matrix).all(axis=1)
    bs_matrix = bs_matrix[valid_pings]

    nan_mask = np.isnan(bs_matrix)
    bs_matrix[nan_mask] = -40.0

    if np.min(angles_deg) < 0:
        center_idx = len(angles_deg) // 2
        bs_folded = (bs_matrix[:, center_idx:] + bs_matrix[:, center_idx-1::-1]) / 2.0
        angles_pos = angles_deg[center_idx:]
    else:
        bs_folded = bs_matrix.copy()
        angles_pos = angles_deg.copy()

    # 插值到标准 0-70 deg 角度网格 (与推理时一致)
    from scipy import interpolate
    angles_standard = np.arange(0, 71, 1.0)
    if len(angles_pos) != len(angles_standard):
        f_interp = interpolate.interp1d(angles_pos, bs_folded, axis=1,
                                         kind='linear', bounds_error=False,
                                         fill_value='extrapolate')
        bs_interp = f_interp(angles_standard)
    else:
        bs_interp = bs_folded

    bs_matrix = bs_interp
    angles_deg = angles_standard

    print(f"Loaded {bs_matrix.shape[0]} valid pings, {len(angles_deg)} angles.")

    # 2. 准备张量
    bs_norm = (bs_matrix + 20.0) / 20.0
    theta_norm = angles_deg / 70.0

    theta_norm_expanded = np.tile(theta_norm, (bs_matrix.shape[0], 1))
    x_numpy = np.stack([bs_norm, theta_norm_expanded], axis=1)

    freq_norm_val = (freq_hz - 35000.0) / 415000.0
    freq_numpy = np.full((bs_matrix.shape[0], 1), freq_norm_val)

    x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
    f_tensor = torch.tensor(freq_numpy, dtype=torch.float32)

    dataset = TensorDataset(x_tensor, f_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    # 3. 模型设置
    model = PINNESABTransformer(seq_len=len(angles_deg)).to(device)
    if os.path.exists(pretrained_weights):
        model.load_state_dict(torch.load(pretrained_weights, map_location=device))
        print("Loaded pre-trained weights.")
    else:
        print("Warning: Pre-trained weights not found, starting from scratch!")

    criterion = PINNLoss(lambda_physics=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # 4. 微调循环 (纯自监督)
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_phys = 0

        pbar = tqdm(loader, desc=f"Fine-tuning {epoch+1}/{epochs}")
        for x, freq in pbar:
            x, freq = x.to(device), freq.to(device)
            dummy_y = torch.zeros(x.size(0), 4).to(device)

            optimizer.zero_grad()
            y_pred = model(x, freq)

            _, l_data, l_phys = criterion(y_pred, dummy_y, x, freq)

            # 数值保护: 物理损失若为 NaN 则跳过此 batch
            if torch.isnan(l_phys) or torch.isinf(l_phys):
                print("  [Warning] NaN/Inf in physics loss, skipping batch")
                continue

            l_phys.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_phys += l_phys.item()
            pbar.set_postfix({'Phys_Loss': f"{l_phys.item():.4f}"})

        print(f"Epoch {epoch+1} Avg Physics Loss: {total_phys/len(loader):.4f}")

    torch.save(model.state_dict(), "pinn_esab_finetuned.pth")
    print("Fine-tuning completed. Model saved to pinn_esab_finetuned.pth")


if __name__ == "__main__":
    finetune_real_data(r"D:\Software\test\55.nc")
