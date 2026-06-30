"""
PINN + SA 混合反演流水线 (方案A)

策略:
  - 95% pings (RMSE <= 7.72 dB): 取 PINN 结果，直接出
  -  5% pings (RMSE >  7.72 dB): 用 PINN 初值跑加速 SA (3cyc×2000it)
  - 合并输出，100% 全覆盖

时间估计:
  PINN推理:  ~10s (已完成)
  SA 精修:   ~3.2h (30K pings, 15核并行)
  ─────────────────
  总计:      ~3.2h

用法:
    python pinn_sa_hybrid.py
"""
import numpy as np
import xarray as xr
import netCDF4
import multiprocessing
import time
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from esab_inversion import invert_esab_single_freq


# ===== 配置 =====
BSAR_PATH     = r"D:\Software\xsf\merged_all.bsar.nc"
PINN_PATH     = r"D:\Software\xsf\merged_all_pinn_features.nc"
OUTPUT_PATH   = r"D:\Software\xsf\merged_all_hybrid.nc"
FREQ_HZ       = 300000.0
RMSE_THRESH   = 7.72    # P95 threshold (worst 5%)
SA_CYCLES     = 3
SA_ITERS      = 2000
SA_STEP_SCALE = 0.5
NUM_CORES     = 15


def sa_worker(args):
    """单 ping SA 工作函数 (多进程)."""
    idx, angles_deg, bs_vals, init_guess = args
    valid = ~np.isnan(angles_deg) & ~np.isnan(bs_vals)
    angles_v = angles_deg[valid]
    bs_v = bs_vals[valid]

    if len(angles_v) < 10:
        return (idx, np.nan, np.nan, np.nan, np.nan, np.nan)

    try:
        result = invert_esab_single_freq(
            angles_v, bs_v, FREQ_HZ,
            initial_guess=init_guess,
            n_cycles=SA_CYCLES,
            iters_per_cycle=SA_ITERS,
            step_scale=SA_STEP_SCALE
        )
        return (idx, result['z'], result['mu_db'],
                result['s1'], result['delta2'], result['rmse'])
    except Exception:
        return (idx, np.nan, np.nan, np.nan, np.nan, np.nan)


def main():
    t0 = time.time()
    print("=" * 60)
    print("PINN + SA 混合反演流水线")
    print("=" * 60)
    print(f"SA config: {SA_CYCLES} cycles, {SA_ITERS} iter/cycle, step_scale={SA_STEP_SCALE}")
    print(f"RMSE threshold: {RMSE_THRESH:.2f} dB (worst ~5%)")
    print(f"Cores: {NUM_CORES}")
    print()

    # 1. 加载 PINN 特征
    print("[1/5] Loading PINN features...")
    dp = xr.open_dataset(PINN_PATH)
    pinn_z  = np.array(dp['z'].values, dtype=float)
    pinn_mu = np.array(dp['mu_db'].values, dtype=float)
    pinn_s1 = np.array(dp['s1'].values, dtype=float)
    pinn_d2 = np.array(dp['delta2'].values, dtype=float)
    pinn_rm = np.array(dp['physics_rmse'].values, dtype=float)
    n_valid = np.array(dp['n_valid_angles'].values, dtype=int)
    ping_times = np.array(dp['ping_time'].values)
    dp.close()
    n_pings = len(pinn_z)
    print(f"  {n_pings} pings loaded")

    # 2. 加载 BSAR 数据 (SA 需要原始 BS 曲线)
    print("[2/5] Loading BSAR data...")
    ds = netCDF4.Dataset(BSAR_PATH, "r")
    grp = ds.groups["by_incidence_angle"]
    angles = grp['angle'][:]
    bs_var = 'mean_bs' if 'mean_bs' in grp.variables else 'raw_mean_bs'
    bs_full = np.array(grp[bs_var][:], dtype=float)
    if hasattr(bs_full, 'filled'):
        bs_full = bs_full.filled(np.nan)

    # 折叠正负角
    center = len(angles) // 2
    bs_folded = (bs_full[:, center:] + bs_full[:, center-1::-1]) / 2.0
    angles_pos = angles[center:]
    ds.close()
    print(f"  BS matrix: {bs_full.shape} -> folded: {bs_folded.shape}")

    # 3. 决定哪些 ping 需要 SA
    print(f"\n[3/5] Classifying pings...")
    valid_for_sa = ~np.isnan(pinn_rm) & (n_valid >= 10)
    need_sa = valid_for_sa & (pinn_rm > RMSE_THRESH)
    easy_pings = valid_for_sa & ~need_sa

    print(f"  Easy (PINN only):  {easy_pings.sum()} ({easy_pings.sum()/n_pings*100:.1f}%)")
    print(f"  Hard (need SA):    {need_sa.sum()} ({need_sa.sum()/n_pings*100:.1f}%)")
    print(f"  Too few angles:    {(~valid_for_sa).sum()}")

    # 4. 初始化结果数组 (先填 PINN 结果)
    print(f"\n[4/5] Initializing results with PINN values...")
    results = {
        'z': pinn_z.copy(),
        'mu': pinn_mu.copy(),
        's1': pinn_s1.copy(),
        'delta2': pinn_d2.copy(),
        'sa_rmse': np.full(n_pings, np.nan),
        'source': np.zeros(n_pings, dtype=np.int32)  # 0=PINN, 1=SA
    }
    # Easy pings: source=0 (default), keep PINN values
    # Need SA: will be overwritten

    # 5. 跑 SA
    sa_count = need_sa.sum()
    if sa_count > 0:
        print(f"\n[5/5] Running accelerated SA on {sa_count} hard pings...")
        sa_indices = np.where(need_sa)[0]
        tasks = []
        for idx in sa_indices:
            init_guess = [float(pinn_z[idx]), float(pinn_mu[idx]),
                         float(pinn_s1[idx]), float(pinn_d2[idx])]
            tasks.append((idx, angles_pos, bs_folded[idx], init_guess))

        t_sa = time.time()
        pool = multiprocessing.Pool(processes=NUM_CORES)
        completed = 0

        for res in pool.imap_unordered(sa_worker, tasks):
            idx, z, mu, s1, d2, rmse = res
            if not np.isnan(z):
                results['z'][idx] = z
                results['mu'][idx] = mu
                results['s1'][idx] = s1
                results['delta2'][idx] = d2
                results['sa_rmse'][idx] = rmse
                results['source'][idx] = 1
            completed += 1
            if completed % 1000 == 0 or completed == sa_count:
                elapsed = time.time() - t_sa
                rate = completed / elapsed
                eta = (sa_count - completed) / rate if rate > 0 else 0
                print(f"  SA: {completed}/{sa_count} ({completed/sa_count*100:.0f}%) "
                      f"| {rate:.1f} ping/s | ETA: {eta/60:.1f} min")

        pool.close()
        pool.join()
        print(f"  SA completed in {(time.time()-t_sa)/60:.1f} min")
    else:
        print(f"\n[5/5] No SA needed - all pings within threshold!")

    # 6. 保存
    print(f"\nSaving to {OUTPUT_PATH}...")
    ds_out = xr.Dataset({
        "z": (["ping_time"], results['z']),
        "mu_db": (["ping_time"], results['mu']),
        "s1": (["ping_time"], results['s1']),
        "delta2": (["ping_time"], results['delta2']),
        "sa_rmse": (["ping_time"], results['sa_rmse']),
        "source": (["ping_time"], results['source']),
    }, coords={"ping_time": ping_times[:n_pings]})
    ds_out.attrs['description'] = "PINN+SA Hybrid Inversion (95% PINN, 5% SA)"
    ds_out.attrs['sa_config'] = f"cycles={SA_CYCLES}, iters={SA_ITERS}, step_scale={SA_STEP_SCALE}"
    ds_out.attrs['rmse_threshold'] = f"{RMSE_THRESH:.2f} dB"
    ds_out.attrs['frequency_hz'] = str(FREQ_HZ)
    ds_out.to_netcdf(OUTPUT_PATH)

    # 7. 总结
    total = time.time() - t0
    sa_done = results['source'] == 1
    print(f"\n{'='*60}")
    print(f"完成! 总耗时: {total/60:.1f} 分钟")
    print(f"{'='*60}")
    print(f"  PINN:   {easy_pings.sum()} pings ({easy_pings.sum()/n_pings*100:.1f}%)")
    print(f"  SA:     {sa_done.sum()} pings ({sa_done.sum()/n_pings*100:.1f}%)")
    sa_rmse = results['sa_rmse'][sa_done]
    print(f"  SA rmse: median={np.nanmedian(sa_rmse):.3f} dB")

    print(f"\n  Final parameter distribution:")
    for name in ['z', 'mu', 's1', 'delta2']:
        vals = results[name][valid_for_sa]
        print(f"    {name}: median={np.median(vals):.3f} "
              f"[{np.percentile(vals,5):.3f}, {np.percentile(vals,95):.3f}]")

    return OUTPUT_PATH


if __name__ == "__main__":
    main()
