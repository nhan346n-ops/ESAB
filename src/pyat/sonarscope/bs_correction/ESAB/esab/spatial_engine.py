import numpy as np
import xarray as xr
import multiprocessing
import traceback
from functools import partial
import netCDF4
import os

from .esab_inversion import invert_esab_single_freq
from ..gsab_model import GsabDataModel
from ..esab_model import EsabDataModel

def _invert_single_ping(ping_idx, angles, bs_vals, freq_hz):
    """
    Independent worker function for multiprocessing.
    Returns (ping_idx, z, mu_db, s1, delta2, rmse) or (ping_idx, NaNs) on failure.
    """
    valid = (~np.isnan(angles)) & (~np.isnan(bs_vals))
    v_angles = angles[valid]
    v_bs = bs_vals[valid]

    # Needs at least a decent number of points (e.g., > 10) to invert properly
    if len(v_angles) < 10:
        return (ping_idx, np.nan, np.nan, np.nan, np.nan, np.nan)

    try:
        # 1. GSAB Fit
        gsab = GsabDataModel(v_angles, v_bs, np.ones_like(v_angles))
        gsab.fit_gsab()

        # 2. ESAB Inversion
        esab = EsabDataModel(gsab, freq_hz=freq_hz)
        res = esab.fit_esab()

        return (ping_idx, res['z'], res['mu_db'], res['s1'], res['delta2'], res['rmse'])
    except Exception:
        # If inversion fails for this pixel, return NaNs
        return (ping_idx, np.nan, np.nan, np.nan, np.nan, np.nan)

def _unpack_and_run(args):
    return _invert_single_ping(*args)

class SpatialEsabEngine:
    def __init__(self, nc_filepath: str, freq_hz: float, nav_filepath: str = None):
        """
        Loads a sliding-window bsar.nc or an aggregated NetCDF file.
        Expects dimensions like (ping_time, angle) or (ping, angle).
        """
        self.nc_filepath = nc_filepath
        self.freq_hz = freq_hz
        self.angles = None
        self.bs_matrix = None
        self.ping_times = None
        self.lons = None
        self.lats = None
        self.nav_filepath = nav_filepath
        
    def load_data(self):
        """
        Robustly extracts the 2D BS matrix and coordinates.
        """
        # First try to find it via netCDF4 groups (like pyat's tool2a output)
        ds = netCDF4.Dataset(self.nc_filepath, "r")
        inc_grp = None
        
        if "by_incidence_angle" in ds.groups:
            inc_grp = ds.groups["by_incidence_angle"]
        else:
            inc_grp = ds
            
        # Look for 2D mean_bs
        bs_var = None
        for v in ['mean_bs', 'raw_mean_bs']:
            if v in inc_grp.variables and len(inc_grp.variables[v].shape) == 2:
                bs_var = v
                break
                
        if not bs_var:
            ds.close()
            raise ValueError("未在文件中找到二维 (Ping x Angle) 的 BS 矩阵。请确保输入的是 Tool 2A 滑动窗口生成的连续文件。")
            
        bs_data = inc_grp.variables[bs_var][:]
        if hasattr(bs_data, 'filled'):
            bs_data = bs_data.filled(np.nan)
        self.bs_matrix = np.array(bs_data, dtype=float)
        
        ang_var = 'angle' if 'angle' in inc_grp.variables else 'incidence_angle'
        ang_data = inc_grp.variables[ang_var][:]
        if hasattr(ang_data, 'filled'):
            ang_data = ang_data.filled(np.nan)
            
        # Angle might be 1D or 2D. If 1D, broadcast it.
        self.angles = np.array(ang_data, dtype=float)
        if len(self.angles.shape) == 1:
            self.angles = np.tile(self.angles, (self.bs_matrix.shape[0], 1))
            
        # Try to extract ping_time, lon, lat
        if 'ping_time' in inc_grp.variables:
            self.ping_times = np.array(inc_grp.variables['ping_time'][:])
        else:
            self.ping_times = np.arange(self.bs_matrix.shape[0])
            
        if 'longitude' in inc_grp.variables and 'latitude' in inc_grp.variables:
            self.lons = np.array(inc_grp.variables['longitude'][:])
            self.lats = np.array(inc_grp.variables['latitude'][:])
        elif self.nav_filepath and os.path.exists(self.nav_filepath):
            # 借用坐标
            try:
                nav_ds = netCDF4.Dataset(self.nav_filepath, "r")
                
                # 递归查找含有 ping_time 和 latitude 的组
                def find_nav_group(grp):
                    if 'ping_time' in grp.variables and ('latitude' in grp.variables or 'platform_latitude' in grp.variables):
                        return grp
                    for sub_grp in grp.groups.values():
                        res = find_nav_group(sub_grp)
                        if res: return res
                    return None
                    
                target_grp = find_nav_group(nav_ds)
                
                if target_grp:
                    nav_pings = target_grp.variables['ping_time'][:]
                    
                    if 'longitude' in target_grp.variables:
                        nav_lons = target_grp.variables['longitude'][:]
                        nav_lats = target_grp.variables['latitude'][:]
                    else:
                        nav_lons = target_grp.variables['platform_longitude'][:]
                        nav_lats = target_grp.variables['platform_latitude'][:]
                        
                    # 按照 ping_time 插值坐标
                    self.lons = np.interp(self.ping_times, nav_pings, nav_lons)
                    self.lats = np.interp(self.ping_times, nav_pings, nav_lats)
                else:
                    raise ValueError("导航文件中没有 ping_time 和 经纬度变量")
                    
                nav_ds.close()
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Warning: Failed to borrow coordinates from nav file: {e}")
                self.lons = np.linspace(0, 1, self.bs_matrix.shape[0])
                self.lats = np.linspace(0, 1, self.bs_matrix.shape[0])
        else:
            # If no nav in bsar, we just use a fake grid or ping indices
            self.lons = np.linspace(0, 1, self.bs_matrix.shape[0])
            self.lats = np.linspace(0, 1, self.bs_matrix.shape[0])
            
        ds.close()
        
    def run_batch_inversion(self, num_cores=None, progress_callback=None):
        """
        Runs the ESAB inversion in parallel across all pings.
        """
        if self.bs_matrix is None:
            self.load_data()
            
        num_pings = self.bs_matrix.shape[0]
        if num_cores is None:
            num_cores = max(1, multiprocessing.cpu_count() - 1)
            
        results = {
            'z': np.full(num_pings, np.nan),
            'mu': np.full(num_pings, np.nan),
            's1': np.full(num_pings, np.nan),
            'delta2': np.full(num_pings, np.nan),
            'rmse': np.full(num_pings, np.nan)
        }

        # We need to process sequentially if using a progress callback via standard Pool.imap
        # To avoid GUI freeze, we can use an iterator.
        pool = multiprocessing.Pool(processes=num_cores)

        tasks = []
        for i in range(num_pings):
            tasks.append((i, self.angles[i], self.bs_matrix[i], self.freq_hz))

        completed = 0
        for res in pool.imap_unordered(_unpack_and_run, tasks):
            idx, z, mu, s1_val, d2, rmse = res
            results['z'][idx] = z
            results['mu'][idx] = mu
            results['s1'][idx] = s1_val
            results['delta2'][idx] = d2
            results['rmse'][idx] = rmse
            
            completed += 1
            if progress_callback:
                progress_callback(int(completed / num_pings * 100))
                
        pool.close()
        pool.join()
        
        self.results = results
        return results
        
    def save_grid_to_netcdf(self, output_path: str, grid_res: float = 0.0001):
        """
        Saves the resulting arrays into a NetCDF4 file structured like a DTM.
        This file can be directly read by pyat-main's dtm_renderer.py.
        grid_res is in degrees (approx 10m).
        """
        from scipy.interpolate import griddata
        
        if not hasattr(self, 'results'):
            raise RuntimeError("必须先运行 run_batch_inversion()")
            
        # 1. 提取有效点
        valid = (~np.isnan(self.lons)) & (~np.isnan(self.lats)) & (~np.isnan(self.results['z']))
        pts = np.column_stack((self.lons[valid], self.lats[valid]))
        
        if len(pts) == 0:
            raise ValueError("没有有效的反演结果可以网格化！")
            
        z_vals = self.results['z'][valid]
        mu_vals = self.results['mu'][valid]
        s1_vals = self.results['s1'][valid]
        d2_vals = self.results['delta2'][valid]
        rmse_vals = self.results['rmse'][valid]
        
        # 2. 生成规则网格
        is_fake_grid = np.all(self.lons == self.lats)
        
        # Grid parameters
        if is_fake_grid:
            # Just save it as a 1D sequence without gridding
            ds = xr.Dataset(
                {
                    "z": (["ping_time"], z_vals),
                    "mu": (["ping_time"], mu_vals),
                    "s1": (["ping_time"], s1_vals),
                    "delta2": (["ping_time"], d2_vals),
                    "rmse": (["ping_time"], rmse_vals),
                },
                coords={"ping_time": self.ping_times[valid]}
            )
            ds.attrs["description"] = "ESAB Spatial Inversion Result (1D Sequential)"
            ds.to_netcdf(output_path)
            return output_path
            
        lon_min, lon_max = np.min(self.lons[valid]), np.max(self.lons[valid])
        lat_min, lat_max = np.min(pts[:, 1]), np.max(pts[:, 1])
        
        lon_min -= grid_res; lon_max += grid_res
        lat_min -= grid_res; lat_max += grid_res
        
        # 避免网格过大崩溃
        if (lon_max - lon_min) / grid_res > 2000 or (lat_max - lat_min) / grid_res > 2000:
            grid_res = max((lon_max - lon_min)/1000, (lat_max - lat_min)/1000)
            
        grid_lon, grid_lat = np.meshgrid(
            np.arange(lon_min, lon_max, grid_res),
            np.arange(lat_min, lat_max, grid_res)
        )
        
        # 3. 插值
        grid_z = griddata(pts, z_vals, (grid_lon, grid_lat), method='linear')
        grid_mu = griddata(pts, mu_vals, (grid_lon, grid_lat), method='linear')
        grid_s1 = griddata(pts, s1_vals, (grid_lon, grid_lat), method='linear')
        grid_d2 = griddata(pts, d2_vals, (grid_lon, grid_lat), method='linear')

        # 4. 构建 Dataset
        ds = xr.Dataset(
            {
                "z": (["lat", "lon"], grid_z),
                "mu_db": (["lat", "lon"], grid_mu),
                "s1": (["lat", "lon"], grid_s1),
                "delta2": (["lat", "lon"], grid_d2)
            },
            coords={
                "lon": grid_lon[0, :],
                "lat": grid_lat[:, 0]
            }
        )
        
        ds.attrs['title'] = "ESAB Spatial Inversion DTM"
        ds.to_netcdf(output_path)
        return output_path
