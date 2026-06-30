"""Pure Python fallback implementation for dtm2ascii export."""
import math

def export_xyz(output_path, export_missing_values, array_longitude, array_latitude, array_depth, column_separator, column_order):
    if isinstance(output_path, bytes):
        output_path = output_path.decode('utf-8')
        
    y_max, x_max = array_depth.shape
    
    # column_order mapping:
    # XYZ = 0, XZY = 1, YXZ = 2, YZX = 3, ZXY = 4, ZYX = 5
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for y in range(y_max):
            lat = array_latitude[y]
            for x in range(x_max):
                lon = array_longitude[x]
                elev = array_depth[y, x]
                
                is_nan_elev = math.isnan(elev)
                if is_nan_elev and not export_missing_values:
                    continue
                    
                if is_nan_elev:
                    if column_order in (0, 1, 4): # XYZ, XZY, ZXY -> lon first
                        f.write(f"{lon:.8f}{column_separator}{lat:.8f}\n")
                    else: # YXZ, YZX, ZYX -> lat first
                        f.write(f"{lat:.8f}{column_separator}{lon:.8f}\n")
                else:
                    if column_order == 0: # XYZ
                        f.write(f"{lon:.8f}{column_separator}{lat:.8f}{column_separator}{elev:.2f}\n")
                    elif column_order == 1: # XZY
                        f.write(f"{lon:.8f}{column_separator}{elev:.2f}{column_separator}{lat:.8f}\n")
                    elif column_order == 2: # YXZ
                        f.write(f"{lat:.8f}{column_separator}{lon:.8f}{column_separator}{elev:.2f}\n")
                    elif column_order == 3: # YZX
                        f.write(f"{lat:.8f}{column_separator}{elev:.2f}{column_separator}{lon:.8f}\n")
                    elif column_order == 4: # ZXY
                        f.write(f"{elev:.2f}{column_separator}{lon:.8f}{column_separator}{lat:.8f}\n")
                    elif column_order == 5: # ZYX
                        f.write(f"{elev:.2f}{column_separator}{lat:.8f}{column_separator}{lon:.8f}\n")

def export_emo(output_path, export_missing_values, array_longitude, array_latitude, array_depth,
               mins, max, stdev_variable, value_count, interpolation_flag, smoothed_depth,
               cdi_index, cdi_values, cprd_values):
    if isinstance(output_path, bytes):
        output_path = output_path.decode('utf-8')
        
    y_max, x_max = array_depth.shape
    max_cdi_index = len(cdi_values) if cdi_values is not None else 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for y in range(y_max):
            lat = array_latitude[y]
            for x in range(x_max):
                lon = array_longitude[x]
                elevation = array_depth[y, x]
                
                is_nan_elev = math.isnan(elevation)
                if is_nan_elev:
                    if export_missing_values:
                        f.write(f"{lon:.8f};{lat:.8f};;;;;;;;;;;\n")
                else:
                    f.write(f"{lon:.8f};{lat:.8f};")
                    
                    if max is not None:
                        val = max[y, x]
                        if not math.isnan(val):
                            f.write(f"{-val:.2f}")
                    f.write(";")
                    
                    if mins is not None:
                        val = mins[y, x]
                        if not math.isnan(val):
                            f.write(f"{-val:.2f}")
                    f.write(";")
                    
                    f.write(f"{-elevation:.2f};")
                    
                    if stdev_variable is not None:
                        val = stdev_variable[y, x]
                        if not math.isnan(val):
                            f.write(f"{val:.2f}")
                    f.write(";")
                    
                    if value_count is not None:
                        val = value_count[y, x]
                        if val >= 0:
                            f.write(f"{val}")
                    f.write(";")
                    
                    if interpolation_flag is not None:
                        val = interpolation_flag[y, x]
                        if val != 0 and val != 127 and val >= 0:
                            f.write(f"{val}")
                    f.write(";")
                    
                    if smoothed_depth is not None:
                        val = smoothed_depth[y, x]
                        if not math.isnan(val):
                            f.write(f"{-val:.2f}")
                        f.write(";")
                        if not math.isnan(val):
                            f.write(f"{abs(elevation - val):.2f}")
                    else:
                        f.write(";")
                    f.write(";")
                    
                    if cdi_index is not None:
                        cdi_idx_val = cdi_index[y, x]
                        if 0 <= cdi_idx_val < max_cdi_index:
                            f.write(cdi_values[cdi_idx_val])
                            f.write(";")
                            f.write(cprd_values[cdi_idx_val])
                        else:
                            f.write(";")
                    else:
                        f.write(";")
                    
                    f.write(";\n")
