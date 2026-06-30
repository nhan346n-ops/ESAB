import difflib
import os

files_to_diff = [
    'gsab_model.py',
    'angular_renormalization.py',
    'sliding_angular_renormalization.py',
    'mean_bs_model.py',
    'mean_bs_processes.py',
    'seafloor_bs_angular_model.py'
]

dir1 = 'C:/Users/GUO/Desktop/BS-process/src/pyat/sonarscope/bs_correction'
dir2 = 'C:/Users/GUO/Desktop/pyat-main/src/pyat/sonarscope/bs_correction'

for f in files_to_diff:
    p1 = os.path.join(dir1, f)
    p2 = os.path.join(dir2, f)
    
    if os.path.exists(p1) and os.path.exists(p2):
        with open(p1, 'r', encoding='utf-8', errors='ignore') as file1:
            lines1 = file1.read().splitlines()
        with open(p2, 'r', encoding='utf-8', errors='ignore') as file2:
            lines2 = file2.read().splitlines()
            
        diff = list(difflib.unified_diff(lines1, lines2, fromfile=f"BS-process/{f}", tofile=f"pyat-main/{f}", n=3))
        
        if diff:
            print(f"========== DIFF FOR {f} ==========")
            for line in diff:
                print(line)
            print("\n")
        else:
            print(f"========== NO DIFFERENCE for {f} (except CRLF) ==========\n")
