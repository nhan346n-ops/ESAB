import os
import filecmp

def compare_dirs(dir1, dir2):
    dcmp = filecmp.dircmp(dir1, dir2)
    
    real_diffs = []
    
    def process_dcmp(dcmp, current_path=""):
        for f in dcmp.diff_files:
            p1 = os.path.join(dcmp.left, f)
            p2 = os.path.join(dcmp.right, f)
            
            with open(p1, 'r', encoding='utf-8', errors='ignore') as file1:
                lines1 = file1.read().splitlines()
            with open(p2, 'r', encoding='utf-8', errors='ignore') as file2:
                lines2 = file2.read().splitlines()
                
            if lines1 != lines2:
                real_diffs.append(os.path.join(current_path, f))
                
        for name, sub_dcmp in dcmp.subdirs.items():
            if name in ['.git', '__pycache__', '.pytest_cache', 'build']:
                continue
            process_dcmp(sub_dcmp, os.path.join(current_path, name))

    process_dcmp(dcmp)
    
    print(f"Total semantically different files (ignoring line endings): {len(real_diffs)}")
    for f in real_diffs:
        print(f" - {f}")

compare_dirs('C:/Users/GUO/Desktop/BS-process/src/pyat', 'C:/Users/GUO/Desktop/pyat-main/src/pyat')
