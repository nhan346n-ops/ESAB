import os
import shutil
import filecmp

def backup_project():
    src_dir = 'C:/Users/GUO/Desktop/BS-process/src/pyat'
    backup_dir = 'C:/Users/GUO/Desktop/BS-process/scratch/pyat_backup'
    if not os.path.exists(backup_dir):
        shutil.copytree(src_dir, backup_dir)
        print("Backup completed.")
    else:
        print("Backup already exists.")

def sync_optimizations():
    dir1 = 'C:/Users/GUO/Desktop/BS-process/src/pyat'
    dir2 = 'C:/Users/GUO/Desktop/pyat-main/src/pyat'
    
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
                
        # Also include any new files in pyat-main (right_only)
        for f in dcmp.right_only:
            real_diffs.append(os.path.join(current_path, f))
                
        for name, sub_dcmp in dcmp.subdirs.items():
            if name in ['.git', '__pycache__', '.pytest_cache', 'build', 'gui', 'requirements']:
                continue
            process_dcmp(sub_dcmp, os.path.join(current_path, name))

    process_dcmp(dcmp)
    
    print(f"Syncing {len(real_diffs)} files...")
    for rel_path in real_diffs:
        src_path = os.path.join(dir2, rel_path)
        dst_path = os.path.join(dir1, rel_path)
        
        # Ensure destination dir exists
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        # Copy file
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Synced: {rel_path}")
        elif os.path.isdir(src_path):
            print(f"Skipping dir sync for now: {rel_path}")

backup_project()
sync_optimizations()
print("Done.")
