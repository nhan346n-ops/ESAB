import filecmp
import os
import json

def compare_dirs(dir1, dir2):
    dcmp = filecmp.dircmp(dir1, dir2)
    
    result = {
        'left_only': [], # in dir1 only (BS-process)
        'right_only': [], # in dir2 only (pyat-main)
        'diff_files': [], # in both but different
        'common_dirs': []
    }
    
    def process_dcmp(dcmp, current_path=""):
        for f in dcmp.left_only:
            result['left_only'].append(os.path.join(current_path, f))
        for f in dcmp.right_only:
            result['right_only'].append(os.path.join(current_path, f))
        for f in dcmp.diff_files:
            result['diff_files'].append(os.path.join(current_path, f))
        
        for name, sub_dcmp in dcmp.subdirs.items():
            if name in ['.git', '__pycache__', '.pytest_cache', 'build']:
                continue
            process_dcmp(sub_dcmp, os.path.join(current_path, name))

    process_dcmp(dcmp)
    
    with open('C:/Users/GUO/Desktop/BS-process/scratch/compare_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)

compare_dirs('C:/Users/GUO/Desktop/BS-process', 'C:/Users/GUO/Desktop/pyat-main')
print("Comparison done.")
