import ast
import os

def get_ast_info(filepath):
    if not os.path.exists(filepath):
        return set(), set()
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return set(), set()
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    return classes, functions

def compare_modules(module_path_rel):
    dir1 = os.path.join('C:/Users/GUO/Desktop/BS-process', module_path_rel)
    dir2 = os.path.join('C:/Users/GUO/Desktop/pyat-main', module_path_rel)
    
    if not os.path.exists(dir2):
        print(f"Module {module_path_rel} not found in pyat-main")
        return

    added_classes_total = {}
    added_functions_total = {}
    
    for root, _, files in os.walk(dir2):
        for f in files:
            if not f.endswith('.py'):
                continue
            path2 = os.path.join(root, f)
            rel_path = os.path.relpath(path2, dir2)
            path1 = os.path.join(dir1, rel_path)
            
            classes1, funcs1 = get_ast_info(path1)
            classes2, funcs2 = get_ast_info(path2)
            
            added_classes = classes2 - classes1
            added_funcs = funcs2 - funcs1
            
            if added_classes:
                added_classes_total[rel_path] = added_classes
            if added_funcs:
                added_functions_total[rel_path] = added_funcs

    print(f"--- Analysis for {module_path_rel} ---")
    if added_classes_total:
        print("New Classes:")
        for f, cls in added_classes_total.items():
            print(f"  {f}: {', '.join(cls)}")
    if added_functions_total:
        print("New Functions:")
        for f, func in added_functions_total.items():
            print(f"  {f}: {', '.join(func)}")
    print()

modules_to_check = [
    'src/pyat/dtm/numba',
    'src/pyat/dtm/transform/interpolation/coronis',
    'src/pyat/xsf/bathy',
    'src/pyat/sonarscope/bs_correction'
]

for m in modules_to_check:
    compare_modules(m)
