import os

files_to_fix = [
    r'C:\Users\GUO\Desktop\BS-process\gui\core\json_builder.py',
    r'C:\Users\GUO\Desktop\BS-process\gui\dialogs\sounder_to_dtm_wizard.py',
    r'C:\Users\GUO\Desktop\BS-process\gui\main_window.py'
]

for filepath in files_to_fix:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content.replace('valid_sounds_only', 'valid_soundings_only')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"Fixed {filepath}")
