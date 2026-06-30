import os

base_dir = r"C:\Users\GUO\Desktop\BS-process\resources\svg"
os.makedirs(base_dir, exist_ok=True)

dummy_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
  <circle cx="12" cy="12" r="10" fill="#cccccc" />
</svg>"""

files = [
    "data_export.svg",
    "backscatter.svg",
    "bias_correction.svg",
    "auto_bias.svg"
]

for f in files:
    with open(os.path.join(base_dir, f), 'w') as file:
        file.write(dummy_svg)

print("Dummy SVG files created.")
