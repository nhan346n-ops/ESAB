r"""pyat GUI launcher.

Usage:
    python run_gui.py
"""
import sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")

# Ensure project root and src are on path
for p in [ROOT, SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.chdir(ROOT)

from gui.app import main
main()
