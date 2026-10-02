"""
run_ablations.py
----------------
Master runner for Phase 12: Ablation experiments.
"""
import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from experiments.ablations.run_ablations import run_ablations

if __name__ == "__main__":
    run_ablations()
