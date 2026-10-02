"""
run_source_holdout.py
---------------------
Master runner for Phase 10: Leave-One-Source-Out (LOSO) cross-validation.
"""
import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from experiments.source_shift.run_source_holdout import run_source_holdout

if __name__ == "__main__":
    run_source_holdout()
