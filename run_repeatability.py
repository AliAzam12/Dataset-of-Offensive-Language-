"""
run_repeatability.py
--------------------
Master runner for Phase 8: 5-seed repeatability experiment.
"""
import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from experiments.repeatability.run_repeatability import run_repeatability_study

if __name__ == "__main__":
    run_repeatability_study()
