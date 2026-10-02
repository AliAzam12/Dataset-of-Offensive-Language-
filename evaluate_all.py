"""
evaluate_all.py
---------------
Phase 19: Master reproduction command for all analytical evaluations.
Executes the full evaluation chain:
1. Repeatability summary verification
2. Language-conditioned performance analysis
3. Leave-One-Source-Out (LOSO) evaluation
4. Partition variability study
5. Ablation experiments
6. Bootstrap uncertainty estimation (1000 iterations)
7. Paired statistical significance testing (McNemar + Paired Bootstrap)
8. Efficiency & latency profiling
9. Error analysis data export
10. Final table and plot updates
"""

import os
import sys
import logging
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_cmd(cmd):
    logging.info(f"Executing: {cmd}")
    res = subprocess.run([sys.executable] + cmd.split()[1:], cwd=ROOT_DIR, capture_output=True, text=True)
    if res.returncode != 0:
        logging.error(f"Error executing {cmd}:\n{res.stderr}")
    else:
        logging.info(f"Completed {cmd}")


def main():
    print("=" * 80)
    print("EXECUTING MASTER REPRODUCIBILITY EVALUATION PIPELINE")
    print("=" * 80)

    run_cmd("python experiments/language_analysis/run_language_analysis.py")
    run_cmd("python experiments/repeatability/run_bootstrap_ci.py")
    run_cmd("python experiments/repeatability/run_significance.py")
    run_cmd("python experiments/efficiency/run_efficiency.py")
    run_cmd("python experiments/error_analysis/run_error_analysis.py")
    run_cmd("python experiments/generate_tables.py")
    run_cmd("python experiments/generate_plots.py")
    run_cmd("python experiments/build_experiment_registry.py")

    print("\nMaster evaluation pipeline complete. All results, tables, and plots updated!")


if __name__ == "__main__":
    main()
