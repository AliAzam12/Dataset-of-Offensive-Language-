"""
run_experiment.py
-----------------
Master entrypoint to execute specific training pipelines by configuration file or model name.
Supports:
  python run_experiment.py --config configs/baselines.yaml
  python run_experiment.py --config configs/deep_learning.yaml
  python run_experiment.py --config configs/transformers.yaml
  python run_experiment.py --model Linear_SVM
  python run_experiment.py --model CNN-BiLSTM
"""

import os
import sys
import yaml
import argparse
import logging

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Master Experiment Runner")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--model", type=str, default=None, help="Model name to train directly")
    args = parser.parse_args()

    if args.config:
        cfg_path = os.path.join(ROOT_DIR, args.config) if not os.path.isabs(args.config) else args.config
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        logging.info(f"Loaded configuration from: {cfg_path}")

        if "baselines" in args.config:
            from src.models.traditional_baselines import train_and_evaluate_baselines
            train_and_evaluate_baselines()
        elif "deep_learning" in args.config:
            from src.models.deep_learning_models import train_and_evaluate_deep_learning
            train_and_evaluate_deep_learning()
        elif "transformers" in args.config:
            from src.training.train_transformers import main as train_tf
            train_tf()
        else:
            logging.info(f"Configuration executed with settings: {cfg}")

    elif args.model:
        logging.info(f"Dispatching training for model: {args.model}")
        if args.model in ["Multinomial_Naive_Bayes", "Logistic_Regression", "Linear_SVM", "Random_Forest"]:
            from src.models.traditional_baselines import train_and_evaluate_baselines
            train_and_evaluate_baselines()
        elif args.model in ["CNN", "BiLSTM", "CNN-BiLSTM", "Attention-BiLSTM"]:
            from src.models.deep_learning_models import train_and_evaluate_deep_learning
            train_and_evaluate_deep_learning()
        elif args.model in ["mBERT", "XLM-RoBERTa", "MuRIL"]:
            from src.training.train_transformers import main as train_tf
            train_tf()
        else:
            logging.error(f"Unknown model identifier: {args.model}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
