"""
eval_only.py: Resume evaluation and result saving after training is already done.
"""

import time
import yaml
import logging
from pathlib import Path

import torch

from utils.tools import (
    parse_args, load_config, overwrite_config,
    setup_logging, set_seed, save_results
)
from data_loader import get_dataloader
from models import get_model
from exps.evaluate import evaluate_model
# from exps.evaluate_tree_region import evaluate_model


def main():
    st_time = time.time()

    # Parse args & config
    args = parse_args()
    cfg = load_config(args.config)
    if args.overwrite_cfg:
        cfg = overwrite_config(args, cfg)

    exp_id = cfg['experiment']['id'].zfill(3)
    exp_name = f"exp{exp_id}_{cfg['model']['name']}"

    # Logger
    logger = setup_logging(cfg['logging']['log_dir'], exp_name)
    logger.info(f"Resume evaluation for experiment: {exp_name}")

    # Seed
    set_seed(cfg['experiment']['seed'])

    # Only need test loader
    _, val_loader, test_loader = get_dataloader(cfg)

    # Build model (weights will be loaded inside evaluate_model)
    model = get_model(cfg['model'])

    # Run evaluation
    eval_metrics = evaluate_model(
        model,
        test_loader,
        cfg,
        exp_name
    )

    # Save results
    results_root = Path(cfg['output']['results_dir']) / exp_name
    save_results(eval_metrics, exp_name, str(results_root))

    logger.info(
        f"Spent {(time.time() - st_time) / 60:.1f} min. "
        f"Evaluation completed. Metrics: {eval_metrics}"
    )


if __name__ == "__main__":
    main()
