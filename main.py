from pathlib import Path
import time

import yaml

from utils.tools import (
    load_config,
    overwrite_config,
    parse_args,
    save_results,
    set_seed,
    setup_logging,
)
from data_loader import get_dataloader
from model.treemortseg import build_treemortseg
from utils.train import train_model
from scripts.evaluate import evaluate_model

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    """Run training, evaluation, and result persistence for one experiment."""
    start_time = time.time()

    args = parse_args()
    cfg = load_config(args.config)
    if args.overwrite_cfg:
        cfg = overwrite_config(args, cfg)

    exp_name = cfg["experiment"]["id"].zfill(3)
    exp_name = f"exp{exp_name}"

    logger = setup_logging(PROJECT_ROOT / cfg["logging"]["log_dir"], exp_name)
    logger.info(f"Starting experiment: {exp_name}")
    logger.info("Configuration:\n" + yaml.dump(cfg, sort_keys=False))

    set_seed(cfg["experiment"]["seed"])

    train_loader, val_loader, test_loader = get_dataloader(cfg)
    model = build_treemortseg({"in_channels": cfg["model"]["in_channels"]})

    train_metrics = train_model(model, train_loader, val_loader, cfg, exp_name)

    eval_metrics = evaluate_model(model, test_loader, cfg, exp_name)

    results_root = PROJECT_ROOT / cfg["output"]["results_dir"] / exp_name
    for (t_key, t_val), (e_key, e_val) in zip(
        train_metrics.items(),
        eval_metrics.items(),
    ):
        eval_metrics[e_key] = {
            t_key: t_val,
            **e_val,
        }

    save_results(eval_metrics, exp_name, str(results_root))
    elapsed_hours = (time.time() - start_time) / 3600
    logger.info(
        f"Spent {elapsed_hours:.1f}h: Experiment {exp_name} completed. "
        f"Combined metrics: {eval_metrics}"
    )


if __name__ == "__main__":
    main()
