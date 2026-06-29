import logging
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

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
from utils.metrics import ConfusionMatrixTracker

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def evaluate_model(model, test_loader, cfg, exp_name):
    """Load the best checkpoint when available and evaluate on the test loader."""
    logger = logging.getLogger(__name__)

    gpu_id = cfg["experiment"].get("gpu_id", 0)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    ckpt_root = Path(cfg["output"]["checkpoint_dir"])
    if not ckpt_root.is_absolute():
        ckpt_root = PROJECT_ROOT / ckpt_root
    ckpt_path = ckpt_root / exp_name / f"{exp_name}_best.pth"

    if ckpt_path.exists():
        logger.info(f"Loading model weights from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        logger.warning(
            f"Checkpoint not found at {ckpt_path}. Using current model weights."
        )

    model.to(device)
    model.eval()

    metric_names = cfg["evaluation"].get(
        "metrics",
        ["precision", "recall", "f1", "iou", "accuracy"],
    )
    tracker = ConfusionMatrixTracker(num_classes=cfg["model"]["num_classes"])

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Evaluating"):
            imgs = batch["naip"].to(device)
            gt = batch["label"].unsqueeze(1).numpy().astype(np.uint8)
            no_data = batch["no_data_mask"].unsqueeze(1).numpy()

            outputs = model(imgs)

            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(np.uint8)

            tracker.update(preds=preds, gts=gt, no_data_mask=no_data)

    results = tracker.compute_metrics(metric_names)
    logger.info(f"Evaluation results:\n{results}")
    return results


def main():
    """Run standalone evaluation from the configured checkpoint."""
    start_time = time.time()

    args = parse_args()
    cfg = load_config(args.config)
    if args.overwrite_cfg:
        cfg = overwrite_config(args, cfg)

    exp_id = cfg["experiment"]["id"].zfill(3)
    exp_name = f"exp{exp_id}_{cfg['model']['name']}"

    log_dir = Path(cfg["logging"]["log_dir"])
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir
    logger = setup_logging(log_dir, f"eval_{exp_name}")
    logger.info(f"Resume evaluation for experiment: {exp_name}")

    set_seed(cfg["experiment"]["seed"])

    _, _, test_loader = get_dataloader(cfg)

    model = build_treemortseg({"in_channels": cfg["model"]["in_channels"]})

    eval_metrics = evaluate_model(model, test_loader, cfg, exp_name)

    results_dir = Path(cfg["output"]["results_dir"])
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir
    results_root = results_dir / exp_name

    save_results(eval_metrics, exp_name, str(results_root))

    elapsed_minutes = (time.time() - start_time) / 60
    logger.info(f"Spent {elapsed_minutes:.1f} min. Evaluation completed.")


if __name__ == "__main__":
    main()
