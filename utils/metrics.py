import numpy as np
import torch


def compute_iou_stats(logits, targets, mask=None, threshold=0.5):
    """Compute the intersection and union statistics for IoU."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).bool()
    targets = targets.bool()

    if mask is not None:
        mask = mask.bool()
        preds = preds & mask
        targets = targets & mask

    inter = (preds & targets).float().sum()
    union = (preds | targets).float().sum()

    return inter, union


class ConfusionMatrixTracker:
    def __init__(self, num_classes=1):
        self.num_classes = num_classes
        self.stats = {
            c: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for c in range(num_classes + 1)
        }
        self.stats["correct"] = 0
        self.stats["total"] = 0

    def update(
        self, preds: np.ndarray, gts: np.ndarray, no_data_mask: np.ndarray = None
    ):
        """Update confusion-matrix statistics from a prediction batch."""
        preds = preds.astype(np.uint8)
        gts = gts.astype(np.uint8)

        for i in range(preds.shape[0]):
            if no_data_mask is not None:
                valid = ~no_data_mask[i, 0]
                p_flat = preds[i, 0][valid]
                g_flat = gts[i, 0][valid]
            else:
                p_flat = preds[i, 0].ravel()
                g_flat = gts[i, 0].ravel()

            for c in range(self.num_classes + 1):
                self.stats[c]["tp"] += np.logical_and(
                    p_flat == c,
                    g_flat == c,
                ).sum()
                self.stats[c]["fp"] += np.logical_and(
                    p_flat == c,
                    g_flat != c,
                ).sum()
                self.stats[c]["fn"] += np.logical_and(
                    p_flat != c,
                    g_flat == c,
                ).sum()
                self.stats[c]["tn"] += np.logical_and(
                    p_flat != c,
                    g_flat != c,
                ).sum()

            self.stats["correct"] += (p_flat == g_flat).sum()
            self.stats["total"] += p_flat.size

    def compute_metrics(self, metric_names: list) -> dict:
        """Return requested metrics from accumulated confusion statistics."""
        results = {}

        for c in range(self.num_classes + 1):
            tp = self.stats[c]["tp"]
            fp = self.stats[c]["fp"]
            fn = self.stats[c]["fn"]

            p = self._safe_div(tp, tp + fp)
            r = self._safe_div(tp, tp + fn)

            if "precision" in metric_names:
                results[f"class{c}_precision"] = p
            if "recall" in metric_names:
                results[f"class{c}_recall"] = r
            if "f1" in metric_names:
                results[f"class{c}_f1"] = self._safe_div(2 * p * r, p + r)
            if "iou" in metric_names:
                results[f"class{c}_iou"] = self._safe_div(tp, tp + fp + fn)

        for m in ["precision", "recall", "f1", "iou"]:
            if m in metric_names:
                results[f"res_{m}_macro"] = 0.5 * (
                    results[f"class0_{m}"] + results[f"class1_{m}"]
                )

        if "accuracy" in metric_names:
            results["overall_accuracy"] = self._safe_div(
                self.stats["correct"],
                self.stats["total"],
            )

        return results

    @staticmethod
    def _safe_div(numerator, denominator):
        return float(numerator) / (denominator + 1e-6)
