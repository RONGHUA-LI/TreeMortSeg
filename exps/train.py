import os
import time
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
from torchvision.ops import sigmoid_focal_loss
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets, mask=None):
        probs = torch.sigmoid(logits)  # shape (B,1,H,W)
        if mask is not None:
            probs = probs * mask

        p_flat = probs.view(-1)
        t_flat = targets.float().view(-1)

        inter = (p_flat * t_flat).sum()
        union = p_flat.sum() + t_flat.sum()
        dice = (inter + self.eps) / (union + self.eps)
        return 1 - dice


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction  # 'none' is recommended for external masking

    def forward(self, logits, targets):
        return sigmoid_focal_loss(
            inputs=logits,
            targets=targets.float(),
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction
        )


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, lambda_dice=1.0):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.lambda_dice = lambda_dice  # 控制 Dice Loss 权重

    def forward(self, logits, targets):
        loss_focal = self.focal(logits, targets)
        loss_dice = self.dice(logits, targets)
        return loss_focal + self.lambda_dice * loss_dice


class EdgeBoundaryLoss(nn.Module):
    def __init__(self, lambda_boundary=1.0, max_dist=10.0):
        super().__init__()
        self.dice = DiceLoss()
        self.lambda_boundary = lambda_boundary
        self.max_dist = max_dist

    def forward(self, logits, gts, dist_maps, mask=None):
        loss_dice = self.dice(logits, gts, mask)

        probs = torch.sigmoid(logits)

        # The distance beyond max_dist will not increase the penalty, and scaled to 0-1.
        dist_maps = torch.clamp(dist_maps, 0, self.max_dist) / self.max_dist

        if mask is not None:
            probs = probs * mask
            dist_maps = dist_maps * mask
            loss_boundary = (probs * dist_maps).sum() / (mask.sum() + 1e-6)
        else:
            loss_boundary = (probs * dist_maps).mean()

        return loss_dice + self.lambda_boundary * loss_boundary


def compute_iou_stats(logits, targets, mask=None, threshold=0.5):
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


def train_model(model, train_loader, val_loader, cfg, exp_name):
    """
    Train the model with validation and early stopping.

    Args:
      model: nn.Module
      train_loader: DataLoader for training
      val_loader: DataLoader for validation
      cfg: full config dict
      exp_name: experiment identifier

    Returns:
      dict containing best monitored metric
    """
    logger = logging.getLogger(__name__)
    # Device
    gpu_id = cfg['experiment'].get('gpu_id', 0)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")

    # Unpack training cfg
    tr_cfg = cfg['training']
    criterion_type = tr_cfg['criterion'].get('type', 'BCEWithLogitsLoss')
    if criterion_type == 'BCEWithLogitsLoss':
        if tr_cfg['criterion'].get('w_pos', 0) > 0:
            pos_weight = torch.tensor(tr_cfg['criterion']['w_pos'], device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        else:
            criterion = nn.BCEWithLogitsLoss(reduction='none')
    elif criterion_type == 'FocalLoss':
        alpha = tr_cfg['criterion'].get('alpha', 0.25)
        gamma = tr_cfg['criterion'].get('gamma', 2.0)
        criterion = FocalLoss(alpha=alpha, gamma=gamma)
    else:
        raise ValueError(f"Unsupported loss type: {criterion_type}")
    dice_loss = DiceLoss()
    w_dice = tr_cfg.get('w_dice', 0.5)

    w_mask = cfg['training'].get('w_mask', 1.0)
    w_edge = cfg['training'].get('w_edge', 1.0)
    w_dist = cfg['training'].get('w_dist', 5.0)
    w_boundary = cfg['training'].get('w_boundary', 0.2)

    edge_criterion = EdgeBoundaryLoss(lambda_boundary=w_boundary)
    dist_criterion = nn.L1Loss(reduction='none')

    scaler = GradScaler()

    optimizer = getattr(optim, tr_cfg['optimizer']['type'])(
        model.parameters(),
        lr=float(tr_cfg['learning_rate']),
        weight_decay=tr_cfg['optimizer'].get('weight_decay', 0)
    )
    scheduler = None
    if 'scheduler' in tr_cfg:
        sc = tr_cfg['scheduler']
        sched_type = sc['type']
        SchedulerClass = getattr(optim.lr_scheduler, sched_type)

        if sched_type == 'ExponentialLR':
            scheduler = SchedulerClass(
                optimizer,
                gamma=sc.get('gamma', 0.95)
            )
        elif sched_type == 'StepLR':
            scheduler = SchedulerClass(
                optimizer,
                step_size=sc.get('step_size', 10),
                gamma=sc.get('gamma', 0.1)
            )
        elif sched_type == 'CosineAnnealingWarmRestarts':
            scheduler = SchedulerClass(
                optimizer,
                T_0=sc.get('T_0', 10),
                T_mult=sc.get('T_mult', 1),
                eta_min=float(sc.get('eta_min', 0))
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {sched_type}")

    # Early stopping config
    es_cfg = tr_cfg['early_stopping']
    es_enabled = es_cfg.get('enabled', False)
    monitor = es_cfg.get('monitor', 'val_loss')
    mode = es_cfg.get('mode', 'min')
    patience = es_cfg.get('patience', 10)
    if mode == 'min':
        best_val = float('inf')
        improve = lambda cur, best: cur < best
    else:

        best_val = -float('inf')
        improve = lambda cur, best: cur > best
    epochs_no_improve = 0

    # Checkpoint directory
    ckpt_root = Path(cfg['output']['checkpoint_dir'])
    ckpt_dir = ckpt_root / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    results_root = Path(cfg['output']['results_dir'])
    results_dir = results_root / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = tr_cfg['epochs']
    log_interval = cfg['logging'].get('log_interval', 100)

    # ===== Resume =====
    train_losses, val_losses = [], []
    start_epoch = 1

    resume_path = tr_cfg.get('resume', {}).get('checkpoint', None)
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if scheduler and ckpt['scheduler']:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val = ckpt['best_val']
        epochs_no_improve = ckpt['epochs_no_improve']
        train_losses = ckpt['train_losses']
        val_losses = ckpt['val_losses']
        logger.info(
            f"Resume from {resume_path}, start_epoch={start_epoch}, "
            f"history={len(train_losses)}"
        )

    # Training loop
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()

        # Training
        model.train()
        train_loss_total = 0.0
        train_loss_mask = 0.0
        train_loss_edge = 0.0
        train_loss_dist = 0.0


        step_times = []
        for step, batch in enumerate(train_loader, 1):
            step_start = time.time()
            naips = batch['naip'].to(device)
            labels = batch['label'].unsqueeze(1).float().to(device)  # Mask
            edges = batch['edge'].unsqueeze(1).to(device)  # Edge
            boundary_dists = batch['boundary_dist'].unsqueeze(1).to(device)  # EdgeDist
            dists = batch['dist'].unsqueeze(1).to(device)  # Dist

            no_data = batch['no_data_mask'].unsqueeze(1).to(device)
            valid_mask = (~no_data).float()

            optimizer.zero_grad()

            with autocast(device_type='cuda'):

                outputs = model(naips)
                if not isinstance(outputs, (tuple, list)):
                    outputs = (outputs,)
                out_masks = outputs[0]
                out_edges = outputs[1] if len(outputs) > 1 else None
                out_dists = outputs[2] if len(outputs) > 2 else None

                # 1. Mask Loss
                raw_loss = criterion(out_masks, labels)  # (B,1,H,W)
                bce_loss = (raw_loss * valid_mask).sum() / valid_mask.sum()
                if w_dice > 0:
                    dloss = dice_loss(out_masks, labels, valid_mask)
                else:
                    dloss = torch.tensor(0.0, device=device)

                loss_mask = bce_loss + w_dice * dloss

                # 2. Edge Loss
                if out_edges is not None:
                    loss_edge = edge_criterion(out_edges, edges, boundary_dists, valid_mask)
                else:
                    loss_edge = torch.tensor(0.0, device=device)

                # 3. Distance Loss
                if out_dists is not None:
                    pos_mask = (dists > 0).float()
                    loss_dist = (dist_criterion(torch.sigmoid(out_dists), dists) * pos_mask).sum() / (pos_mask.sum() + 1e-8)

                else:
                    loss_dist = torch.tensor(0.0, device=device)

                total_loss = loss_mask * w_mask
                if out_edges is not None:
                    total_loss += loss_edge * w_edge
                if out_dists is not None:
                    total_loss += loss_dist * w_dist

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_total += total_loss.item()
            train_loss_mask += loss_mask.item() * w_mask
            train_loss_edge += loss_edge.item() * w_edge
            train_loss_dist += loss_dist.item() * w_dist
            step_times.append(time.time() - step_start)

            if step % log_interval == 0:
                avg_total = train_loss_total / step
                avg_mask = train_loss_mask / step
                avg_edge = train_loss_edge / step
                avg_dist = train_loss_dist / step

                log_parts = [
                    f"[Epoch {epoch}/{num_epochs}]",
                    f"Step {step}/{len(train_loader)}",
                    f"Time Spent: {sum(step_times) / 60:.1f}m",
                    f"Train Total Loss: {avg_total:.4f}",
                ]

                if avg_mask != 0:
                    log_parts.append(f"Mask Loss: {avg_mask:.4f}")
                if avg_edge != 0:
                    log_parts.append(f"Edge Loss: {avg_edge:.4f}")
                if avg_dist != 0:
                    log_parts.append(f"Dist Loss: {avg_dist:.4f}")

                log_str = " - ".join(log_parts)
                logger.info(log_str)

                step_times = []

        avg_train = train_loss_total / len(train_loader)
        train_losses.append(avg_train)

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_loss_mask = 0.0
        val_loss_edge = 0.0
        val_loss_dist = 0.0
        total_inter = 0.0
        total_union = 0.0

        val_time = time.time()
        with torch.inference_mode(), autocast(device_type='cuda'):
            for batch in val_loader:
                naips = batch['naip'].to(device)
                labels = batch['label'].unsqueeze(1).float().to(device)  # Mask
                edges = batch['edge'].unsqueeze(1).to(device)  # Edge
                boundary_dists = batch['boundary_dist'].unsqueeze(1).to(device)  # EdgeDist
                dists = batch['dist'].unsqueeze(1).to(device)  # Dist

                no_data = batch['no_data_mask'].unsqueeze(1).to(device)
                valid_mask = (~no_data).float()

                outputs = model(naips)
                if not isinstance(outputs, (tuple, list)):
                    outputs = (outputs,)
                out_masks = outputs[0]
                out_edges = outputs[1] if len(outputs) > 1 else None
                out_dists = outputs[2] if len(outputs) > 2 else None

                # 1. Mask Loss
                raw_loss = criterion(out_masks, labels)
                bce_loss = (raw_loss * valid_mask).sum() / valid_mask.sum()
                if w_dice > 0:
                    dloss = dice_loss(out_masks, labels, valid_mask)
                else:
                    dloss = torch.tensor(0.0, device=device)
                loss_mask = bce_loss + w_dice * dloss

                val_loss_mask += loss_mask.item() * w_mask

                # 2. Edge Loss
                if out_edges is not None:
                    loss_edge = edge_criterion(out_edges, edges, boundary_dists, valid_mask)
                    val_loss_edge += loss_edge.item() * w_edge

                # 3. Distance Loss
                if out_dists is not None:
                    pos_mask = (dists > 0).float()
                    loss_dist = (dist_criterion(torch.sigmoid(out_dists), dists) * pos_mask).sum() / (pos_mask.sum() + 1e-8)
                    val_loss_dist += loss_dist.item() * w_dist

                total_loss = loss_mask * w_mask
                if out_edges is not None:
                    total_loss += loss_edge * w_edge
                if out_dists is not None:
                    total_loss += loss_dist * w_dist


                val_loss_total += total_loss.item()
                inter, union = compute_iou_stats(out_masks, labels, valid_mask)
                total_inter += inter.item()
                total_union += union.item()

        avg_val = val_loss_total / len(val_loader)
        avg_mask = val_loss_mask / len(val_loader)
        avg_edge = val_loss_edge / len(val_loader)
        avg_dist = val_loss_dist / len(val_loader)
        avg_iou = total_inter / (total_union + 1e-6)

        val_losses.append(avg_val)
        val_time = time.time() - val_time

        log_parts = [
            f"[Epoch {epoch}/{num_epochs}]",
            f"Time Spent: {val_time / 60:.1f}m",
            f"Val Total Loss: {avg_val:.4f}",
        ]

        if avg_mask != 0:
            log_parts.append(f"Mask Loss: {avg_mask:.4f}")
        if avg_edge != 0:
            log_parts.append(f"Edge Loss: {avg_edge:.4f}")
        if avg_dist != 0:
            log_parts.append(f"Dist Loss: {avg_dist:.4f}")

        log_parts.append(f"Val IoU: {avg_iou:.4f}")

        log_str = " - ".join(log_parts)
        logger.info(log_str)


        current = avg_val if monitor == 'val_loss' else None
        if current is not None and improve(current, best_val):
            best_val = avg_val
            epochs_no_improve = 0
            best_path = ckpt_dir / f"{exp_name}_best_loss.pth"
            torch.save(model.state_dict(), best_path)
            logger.info(f"--- New Best Val Loss Model Saved! (Val Loss: {best_val:.4f}) ---")
        else:
            epochs_no_improve += 1
            if es_enabled:
                logger.info(f"No improvement in {monitor} for {epochs_no_improve}/{patience} epochs.")
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break

        # Scheduler step
        if scheduler:
            scheduler.step()

        # Disable epoch checkpoint for saving memory
        # torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch}.pth")
        # torch.save(
        #     {
        #         "epoch": epoch,
        #         "model": model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "scheduler": scheduler.state_dict() if scheduler else None,
        #         "epochs_no_improve": epochs_no_improve,
        #         "train_losses": train_losses,
        #         "val_losses": val_losses,
        #     },
        #     ckpt_dir / f"{exp_name}_last.pth"
        # )

        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch} done in {epoch_time / 60:.1f}m - Train: {avg_train:.4f}, Val: {avg_val:.4f}")

    # plot loss curve
    try:
        epochs = list(range(1, epoch + 1))
        plt.figure()
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        loss_curve_path = results_dir / f"{exp_name}_loss_curve.png"
        plt.savefig(loss_curve_path)
        plt.close()
        logger.info(f"Loss curve saved to {loss_curve_path}")
    except Exception as e:
        logger.warning(f"Failed to plot loss curve: {e}")

    return {f"best_{monitor}": best_val}

