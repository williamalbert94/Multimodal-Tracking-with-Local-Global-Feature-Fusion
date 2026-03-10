"""
Utilities for model checkpoint management.

This module provides functions to save, load, and manage model checkpoints
during training, including best model tracking and automatic cleanup.
"""

import os
import glob
import torch


def save_checkpoint(args, net, optimizer, epoch, metrics, is_best=False, logger=None):
    """
    Save model checkpoint.

    Args:
        args: Configuration arguments
        net: Neural network model
        optimizer: Optimizer
        epoch: Current epoch number
        metrics: Dictionary of metrics from this epoch (can contain 'train' and 'val' subdicts)
        is_best: Whether this is the best model so far based on validation metric
        logger: Logger for output (optional)

    Returns:
        checkpoint_path: Path to saved checkpoint

    Example:
        >>> save_checkpoint(args, net, optimizer, epoch=5,
        ...                 metrics={'train': {...}, 'val': {...}},
        ...                 is_best=True, logger=logger)
        'checkpoints/exp_name/models/checkpoint_epoch_5.pth'
    """
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.exp_name, 'models')
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state': net.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'metrics': metrics,
        'args': vars(args)  # Save config for reference
    }

    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)

    if logger:
        logger.info(f'  ✓ Saved checkpoint: {checkpoint_path}')

    # Save best checkpoint if this is the best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)

        # Extract validation mIoU for logging
        val_miou = metrics.get('val', {}).get('mIoU', 0.0)
        if logger:
            logger.info(f'  ✓ Saved BEST checkpoint: {best_path} (val mIoU: {val_miou:.2f}%)')

    return checkpoint_path


def load_checkpoint(checkpoint_path, net, optimizer=None, logger=None):
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file (.pth)
        net: Neural network model to load state into
        optimizer: Optimizer to load state into (optional)
        logger: Logger for output (optional)

    Returns:
        epoch: Epoch number from checkpoint
        metrics: Metrics dictionary from checkpoint

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist

    Example:
        >>> epoch, metrics = load_checkpoint('checkpoint_best.pth', net, optimizer, logger)
        >>> print(f"Resumed from epoch {epoch}")
        Resumed from epoch 10
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if logger:
        logger.info(f'Loading checkpoint from: {checkpoint_path}')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cuda')

    # Load model state
    net.load_state_dict(checkpoint['model_state'])

    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    # Extract epoch and metrics
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})

    if logger:
        logger.info(f'✓ Loaded checkpoint from epoch {epoch}')
        # Log validation metrics if available
        if 'val' in metrics:
            val_metrics = metrics['val']
            logger.info(f'  Checkpoint val metrics: EPE={val_metrics.get("EPE", 0):.4f}, '
                       f'mIoU={val_metrics.get("mIoU", 0):.2f}%')

    return epoch, metrics


def cleanup_old_checkpoints(checkpoint_dir, keep_last=3, logger=None):
    """
    Remove old checkpoints, keeping only the last K and the best.

    This function keeps disk usage under control by removing older checkpoints
    while preserving recent ones and the best model.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        keep_last: Number of recent checkpoints to keep (default: 3)
        logger: Logger for output (optional)

    Note:
        - The best checkpoint (checkpoint_best.pth) is never deleted
        - Only regular epoch checkpoints (checkpoint_epoch_*.pth) are cleaned up

    Example:
        >>> cleanup_old_checkpoints('checkpoints/exp/models', keep_last=3, logger=logger)
        # Keeps: checkpoint_epoch_8.pth, checkpoint_epoch_9.pth, checkpoint_epoch_10.pth
        # Keeps: checkpoint_best.pth (always preserved)
        # Removes: checkpoint_epoch_7.pth, checkpoint_epoch_6.pth, ...
    """
    if not os.path.exists(checkpoint_dir):
        return

    # Find all regular checkpoints (not best)
    pattern = os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth')
    checkpoints = glob.glob(pattern)

    if len(checkpoints) <= keep_last:
        return  # Nothing to clean up

    # Sort by epoch number (extract from filename)
    # Filename format: checkpoint_epoch_{epoch}.pth
    def extract_epoch(path):
        try:
            basename = os.path.basename(path)  # checkpoint_epoch_5.pth
            epoch_str = basename.split('_epoch_')[1].split('.pth')[0]  # '5'
            return int(epoch_str)
        except (IndexError, ValueError):
            return 0

    checkpoints.sort(key=extract_epoch)

    # Remove older checkpoints (keep only last K)
    to_remove = checkpoints[:-keep_last]

    for ckpt in to_remove:
        try:
            os.remove(ckpt)
            if logger:
                logger.info(f'  Removed old checkpoint: {os.path.basename(ckpt)}')
        except OSError as e:
            if logger:
                logger.warning(f'  Failed to remove {ckpt}: {e}')
