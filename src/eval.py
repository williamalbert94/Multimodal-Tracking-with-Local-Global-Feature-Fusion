"""
Standalone Evaluation Script
=============================

Loads trained models and runs the validation stage from trainer_simple.py.
Uses the SAME visualization and metrics as training validation.

Usage:
    python src/eval.py --config configs/eval_config.yaml
"""

import os
import sys
import torch
import yaml
import logging
import argparse
from pathlib import Path

# Simple EasyDict replacement (no external dependency needed)
class EasyDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from model.model import rastreador
from models.reid_module import ReIDTrackingModule
from dataloader.datagen import TrackingDataVOD
from utils.trainer_simple import run_epoch_simple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized point clouds and Object_3D instances.
    """
    transposed = list(zip(*batch))
    collated = []

    for i, samples in enumerate(transposed):
        # Elements that can vary in size - keep as list
        if i in [0, 1, 2, 3, 4, 8, 9, 10]:
            collated.append(samples)
        # Fixed-size elements
        elif i == 5:  # curr_idx
            collated.append(torch.tensor(samples))
        elif i == 6:  # clip names
            collated.append(samples)
        elif i == 7:  # ego_motion
            collated.append(torch.stack([torch.from_numpy(s) if not isinstance(s, torch.Tensor) else s for s in samples]))
        elif i == 11:  # new_seq
            collated.append(samples)
        else:
            collated.append(samples)

    return tuple(collated)


def load_models(args, device='cuda'):
    """
    Load segmentation backbone and Re-ID module from checkpoints.

    Args:
        args: Config with checkpoint paths
        device: Device to load on

    Returns:
        net: Main model (for segmentation)
        reid_module: Re-ID tracking module
    """
    logger.info('=' * 80)
    logger.info('Loading Models')
    logger.info('=' * 80)

    # ===== LOAD SEGMENTATION MODEL =====
    seg_checkpoint_path = os.path.join(
        args.segmentation_checkpoint_dir,
        'models/best_miou_model.pth'
    )
    logger.info(f'Loading segmentation model from: {seg_checkpoint_path}')

    # Set required attributes for model initialization
    args.extractor = 'LocalGlobalFusionSimple'

    net = rastreador(args).to(device)

    seg_checkpoint = torch.load(seg_checkpoint_path, map_location=device)

    # Try different checkpoint formats
    if 'model_state_dict' in seg_checkpoint:
        net.load_state_dict(seg_checkpoint['model_state_dict'], strict=False)
    elif 'model_state' in seg_checkpoint:
        net.load_state_dict(seg_checkpoint['model_state'], strict=False)
    else:
        net.load_state_dict(seg_checkpoint, strict=False)

    net.eval()
    logger.info('Segmentation model loaded')

    # ===== LOAD RE-ID MODEL =====
    reid_checkpoint_path = os.path.join(
        args.reid_checkpoint_dir,
        'models/best_samota_model.pth'
    )
    logger.info(f'Loading Re-ID model from: {reid_checkpoint_path}')

    reid_module = ReIDTrackingModule(args).to(device)

    reid_checkpoint = torch.load(reid_checkpoint_path, map_location=device)

    # Try different checkpoint formats
    if 'reid_module_state_dict' in reid_checkpoint:
        reid_module.load_state_dict(reid_checkpoint['reid_module_state_dict'], strict=False)
    elif 'model_state_dict' in reid_checkpoint:
        reid_module.load_state_dict(reid_checkpoint['model_state_dict'], strict=False)
    elif 'model_state' in reid_checkpoint:
        reid_module.load_state_dict(reid_checkpoint['model_state'], strict=False)
    else:
        logger.warning('Trying direct load')
        reid_module.load_state_dict(reid_checkpoint, strict=False)

    reid_module.eval()
    logger.info('Re-ID model loaded')

    return net, reid_module


def create_dataloader(args):
    """Create validation dataloader."""
    logger.info('Creating validation dataset...')

    # Set eval mode for dataset
    args.eval = True

    val_dataset = TrackingDataVOD(
        args=args,
        data_dir='val'
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=custom_collate_fn
    )

    logger.info(f'Validation set: {len(val_dataset)} samples')
    return val_loader


def run_evaluation(args):
    """
    Run evaluation using the validation stage from trainer_simple.py.

    This uses the SAME code as training validation, ensuring identical:
      - Metrics computation
      - Visualization generation
      - MOT evaluation
    """
    device = torch.device(f'cuda:{args.cuda_device}' if not args.no_cuda else 'cpu')

    # Load models
    net, reid_module = load_models(args, device)

    # Create dataloader
    val_loader = create_dataloader(args)

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)

    logger.info('=' * 80)
    logger.info('Running Evaluation (Validation Stage)')
    logger.info('=' * 80)
    logger.info(f'Output directory: {output_dir}')
    logger.info(f'Visualizations: {vis_dir}')
    logger.info(f'Save visualizations: {args.save_visualizations}')

    # Run validation epoch using trainer_simple's function
    # This is the SAME function used during training validation
    epoch_losses, epoch_metrics = run_epoch_simple(
        args=args,
        net=net,
        train_loader=val_loader,
        logger=logger,
        optimizer=None,  # No optimizer in eval mode
        mode='val',  # Validation mode
        ep_num=0,
        pretrain=False,
        save_visualizations=args.save_visualizations,
        vis_dir=str(vis_dir),
        reid_module=reid_module,
        train_mode='reid_only'  # Re-ID evaluation mode
    )

    # Print final metrics
    logger.info('=' * 80)
    logger.info('Evaluation Results')
    logger.info('=' * 80)

    if 'MOTA' in epoch_metrics:
        logger.info(f'MOTA:    {epoch_metrics["MOTA"]:.4f}')
    if 'IDF1' in epoch_metrics:
        logger.info(f'IDF1:    {epoch_metrics["IDF1"]:.4f}')
    if 'sAMOTA' in epoch_metrics:
        logger.info(f'sAMOTA:  {epoch_metrics["sAMOTA"]:.4f}')
    if 'box_precision' in epoch_metrics:
        logger.info(f'Box Precision: {epoch_metrics["box_precision"]:.4f}')
    if 'box_recall' in epoch_metrics:
        logger.info(f'Box Recall:    {epoch_metrics["box_recall"]:.4f}')

    # Save metrics to file
    import json
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        # Convert numpy values to float for JSON serialization
        metrics_json = {k: float(v) if hasattr(v, 'item') else v
                       for k, v in epoch_metrics.items()}
        json.dump(metrics_json, f, indent=2)

    logger.info(f'Metrics saved to {metrics_file}')
    logger.info(f'Visualizations saved to {vis_dir}')

    logger.info('=' * 80)
    logger.info('Evaluation Complete!')
    logger.info('=' * 80)

    return epoch_metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Standalone Re-ID Evaluation')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to evaluation config (YAML)')
    args_cli = parser.parse_args()

    # Load config
    with open(args_cli.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    args = EasyDict(config_dict)

    # Set defaults
    args.setdefault('cuda_device', '0')
    args.setdefault('no_cuda', False)
    args.setdefault('num_workers', 4)
    args.setdefault('num_points', 512)
    args.setdefault('batch_size', 1)
    args.setdefault('output_dir', 'eval_results')
    args.setdefault('save_visualizations', True)

    # Run evaluation
    metrics = run_evaluation(args)


if __name__ == '__main__':
    main()
