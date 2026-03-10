"""
Simplified training script following baseline RaTrack approach.
Uses simple losses and metrics like the original working baseline.
"""

import argparse
from numba import NumbaWarning
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import os
import sys
import torch
from utils import parse_args_from_yaml, create_log_file, setup_logger
import logging
from torch.utils.data import DataLoader
from dataloader.datagen import TrackingDataVOD
from utils.trainer_simple import run_train_simple


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized point clouds.
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


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')


def main(config_path: str):
    args = parse_args_from_yaml(config_path)
    torch.cuda.empty_cache()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

    if args.mode == 'train':
        _init_(args)
        log_path = create_log_file(args.exp_name, args.mode)
        logger = setup_logger(log_path)

        # Log configuration
        logger.info("="*80)
        logger.info("SIMPLIFIED BASELINE TRAINING (Following RaTrack)")
        logger.info("="*80)
        logger.info("Changes from complex version:")
        logger.info("  - Simple BCE loss (no Focal Loss)")
        logger.info("  - Simple EPE loss (no MSE)")
        logger.info("  - Baseline weights: 1.0*seg + 0.5*flow")
        logger.info("  - Pretrain mode: first 5 epochs only segmentation")
        logger.info("  - Validation every 2 epochs with visualization")
        logger.info("  - Best model checkpointing by mIoU and F1")
        logger.info("="*80)
        for key, value in vars(args).items():
            logger.info(f"{key}: {value}")
        logger.info("="*80)

        # Create train dataset

        train_dataset = TrackingDataVOD(args, args.dataset_path)
        train_loader = DataLoader(
            train_dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=custom_collate_fn
        )

        # Create validation dataset
        logger.info("Creating validation dataset...")
        args.eval=True
        val_dataset = TrackingDataVOD(args, args.dataset_path)
        val_loader = DataLoader(
            val_dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=custom_collate_fn
        )
        logger.info(f"Validation set size: {len(val_dataset)} samples")
        args.eval=False
        run_train_simple(args, logger, train_loader, val_loader)

        return logger


if __name__ == '__main__':
    # Suppress warnings
    numba_logger = logging.getLogger("numba")
    numba_logger.setLevel(logging.CRITICAL)

    import warnings
    warnings.filterwarnings("ignore", category=NumbaWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Configure stdout for Docker
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description='Process config path.')
    parser.add_argument('--config', type=str, default='./configs/configs.yaml',
                        help='Path to the configuration file')
    args = parser.parse_args()

    main(args.config)
