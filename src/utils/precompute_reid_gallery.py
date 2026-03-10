"""
Pre-compute Re-ID Gallery for all sequences.

This script processes each sequence in the dataset and extracts:
  - Re-ID embeddings
  - Bounding boxes
  - Track IDs
  - Class labels
  - Segmented points (optional)

Saves to disk in efficient format for training/evaluation.

Usage:
    python src/utils/precompute_reid_gallery.py --config configs/reid_only.yaml
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.reid_module import ReIDTrackingModule, convert_o3d_boxes_to_tensor
from models.gallery_manager import (
    GalleryManager,
    compute_motion_representation,
    compute_velocities_from_boxes,
    extract_ego_motion_from_batch
)
from utils.parser_util import parser
from dataloader.datagen import get_train_dataset, get_val_dataset


def extract_sequence_embeddings(model, dataloader, sequence_name, args, device='cuda'):
    """
    Extract embeddings for a single sequence.

    Args:
        model: ReIDTrackingModule
        dataloader: DataLoader for the sequence
        sequence_name: str, sequence identifier
        args: config args
        device: device for computation

    Returns:
        sequence_data: dict with structure for GalleryManager
    """
    print(f"\n[Pre-compute] Processing sequence: {sequence_name}")

    model.eval()

    # Collect embeddings per track
    tracks_collection = defaultdict(lambda: {
        'embeddings': [],
        'boxes': [],
        'timestamps': [],
        'class': None,
        'points': [],
        'ego_motion': [],      # NEW: Ego motion per frame
    })

    frame_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Extracting {sequence_name}")):
            # Move batch to device
            pc1 = batch['pc1'].to(device)  # [B, N, 3] or [B, 3, N]
            seg_gt = batch['seg_gt'].to(device)  # [B, N]
            boxes_gt = batch['boxes_gt']  # List[dict] of Open3D boxes
            track_ids_gt = batch['track_ids_gt']  # List of track IDs per sample
            class_labels_gt = batch.get('class_labels_gt', [None] * len(boxes_gt))

            # Extract ego motion (if available)
            ego_motion_batch = extract_ego_motion_from_batch(batch)  # [B, 3] or None

            # Ensure correct shape [B, N, 3]
            if pc1.shape[1] == 3:  # [B, 3, N]
                pc1 = pc1.permute(0, 2, 1)

            batch_size = pc1.shape[0]

            # Forward pass to get embeddings
            reid_batch = {
                'pc1': pc1,
                'seg_gt': seg_gt,
                'boxes_gt': boxes_gt,
                'track_ids_gt': track_ids_gt,
                'class_labels_gt': class_labels_gt,
            }

            # Extract features
            outputs = model(reid_batch, epoch=0)

            if 'embeddings' not in outputs:
                continue

            embeddings = outputs['embeddings']  # List[Tensor] per batch
            pred_boxes = outputs['boxes']       # List[Tensor] per batch

            # Process each sample in batch
            for b in range(batch_size):
                # Convert GT boxes to tensor format
                boxes_gt_tensor, gt_track_ids = convert_o3d_boxes_to_tensor(boxes_gt[b])

                if len(gt_track_ids) == 0:
                    continue

                # Get embeddings for this sample
                emb_b = embeddings[b].cpu().numpy()  # [M, D]
                boxes_b = pred_boxes[b].cpu().numpy()  # [M, 7]

                # Match predicted boxes/embeddings to GT track IDs
                # (using IoU matching)
                matched_ids = model._assign_gt_ids_to_pred_boxes(
                    [pred_boxes[b]],
                    [boxes_gt_tensor.to(device)],
                    [gt_track_ids.to(device)]
                )[0].cpu().numpy()  # [M]

                # Get class labels
                class_labels_dict = class_labels_gt[b] if class_labels_gt[b] is not None else {}

                # Store embeddings per track
                for i, track_id in enumerate(matched_ids):
                    if track_id == -1:
                        # Unmatched prediction - skip
                        continue

                    track_id = int(track_id)

                    # Get class label
                    class_label = class_labels_dict.get(track_id, 'unknown')

                    # Store data
                    tracks_collection[track_id]['embeddings'].append(emb_b[i])
                    tracks_collection[track_id]['boxes'].append(boxes_b[i])
                    tracks_collection[track_id]['timestamps'].append(frame_count + b)

                    # Store ego motion
                    if ego_motion_batch is not None:
                        tracks_collection[track_id]['ego_motion'].append(ego_motion_batch[b])
                    else:
                        tracks_collection[track_id]['ego_motion'].append(np.zeros(3))

                    # Set class (first occurrence)
                    if tracks_collection[track_id]['class'] is None:
                        tracks_collection[track_id]['class'] = class_label

                    # Optionally store segmented points
                    if args.save_points:
                        # Extract points for this box
                        seg_mask_b = seg_gt[b].cpu().numpy()  # [N]
                        pc_b = pc1[b].cpu().numpy()  # [N, 3]

                        # Get points belonging to this object
                        obj_mask = seg_mask_b > 0  # Simple: all foreground
                        obj_points = pc_b[obj_mask]

                        tracks_collection[track_id]['points'].append(obj_points)

            frame_count += batch_size

    # Convert lists to numpy arrays
    tracks_dict = {}
    for track_id, track_data in tracks_collection.items():
        if len(track_data['embeddings']) == 0:
            continue

        # Convert to numpy arrays
        embeddings_arr = np.stack(track_data['embeddings'], axis=0)  # [T, D]
        boxes_arr = np.stack(track_data['boxes'], axis=0)            # [T, 7]
        timestamps_arr = np.array(track_data['timestamps'])          # [T]
        ego_motion_arr = np.stack(track_data['ego_motion'], axis=0) if len(track_data['ego_motion']) > 0 else None  # [T, 3]

        # ===== COMPUTE MOTION FEATURES =====
        # Compute velocities from box positions
        velocities_arr = compute_velocities_from_boxes(boxes_arr, timestamps_arr)  # [T, 3]

        # Compute motion representation (velocity + acceleration + ego motion + angular velocity)
        motion_map_arr = compute_motion_representation(
            boxes_arr,
            ego_motion=ego_motion_arr,
            timestamps=timestamps_arr
        )  # [T, 10]

        tracks_dict[track_id] = {
            'embeddings': embeddings_arr,
            'boxes': boxes_arr,
            'timestamps': timestamps_arr,
            'class': track_data['class'],
            'points': track_data['points'] if args.save_points else None,
            'ego_motion': ego_motion_arr,          # [T, 3]
            'velocities': velocities_arr,          # [T, 3]
            'motion_map': motion_map_arr,          # [T, 10]
        }

    # Create sequence data structure
    sequence_data = {
        'sequence_id': sequence_name,
        'tracks': tracks_dict,
        'metadata': {
            'num_frames': frame_count,
            'num_tracks': len(tracks_dict),
            'embedding_dim': list(tracks_dict.values())[0]['embeddings'].shape[1] if tracks_dict else 0,
            'motion_enabled': True,  # Motion features included
        }
    }

    print(f"[Pre-compute] Extracted {len(tracks_dict)} tracks, {frame_count} frames")

    return sequence_data


def precompute_all_sequences(args):
    """
    Pre-compute embeddings for all sequences in train/val splits.

    Args:
        args: Config arguments
    """
    device = torch.device(f'cuda:{args.cuda_device}' if not args.no_cuda else 'cpu')

    # Initialize Re-ID model
    print("\n" + "="*80)
    print("🔧 Initializing Re-ID Model for Pre-computation")
    print("="*80)

    reid_module = ReIDTrackingModule(args).to(device)

    # Load checkpoint if specified
    if args.load_checkpoint and args.model_path:
        print(f"\nLoading checkpoint: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        reid_module.load_state_dict(checkpoint['reid_module'], strict=False)
        print("Checkpoint loaded")
    else:
        print("\nNo checkpoint specified - using randomly initialized model")
        print("   For better embeddings, train the model first or specify --model_path")

    reid_module.eval()

    # Initialize Gallery Manager
    gallery_dir = Path(args.checkpoint_dir) / 'reid_gallery'
    gallery_manager = GalleryManager(gallery_dir=gallery_dir, device=device)

    print(f"\n📁 Gallery directory: {gallery_dir}")

    # Process training sequences
    print("\n" + "="*80)
    print("Processing TRAINING sequences")
    print("="*80)

    # Get training dataset
    # Note: We need to iterate sequence by sequence
    # The dataloader returns batches, so we need to track sequences manually

    # For VOD dataset, sequences are organized by clip
    # We'll process each clip separately

    if args.dataset == 'vod':
        from dataloader.track_vod_3d import VODTrackingDataset

        # Get training clips
        train_dataset = VODTrackingDataset(
            root_dir=args.dataset_path,
            split='training',
            num_points=args.num_points,
            adjacent_frames=True,
            return_metadata=True,  # Get sequence info
        )

        # Group by sequence
        sequences_train = {}
        for idx in range(len(train_dataset)):
            sample = train_dataset[idx]
            seq_name = sample.get('sequence_name', f'train_seq_{idx}')

            if seq_name not in sequences_train:
                sequences_train[seq_name] = []

            sequences_train[seq_name].append(idx)

        # Process each sequence
        for seq_name, indices in sequences_train.items():
            # Create subset dataloader for this sequence
            subset = torch.utils.data.Subset(train_dataset, indices)
            seq_loader = torch.utils.data.DataLoader(
                subset,
                batch_size=args.batch_size,
                shuffle=False,  # Keep temporal order
                num_workers=args.num_workers,
            )

            # Extract embeddings
            sequence_data = extract_sequence_embeddings(
                reid_module, seq_loader, seq_name, args, device
            )

            # Save to disk
            gallery_manager.save_sequence_gallery(seq_name, sequence_data)

    # Process validation sequences
    print("\n" + "="*80)
    print("Processing VALIDATION sequences")
    print("="*80)

    if args.dataset == 'vod':
        # Get validation clips
        val_dataset = VODTrackingDataset(
            root_dir=args.dataset_path,
            split='validation',
            num_points=args.num_points,
            adjacent_frames=True,
            return_metadata=True,
        )

        # Group by sequence
        sequences_val = {}
        for idx in range(len(val_dataset)):
            sample = val_dataset[idx]
            seq_name = sample.get('sequence_name', f'val_seq_{idx}')

            if seq_name not in sequences_val:
                sequences_val[seq_name] = []

            sequences_val[seq_name].append(idx)

        # Process each sequence
        for seq_name, indices in sequences_val.items():
            # Create subset dataloader for this sequence
            subset = torch.utils.data.Subset(val_dataset, indices)
            seq_loader = torch.utils.data.DataLoader(
                subset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )

            # Extract embeddings
            sequence_data = extract_sequence_embeddings(
                reid_module, seq_loader, seq_name, args, device
            )

            # Save to disk
            gallery_manager.save_sequence_gallery(seq_name, sequence_data)

    # Print summary
    print("\n" + "="*80)
    print("Pre-computation Complete!")
    print("="*80)

    available_sequences = gallery_manager.list_available_sequences()
    print(f"\nTotal sequences processed: {len(available_sequences)}")

    # Show details for each sequence
    print("\nSequence Details:")
    for seq_id in available_sequences:
        info = gallery_manager.get_sequence_info(seq_id)
        print(f"  - {seq_id}: {info['num_tracks']} tracks, "
              f"{info['total_embeddings']} embeddings, "
              f"avg {info['avg_embeddings_per_track']:.1f} per track")

    print(f"\nGallery saved to: {gallery_dir}")
    print("\n🚀 Ready for Re-ID training with pre-computed galleries!")


if __name__ == '__main__':
    # Parse arguments
    args = parser()

    # Add precompute-specific args
    args.save_points = getattr(args, 'save_points', False)  # Save segmented points (uses more disk)

    # Run pre-computation
    precompute_all_sequences(args)
