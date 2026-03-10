"""
eval_segmentation.py
====================
Evaluation script for segmentation model with advanced visualizations.

Usage:
    python src/eval_segmentation.py --config src/configs/segmentation_phase1.yaml \
        --model_path checkpoints/segmentation_phase1_testing/best_miou_model.pth \
        --output_dir eval_results

Generates:
    - Advanced 4-panel visualizations (scene flow + segmentation)
    - Evaluation metrics (mIoU, F1, RNE, EPE, SAS, etc.)
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import csv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataloader.datagen import TrackingDataVOD
from utils.parser_util import EasyDict
from model.model import Rastreador
from utils.visualization_eval import plot_advanced_eval_visualization


def custom_collate_fn(batch):
    """Custom collate function from trainer_simple.py"""
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


def compute_metrics(pred_seg, gt_seg, pred_flow, gt_flow, moving_mask):
    """
    Compute evaluation metrics.

    Args:
        pred_seg: [N] predicted segmentation (0/1)
        gt_seg: [N] GT segmentation
        pred_flow: [N, 3] predicted flow
        gt_flow: [N, 3] GT flow
        moving_mask: [N] GT moving mask

    Returns:
        dict of metrics
    """
    metrics = {}

    # Convert to numpy if needed
    if isinstance(pred_seg, torch.Tensor):
        pred_seg = pred_seg.cpu().numpy()
    if isinstance(gt_seg, torch.Tensor):
        gt_seg = gt_seg.cpu().numpy()
    if isinstance(moving_mask, torch.Tensor):
        moving_mask = moving_mask.cpu().numpy()

    # === Segmentation metrics ===
    # mIoU
    moving_union = ((pred_seg == 1) | (gt_seg == 1)).sum()
    static_union = ((pred_seg == 0) | (gt_seg == 0)).sum()

    if moving_union > 0:
        iou_moving = ((pred_seg == 1) & (gt_seg == 1)).sum() / moving_union
    else:
        iou_moving = 0.0

    if static_union > 0:
        iou_static = ((pred_seg == 0) & (gt_seg == 0)).sum() / static_union
    else:
        iou_static = 0.0

    metrics['mIoU'] = (iou_moving + iou_static) / 2
    metrics['IoU_moving'] = float(iou_moving)
    metrics['IoU_static'] = float(iou_static)

    # F1 score
    tp = ((pred_seg == 1) & (gt_seg == 1)).sum()
    fp = ((pred_seg == 1) & (gt_seg == 0)).sum()
    fn = ((pred_seg == 0) & (gt_seg == 1)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    metrics['F1'] = 2 * precision * recall / (precision + recall + 1e-8)

    # === Flow metrics (only on moving points) ===
    if moving_mask.sum() > 0:
        # EPE (End Point Error)
        flow_error = np.linalg.norm(pred_flow[moving_mask] - gt_flow[moving_mask], axis=1)
        metrics['EPE'] = float(np.mean(flow_error))

        # RME (Relative Motion Error)
        gt_flow_norm = np.linalg.norm(gt_flow[moving_mask], axis=1)
        metrics['RME'] = float(np.mean(flow_error / (gt_flow_norm + 1e-8)))

        # SAS (Speed Accuracy Score) - % of points with error < 0.3m
        metrics['SAS'] = float((flow_error < 0.3).mean() * 100)
    else:
        metrics['EPE'] = 0.0
        metrics['RME'] = 0.0
        metrics['SAS'] = 0.0

    return metrics


def evaluate(args, model, eval_loader, output_dir):
    """
    Run evaluation with visualizations.

    Args:
        args: Config arguments
        model: Trained model
        eval_loader: DataLoader for evaluation
        output_dir: Output directory for results
    """
    model.eval()

    # Create output directories
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # Accumulators for metrics
    all_metrics = []

    print("\n" + "="*80)
    print("RUNNING EVALUATION")
    print("="*80)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
            # Unpack batch (following trainer_simple.py format)
            if len(batch) == 17:
                pc1, pc2, ft1, ft2, pc1_compensated, index, seq, ego_motion, pc_last_lidar, pc0_lidar, pc1_lidar, is_new_seq, lbl1, lbl2, transforms1, transforms2, motion_features = batch
            else:
                pc1, pc2, ft1, ft2, pc1_compensated, index, seq, ego_motion, pc_last_lidar, pc0_lidar, pc1_lidar, is_new_seq, lbl1, lbl2, transforms1, transforms2 = batch

            # Sample points and move to GPU
            from utils.trainer_simple import sample_points
            num_points = args.num_points
            pc1 = sample_points(pc1, num_points).permute(0, 2, 1)[:, :3, :].cuda()  # [B, 3, N]
            ft1 = sample_points(ft1, num_points).permute(0, 2, 1)[:, :2, :].cuda()  # [B, 2, N]

            # Get GT labels
            from utils.models_utils import filter_object_points_batch
            gt_mov_pts1_batch, gt_cls1_batch, _, _, _, cls_obj_id1_batch, boxes1_batch, _, _, _, _ = filter_object_points_batch(
                args, lbl1, pc1, transforms1
            )

            # Forward pass
            output = model(pc1, ft1)

            # Extract predictions
            pred_seg_logits = output['seg']  # [B, 1, N]
            pred_seg = (torch.sigmoid(pred_seg_logits) > 0.5).squeeze(1).cpu().numpy()  # [B, N]

            # Extract flow (if available)
            pred_flow = output.get('flow', torch.zeros_like(pc1)).permute(0, 2, 1).cpu().numpy()  # [B, N, 3]

            # Process each sample in batch
            batch_size = pc1.shape[0]
            for b in range(batch_size):
                # Get GT segmentation
                if gt_mov_pts1_batch[b] is not None:
                    if isinstance(gt_mov_pts1_batch[b], torch.Tensor):
                        gt_seg = gt_mov_pts1_batch[b].cpu().numpy()
                    else:
                        gt_seg = gt_mov_pts1_batch[b]

                    # Flatten and resize to num_points
                    if gt_seg.ndim > 1:
                        gt_seg = gt_seg.flatten()
                    gt_seg = np.pad(gt_seg, (0, max(0, num_points - len(gt_seg))))[:num_points]
                else:
                    gt_seg = np.zeros(num_points)

                # Get point cloud
                pc_radar = pc1[b].permute(1, 0).cpu().numpy()  # [N, 3]

                # Get object IDs (for visualization) - FIXED
                if cls_obj_id1_batch[b] is not None:
                    if isinstance(cls_obj_id1_batch[b], torch.Tensor):
                        obj_ids = cls_obj_id1_batch[b].cpu().numpy()
                    else:
                        obj_ids = np.array(cls_obj_id1_batch[b])
                    # Ensure correct size
                    if len(obj_ids) < num_points:
                        obj_ids = np.pad(obj_ids, (0, num_points - len(obj_ids)), constant_values=-1)
                    elif len(obj_ids) > num_points:
                        obj_ids = obj_ids[:num_points]
                else:
                    obj_ids = None

                # Compute metrics
                gt_moving_mask = gt_seg > 0.5
                sample_metrics = compute_metrics(
                    pred_seg[b],
                    gt_seg,
                    pred_flow[b],
                    np.zeros_like(pred_flow[b]),  # TODO: Get GT flow
                    gt_moving_mask
                )
                all_metrics.append(sample_metrics)

                # === GENERATE VISUALIZATION (every N frames) ===
                if batch_idx % 10 == 0:  # Visualize every 10 batches
                    # Get frame index - FIXED
                    if isinstance(index, torch.Tensor):
                        frame_idx = index[b].item()
                    else:
                        frame_idx = index[b] if isinstance(index, (list, tuple)) else batch_idx

                    # Get lidar background (if available) - FIXED
                    pc_lidar = None
                    if pc1_lidar is not None:
                        try:
                            if isinstance(pc1_lidar, torch.Tensor):
                                pc_lidar = pc1_lidar[b].cpu().numpy()
                            elif isinstance(pc1_lidar, (list, tuple)) and len(pc1_lidar) > b:
                                if isinstance(pc1_lidar[b], torch.Tensor):
                                    pc_lidar = pc1_lidar[b].cpu().numpy()
                                else:
                                    pc_lidar = pc1_lidar[b]
                        except Exception as e:
                            print(f"Warning: Could not extract lidar for sample {b}: {e}")
                            pc_lidar = None

                    # Save path
                    save_path = os.path.join(vis_dir, f'frame_{frame_idx:06d}.png')

                    # Generate visualization
                    try:
                        plot_advanced_eval_visualization(
                            frame_idx=frame_idx,
                            pc_radar=pc_radar,
                            pc_radar_prev=None,  # TODO: Get previous frame
                            pc_lidar=pc_lidar,
                            gt_flow_vectors=None,
                            gt_seg_labels=gt_seg,
                            gt_obj_ids=obj_ids,
                            gt_moving_mask=gt_moving_mask,
                            pred_flow_vectors=pred_flow[b],
                            pred_seg_logits=None,
                            pred_seg_binary=pred_seg[b],
                            metrics=sample_metrics,
                            save_path=save_path
                        )
                    except Exception as e:
                        print(f"Warning: Visualization failed for frame {frame_idx}: {e}")
                        import traceback
                        traceback.print_exc()

    # === AGGREGATE METRICS ===
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    print("\nSegmentation Metrics:")
    print(f"  mIoU:          {avg_metrics['mIoU']:.4f}")
    print(f"  F1 Score:      {avg_metrics['F1']:.4f}")
    print(f"  IoU (moving):  {avg_metrics['IoU_moving']*100:.2f}%")
    print(f"  IoU (static):  {avg_metrics['IoU_static']*100:.2f}%")

    print("\nScene Flow Metrics:")
    print(f"  RME:           {avg_metrics['RME']:.4f}")
    print(f"  EPE:           {avg_metrics['EPE']:.2f} m")
    print(f"  SAS:           {avg_metrics['SAS']:.2f}%")

    print("\n" + "="*80)
    print(f"Visualizations saved to: {vis_dir}")
    print("="*80)

    # Save metrics to CSV
    csv_path = os.path.join(output_dir, 'eval_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=avg_metrics.keys())
        writer.writeheader()
        writer.writerow(avg_metrics)

    print(f"Metrics saved to: {csv_path}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Segmentation Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='eval_results', help='Output directory')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device')
    args_cmd = parser.parse_args()

    # Load config
    with open(args_cmd.config, 'r') as f:
        config = yaml.safe_load(f)

    args = EasyDict(config)
    args.cuda_device = args_cmd.gpu

    # Set device
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")

    # Load dataset (eval mode)
    args.eval = True
    eval_dataset = TrackingDataVOD(args, args.dataset_path)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        drop_last=False
    )

    print(f"\nEvaluation samples: {len(eval_dataset)}")

    # Build model
    model = Rastreador(args).to(device)

    # Load checkpoint
    checkpoint = torch.load(args_cmd.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    print(f"Loaded model from: {args_cmd.model_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    if 'metrics' in checkpoint:
        print(f"  mIoU: {checkpoint['metrics'].get('miou', 'N/A')}")

    # Run evaluation
    evaluate(args, model, eval_loader, args_cmd.output_dir)


if __name__ == '__main__':
    main()
