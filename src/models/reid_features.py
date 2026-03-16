"""
Re-ID Feature Extractor (Improved)
Extrae embeddings discriminativos para Re-Identification de objetos.
Usa LocalGlobalFusionSimple (backbone robusto de PointNet++) del modelo principal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import logging

logger = logging.getLogger(__name__)

# Add parent directory to path to import LocalGlobalFusion backbones
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.feature_extractor import (
    LocalGlobalFusionSimple,
    LocalGlobalFusionStrong,
    LocalGlobalFusionCLIO
)


# ===== CLASS LABEL MAPPING =====
# Mapeo de nombres de clases a índices (VOD dataset)
CLASS_TO_IDX = {
    'car': 0,
    'pedestrian': 1,
    'cyclist': 1,
    'rider': 1,  # Rider se agrupa con cyclist
    'van': 0,    # Van se agrupa con car
    'truck': 0,  # Truck se agrupa con car
    # NOTE: pedestrian removed from pipeline (low radar resolution)
    'person_sitting': 2,  # Unknown class (pedestrian removed)
    'unknown': 2,  # Clase desconocida
}

IDX_TO_CLASS = {
    0: 'vehicle',     # Car, Van, Truck
    # NOTE: pedestrian removed (was index 1)
    1: 'cyclist',     # Cyclist, Rider
    2: 'unknown',
}

NUM_CLASSES = 3  # vehicle, cyclist, unknown (pedestrian removed)


def class_name_to_idx(class_name):
    """Convert class name to index."""
    if class_name is None:
        return 2  # unknown
    return CLASS_TO_IDX.get(class_name.lower(), 2)


class ReIDFeatureExtractor(nn.Module):
    """
    Extrae embeddings Re-ID usando backbone PointNet++ robusto.

    Arquitectura:
      - LocalGlobalFusionSimple (PointNet++ multi-scale) como backbone
      - Re-ID head para procesamiento específico de Re-ID
      - Box geometry encoding
      - Fusion de appearance + geometry
      - Max + Avg pooling (más robusto)
      - L2 normalization para metric learning
    """

    def __init__(self, embedding_dim=256, sample_point_num=512, use_pretrained_backbone=False,
                 checkpoint_path=None, backbone_name='LocalGlobalFusionSimple'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sample_point_num = sample_point_num
        self.backbone_name = backbone_name

        # Select backbone based on configuration
        if backbone_name == 'LocalGlobalFusionCLIO':
            # CLIO backbone: 4 SA layers + self-attention + contrastive projection
            self.backbone = LocalGlobalFusionCLIO(
                sample_point_num=sample_point_num,
                in_channels=3,
                contrastive_dim=256
            )
            backbone_output_dim = 128  # LocalGlobalFusionCLIO outputs [B, 128, N]
        elif backbone_name == 'LocalGlobalFusionStrong':
            # Enhanced backbone: 4 SA layers, 256 channels output
            self.backbone = LocalGlobalFusionStrong(
                sample_point_num=sample_point_num,
                in_channels=3
            )
            backbone_output_dim = 128  # LocalGlobalFusionStrong outputs [B, 128, N]
        else:
            # Default backbone: 3 SA layers, 128 channels output
            self.backbone = LocalGlobalFusionSimple(
                sample_point_num=sample_point_num,
                in_channels=3
            )
            backbone_output_dim = 128  # LocalGlobalFusionSimple outputs [B, 128, N]

        # Re-ID head: convierte features per-point → features más discriminativas
        # Adapts to backbone output dimension
        self.reid_head = nn.Sequential(
            nn.Conv1d(backbone_output_dim, 256, 1),  # Per-point: backbone_dim → 256
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(256, 512, 1),      # Per-point: 256 → 512
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        # Box geometry encoder (más profundo que antes)
        # Usa LayerNorm en lugar de BatchNorm1d para soportar batch_size=1
        self.box_encoder = nn.Sequential(
            nn.Linear(7, 64),  # x, y, z, l, w, h, yaw
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        # Fusion network (appearance 512 + geometry 64 = 576)
        # Usa LayerNorm en lugar de BatchNorm1d para soportar batch_size=1
        self.fusion = nn.Sequential(
            nn.Linear(512 + 64, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # ===== CLASSIFICATION HEAD (NEW) =====
        # Clasifica objetos en: vehicle, pedestrian, cyclist, unknown
        # Comparte embeddings con Re-ID → embeddings class-aware
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES),  # 3 classes: vehicle, cyclist, unknown (pedestrian removed)
        )

        # Opcionalmente cargar pesos pre-entrenados del backbone
        if use_pretrained_backbone and checkpoint_path:
            self._load_pretrained_backbone(checkpoint_path)

    def _load_pretrained_backbone(self, checkpoint_path):
        """Carga pesos del backbone desde checkpoint de segmentación"""
        logger.info("=" * 70)
        logger.info("LOADING PRETRAINED BACKBONE")
        logger.info("=" * 70)
        logger.info(f"   Checkpoint path: {checkpoint_path}")
        logger.info(f"   Backbone type: {self.backbone_name}")

        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            logger.info(f"   ✓ Checkpoint loaded successfully")

            # Log checkpoint structure
            checkpoint_keys = list(checkpoint.keys())
            logger.info(f"   Checkpoint keys: {checkpoint_keys}")

            if 'model_state_dict' not in checkpoint:
                logger.error(f"   No 'model_state_dict' key found in checkpoint!")
                logger.error(f"   Available keys: {checkpoint_keys}")
                return

            # Get all model keys
            all_model_keys = list(checkpoint['model_state_dict'].keys())
            logger.info(f"   Total model keys in checkpoint: {len(all_model_keys)}")
            logger.info(f"   First 5 keys: {all_model_keys[:5]}")

            # Intentar cargar solo los pesos del backbone
            backbone_state = {}
            matched_keys = []

            for key, value in checkpoint['model_state_dict'].items():
                # Try multiple matching patterns
                if 'feature_extractor' in key or 'backbone' in key or 'extractor' in key:
                    # Remover prefijos comunes
                    new_key = key.replace('feature_extractor.', '').replace('backbone.', '').replace('extractor.', '')
                    backbone_state[new_key] = value
                    matched_keys.append(key)

            logger.info(f"   Matched {len(matched_keys)} backbone keys")
            if len(matched_keys) > 0:
                logger.info(f"   Sample matched keys: {matched_keys[:5]}")

            if backbone_state:
                # Try to load state dict
                missing_keys, unexpected_keys = self.backbone.load_state_dict(backbone_state, strict=False)

                logger.info("   " + "=" * 66)
                logger.info(f"   LOADED PRETRAINED BACKBONE SUCCESSFULLY")
                logger.info("   " + "=" * 66)
                logger.info(f"   Loaded parameters: {len(backbone_state)}")
                logger.info(f"   Missing keys: {len(missing_keys)}")
                logger.info(f"   Unexpected keys: {len(unexpected_keys)}")

                if len(missing_keys) > 0:
                    logger.warning(f"   Missing keys (first 5): {list(missing_keys)[:5]}")
                if len(unexpected_keys) > 0:
                    logger.warning(f"   Unexpected keys (first 5): {list(unexpected_keys)[:5]}")

                # Count loaded parameters
                loaded_params = sum(p.numel() for p in self.backbone.parameters())
                logger.info(f"   Backbone parameters: {loaded_params:,}")
                logger.info("   " + "=" * 66)
            else:
                logger.error("   " + "=" * 66)
                logger.error(f"   NO BACKBONE WEIGHTS FOUND IN CHECKPOINT")
                logger.error("   " + "=" * 66)
                logger.error(f"   Searched for keys containing: 'feature_extractor', 'backbone', 'extractor'")
                logger.error(f"   Sample checkpoint keys: {all_model_keys[:10]}")
                logger.error("   " + "=" * 66)

        except Exception as e:
            logger.error("   " + "=" * 66)
            logger.error(f"   COULD NOT LOAD PRETRAINED BACKBONE")
            logger.error("   " + "=" * 66)
            logger.error(f"   Error: {e}")
            logger.error(f"   Error type: {type(e).__name__}")
            import traceback
            logger.error(f"   Traceback:\n{traceback.format_exc()}")
            logger.error("   " + "=" * 66)

    def freeze_backbone(self):
        """Freeze backbone parameters for training only Re-ID head."""
        logger.info("🔒 Freezing backbone weights (training Re-ID head only)")

        # Freeze backbone params but KEEP in train mode to preserve gradient flow
        # (Setting to eval() breaks gradient propagation through frozen layers)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Freeze BatchNorm layers specifically (set to eval mode)
        for module in self.backbone.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                module.eval()

        # Ensure reid_head and other layers are trainable and in train mode
        self.reid_head.train()
        self.box_encoder.train()
        self.fusion.train()
        self.classifier.train()

        for param in self.reid_head.parameters():
            param.requires_grad = True
        for param in self.box_encoder.parameters():
            param.requires_grad = True
        for param in self.fusion.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

        # Count trainable params for verification
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"   Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for end-to-end fine-tuning."""
        logger.info("🔓 Unfreezing backbone weights (end-to-end fine-tuning)")
        # Unfreeze backbone and set to train mode
        self.backbone.train()
        for param in self.backbone.parameters():
            param.requires_grad = True

        # Count trainable params for verification
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"   Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def forward(self, points, boxes, cluster_ids):
        """
        Args:
            points: [B, N, 3] all points
            boxes: List[Tensor] boxes per batch [M_i, 7]
            cluster_ids: List[Tensor] cluster ID per point [N_i]

        Returns:
            embeddings: List[Tensor] embeddings per box [M_i, D]
            class_logits: List[Tensor] class predictions [M_i, NUM_CLASSES]
        """
        batch_size = points.shape[0]
        embeddings_batch = []
        class_logits_batch = []

        for b in range(batch_size):
            pts = points[b]  # [N, 3]
            bxs = boxes[b]   # [M, 7]
            ids = cluster_ids[b]  # [N]

            if len(bxs) == 0:
                # No boxes in this frame
                embeddings_batch.append(torch.zeros(0, self.embedding_dim).to(points.device))
                class_logits_batch.append(torch.zeros(0, NUM_CLASSES).to(points.device))
                continue

            # Extract features per box
            box_embeddings = []
            box_class_logits = []
            for box_id in range(len(bxs)):
                # Get points belonging to this box
                box_mask = (ids == box_id)
                box_pts = pts[box_mask]  # [K, 3]

                if len(box_pts) < 10:  # Mínimo de puntos (aumentado de 3 a 10)
                    # Not enough points - use zero embedding and logits
                    box_emb = torch.zeros(self.embedding_dim).to(points.device)
                    box_logits = torch.zeros(NUM_CLASSES).to(points.device)
                else:
                    # Extract appearance features usando el backbone robusto
                    appearance_feat = self._extract_box_features(box_pts)  # [512]

                    # Encode box geometry (LayerNorm works with any shape)
                    box_param = bxs[box_id]  # [7]
                    geometry_feat = self.box_encoder(box_param)  # [64]

                    # Fuse appearance + geometry
                    combined = torch.cat([appearance_feat, geometry_feat], dim=0)  # [576]
                    box_emb = self.fusion(combined)  # [D]

                    # Predict class logits
                    box_logits = self.classifier(box_emb)  # [NUM_CLASSES]

                box_embeddings.append(box_emb)
                box_class_logits.append(box_logits)

            embeddings = torch.stack(box_embeddings)  # [M, D]
            class_logits = torch.stack(box_class_logits)  # [M, NUM_CLASSES]

            # L2 normalize embeddings for metric learning
            embeddings = F.normalize(embeddings, p=2, dim=1)

            embeddings_batch.append(embeddings)
            class_logits_batch.append(class_logits)

        return embeddings_batch, class_logits_batch

    def _extract_box_features(self, box_points):
        """
        Extrae features de puntos de una caja usando LocalGlobalFusionSimple (PointNet++).

        Args:
            box_points: [K, 3] puntos en la caja

        Returns:
            features: [512] embedding global
        """
        # Preparar input para el backbone
        pts = box_points.unsqueeze(0)  # [1, K, 3]

        # Features iniciales: usar las coordenadas como features
        # El backbone espera [B, N, C] donde C son los canales de features
        feats = box_points.unsqueeze(0)  # [1, K, 3]

        try:
            # Forward a través del backbone PointNet++ robusto
            l3_xyz, l0_points = self.backbone(pts, feats)
            # l3_xyz: [1, N_sample, 3] (downsampled points)
            # l0_points: [1, 128, K] (per-point features del backbone)

            # Pasar por Re-ID head (procesamiento específico de Re-ID)
            x = self.reid_head(l0_points)  # [1, 512, K]

            # Global max + avg pooling (más robusto que solo max)
            # Combinación de max y avg pooling es común en Re-ID de imágenes
            max_pool = torch.max(x, dim=2)[0]  # [1, 512]
            avg_pool = torch.mean(x, dim=2)    # [1, 512]

            # Combinar max + avg (captura tanto features salientes como promedio)
            global_feat = max_pool + avg_pool  # [1, 512]

            return global_feat.squeeze(0)  # [512]

        except Exception as e:
            # Fallback: si el backbone falla (ej: muy pocos puntos), usar pooling simple
            print(f"Backbone forward failed for box: {e}, using fallback")
            # Simple mean pooling sobre las coordenadas
            fallback_feat = torch.zeros(512).to(box_points.device)
            return fallback_feat


class TripletLoss(nn.Module):
    """
    Triplet loss for Re-ID learning.

    For each anchor (object at time t):
      - Positive: Same object at time t+1 (same track ID)
      - Negative: Different object

    Goal: ||anchor - positive||² < ||anchor - negative||² + margin
    """

    def __init__(self, margin=0.3, hard_mining=True):
        super().__init__()
        self.margin = margin
        self.hard_mining = hard_mining

    def forward(self, embeddings_t, embeddings_t1, track_ids_t, track_ids_t1):
        """
        Args:
            embeddings_t: List[Tensor] embeddings at time t [M_i, D]
            embeddings_t1: List[Tensor] embeddings at time t+1 [M_i', D]
            track_ids_t: List[Tensor] track IDs at time t [M_i]
            track_ids_t1: List[Tensor] track IDs at time t+1 [M_i']

        Returns:
            loss: scalar triplet loss
        """
        total_loss = 0
        num_triplets = 0

        batch_size = len(embeddings_t)

        for b in range(batch_size):
            emb_t = embeddings_t[b]      # [M, D]
            emb_t1 = embeddings_t1[b]    # [M', D]
            ids_t = track_ids_t[b]       # [M]
            ids_t1 = track_ids_t1[b]     # [M']

            if len(emb_t) == 0 or len(emb_t1) == 0:
                continue

            # For each anchor at time t
            for i in range(len(emb_t)):
                anchor = emb_t[i]        # [D]
                anchor_id = ids_t[i].item()

                # Skip if anchor_id is invalid (e.g., -1, 0)
                if anchor_id <= 0:
                    continue

                # Find positive (same ID at t+1)
                pos_mask = (ids_t1 == anchor_id)
                if pos_mask.sum() == 0:
                    continue  # Object disappeared or not tracked

                positive = emb_t1[pos_mask][0]  # [D] - take first match

                # Find negatives (different IDs at t+1)
                neg_mask = (ids_t1 != anchor_id) & (ids_t1 > 0)  # Valid IDs only
                if neg_mask.sum() == 0:
                    continue  # No negatives available

                negatives = emb_t1[neg_mask]  # [K, D]

                # Compute triplet loss
                if self.hard_mining:
                    # Hard negative mining: closest negative
                    loss = self._hard_triplet_loss(anchor, positive, negatives)
                else:
                    # All negatives
                    loss = self._all_triplets_loss(anchor, positive, negatives)

                total_loss += loss
                num_triplets += 1

        if num_triplets == 0:
            return torch.tensor(0.0).to(embeddings_t[0].device if len(embeddings_t[0]) > 0 else embeddings_t1[0].device)

        return total_loss / num_triplets

    def _hard_triplet_loss(self, anchor, positive, negatives):
        """
        Hard triplet loss: use hardest negative (closest to anchor).
        """
        # Distance to positive
        pos_dist = F.pairwise_distance(anchor.unsqueeze(0), positive.unsqueeze(0))

        # Distances to all negatives
        neg_dists = F.pairwise_distance(
            anchor.unsqueeze(0).expand(len(negatives), -1),
            negatives
        )

        # Hard negative: closest to anchor (smallest distance)
        hard_neg_dist = neg_dists.min()

        # Triplet loss: max(0, pos_dist - neg_dist + margin)
        loss = F.relu(pos_dist - hard_neg_dist + self.margin)

        return loss

    def _all_triplets_loss(self, anchor, positive, negatives):
        """
        All triplets loss: average over all negatives.
        """
        # Distance to positive
        pos_dist = F.pairwise_distance(anchor.unsqueeze(0), positive.unsqueeze(0))

        # Distances to all negatives
        neg_dists = F.pairwise_distance(
            anchor.unsqueeze(0).expand(len(negatives), -1),
            negatives
        )

        # Triplet loss for all negatives
        losses = F.relu(pos_dist - neg_dists + self.margin)

        return losses.mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss (alternative to triplet loss).
    Simpler, pero puede ser efectivo.
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings_t, embeddings_t1, track_ids_t, track_ids_t1):
        """
        Args:
            embeddings_t: List[Tensor] embeddings at time t
            embeddings_t1: List[Tensor] embeddings at time t+1
            track_ids_t: List[Tensor] track IDs at time t
            track_ids_t1: List[Tensor] track IDs at time t+1

        Returns:
            loss: contrastive loss
        """
        total_loss = 0
        num_pairs = 0

        batch_size = len(embeddings_t)

        for b in range(batch_size):
            emb_t = embeddings_t[b]
            emb_t1 = embeddings_t1[b]
            ids_t = track_ids_t[b]
            ids_t1 = track_ids_t1[b]

            if len(emb_t) == 0 or len(emb_t1) == 0:
                continue

            # Pairwise distances
            for i in range(len(emb_t)):
                for j in range(len(emb_t1)):
                    anchor_id = ids_t[i].item()
                    other_id = ids_t1[j].item()

                    if anchor_id <= 0 or other_id <= 0:
                        continue

                    # Distance
                    dist = F.pairwise_distance(
                        emb_t[i].unsqueeze(0),
                        emb_t1[j].unsqueeze(0)
                    )

                    # Positive pair (same ID)
                    if anchor_id == other_id:
                        loss = dist ** 2  # Minimize distance
                    # Negative pair (different ID)
                    else:
                        loss = F.relu(self.margin - dist) ** 2  # Maximize distance

                    total_loss += loss
                    num_pairs += 1

        if num_pairs == 0:
            return torch.tensor(0.0).to(embeddings_t[0].device if len(embeddings_t[0]) > 0 else embeddings_t1[0].device)

        return total_loss / num_pairs


class ClassificationLoss(nn.Module):
    """
    Classification loss for object class prediction.
    Uses CrossEntropyLoss to classify objects into: vehicle, pedestrian, cyclist, unknown.

    This loss makes embeddings class-aware → better Re-ID performance.
    """

    def __init__(self):
        super().__init__()
        # CrossEntropyLoss with label smoothing for better generalization
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, class_logits_batch, class_labels_batch):
        """
        Args:
            class_logits_batch: List[Tensor] class predictions [M_i, NUM_CLASSES]
            class_labels_batch: List[dict] GT class labels {obj_id: 'car'}

        Returns:
            loss: scalar classification loss
            accuracy: classification accuracy (for logging)
        """
        total_loss = 0
        total_correct = 0
        total_samples = 0

        batch_size = len(class_logits_batch)

        for b in range(batch_size):
            logits = class_logits_batch[b]  # [M, NUM_CLASSES]
            labels_dict = class_labels_batch[b] if b < len(class_labels_batch) else {}

            if len(logits) == 0 or len(labels_dict) == 0:
                continue

            # Convert class labels dict to tensor
            # labels_dict format: {obj_id: 'car', obj_id: 'pedestrian', ...}
            # We assume logits[i] corresponds to obj_id=i (in order)
            gt_labels = []
            for obj_id in sorted(labels_dict.keys()):
                class_name = labels_dict[obj_id]
                class_idx = class_name_to_idx(class_name)
                gt_labels.append(class_idx)

            if len(gt_labels) == 0:
                continue

            # Handle case where logits and labels don't match in length
            # (may happen if some boxes were filtered out)
            num_samples = min(len(logits), len(gt_labels))
            if num_samples == 0:
                continue

            logits = logits[:num_samples]  # [N, NUM_CLASSES]
            gt_labels = torch.tensor(gt_labels[:num_samples]).long().to(logits.device)  # [N]

            # Compute classification loss
            loss = self.criterion(logits, gt_labels)
            total_loss += loss

            # Compute accuracy
            predictions = torch.argmax(logits, dim=1)  # [N]
            correct = (predictions == gt_labels).sum().item()
            total_correct += correct
            total_samples += num_samples

        if total_samples == 0:
            # No valid samples
            dummy_device = class_logits_batch[0].device if len(class_logits_batch) > 0 and len(class_logits_batch[0]) > 0 else torch.device('cuda')
            return torch.tensor(0.0).to(dummy_device), 0.0

        avg_loss = total_loss / batch_size if batch_size > 0 else total_loss
        accuracy = total_correct / total_samples

        return avg_loss, accuracy
