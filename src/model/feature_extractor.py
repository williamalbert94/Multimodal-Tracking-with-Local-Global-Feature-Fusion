import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from external.lib import pointnet2_utils as pointutils
from external.lib.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG


class PNHead(nn.Module):
    def __init__(self, sample_point_num, in_channels):
        super(PNHead, self).__init__()

        # PointnetSAModuleMSG concatenates 3D relative coords with features
        # So first layer input is 3 (relative coords) + in_channels (features)
        # Output: two branches of 32 channels each = 64 total
        self.sa1 = PointnetSAModuleMSG(
            npoint=sample_point_num,
            radii=[2, 4],
            nsamples=[4, 8],
            mlps=[[3+in_channels, 16, 16, 32], [3+in_channels, 16, 16, 32]]
        )
        # After linear1: 64 -> 32 channels
        # sa2 input: 3 (coords) + 32 (features) = 35
        self.sa2 = PointnetSAModuleMSG(
            npoint=sample_point_num,
            radii=[4, 8],
            nsamples=[8, 16],
            mlps=[[3+32, 32, 32], [3+32, 32, 64]]
        )
        # After linear2: 96 -> 64 channels
        # sa3 input: 3 (coords) + 64 (features) = 67
        self.sa3 = PointnetSAModuleMSG(
            npoint=sample_point_num,
            radii=[8, 16],
            nsamples=[16, 32],
            mlps=[[3+64, 64, 64], [3+64, 64, 64]]
        )

        self.fp3 = PointnetFPModule(mlp=[128, 128])
        self.fp2 = PointnetFPModule(mlp=[160, 128])
        self.fp1 = PointnetFPModule(mlp=[128, 128])

        # Reduce dimensions after each SA layer
        self.linear1 = nn.Linear(64, 32)   # sa1 output: 32+32=64
        self.linear2 = nn.Linear(96, 64)   # sa2 output: 32+64=96
        self.linear3 = nn.Linear(128, 64)  # sa3 output: 64+64=128

    def forward(self, pc, features):
        # pc: [B, N, 3], features: [B, N, C]
        l0_xyz = pc.contiguous()

        # PointNet modules expect features in (B, C, N) format, NOT (B, N, C)!
        # So we need to transpose
        l0_points = features.transpose(1, 2).contiguous()  # [B, N, C] -> [B, C, N]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.linear1(l1_points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.linear2(l2_points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.linear3(l3_points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        return l3_xyz, l0_points

class LocalGlobalFusionSimple(nn.Module):
    def __init__(self, sample_point_num, in_channels):
        super(LocalGlobalFusionSimple, self).__init__()

        self.sample_point_num = sample_point_num

        # PointnetSAModuleMSG concatenates 3D relative coords with features
        # sa1 input: 3 (relative coords) + in_channels (features)
        # sa1 output: 32 + 32 = 64 channels
        self.sa1 = PointnetSAModuleMSG(
            npoint=sample_point_num,
            radii=[2, 4],
            nsamples=[4, 8],
            mlps=[[3+in_channels, 16, 16, 32], [3+in_channels, 16, 16, 32]]
        )
        # After linear1: 64 -> 32 channels
        # sa2 input: 3 (coords) + 32 (features) = 35
        self.sa2 = PointnetSAModuleMSG(
            npoint=sample_point_num,
            radii=[4, 8],
            nsamples=[8, 16],
            mlps=[[3+32, 32, 32], [3+32, 32, 64]]
        )
        # After linear2: 96 -> 64 channels
        # sa3 input: 3 (coords) + 64 (features) = 67
        self.sa3 = PointnetSAModuleMSG(
            npoint=sample_point_num,
            radii=[8, 16],
            nsamples=[16, 32],
            mlps=[[3+64, 64, 64], [3+64, 64, 64]]
        )
        
        # Nuevos módulos para fusión global
        self.global_stats = GlobalStatisticsModule()
        self.fusion_conv = nn.Conv1d(64 + 5, 64, 1)  # 64 canales + 5 estadísticas
        
        # Módulos existentes de FP
        self.fp3 = PointnetFPModule(mlp=[128, 128])
        self.fp2 = PointnetFPModule(mlp=[160, 128])
        self.fp1 = PointnetFPModule(mlp=[128, 128])
        
        # Lineales existentes
        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(96, 64)
        self.linear3 = nn.Linear(128, 64)

    def forward(self, pc, features):
        # pc: [B, N, 3], features: [B, N, C]
        l0_xyz = pc.contiguous()

        # PointNet modules expect features in (B, C, N) format, NOT (B, N, C)!
        # So we need to transpose
        l0_points = features.transpose(1, 2).contiguous()  # [B, N, C] -> [B, C, N]

        # Paso 1: Extraer características locales (downsampling)
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.linear1(l1_points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.linear2(l2_points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.linear3(l3_points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        
        # Paso 2: Extraer estadísticas globales de l3_points (nivel más profundo)
        # l3_points tiene shape: [1, 64, 512]
        global_stats = self.global_stats(l3_points)  # [1, 5, 512]
        
        # Paso 3: Fusionar características locales (l3_points) con estadísticas globales
        l3_fused = torch.cat([l3_points, global_stats], dim=1)  # [1, 69, 512]
        l3_fused = self.fusion_conv(l3_fused)  # [1, 64, 512]
        
        # Paso 4: Upsampling con características fusionadas
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_fused)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        return l3_xyz, l0_points

class GlobalStatisticsModule(nn.Module):
    """Extrae estadísticas globales de las características"""
    def __init__(self):
        super(GlobalStatisticsModule, self).__init__()

    def forward(self, x):
        """
        x: [B, C, N]
        Retorna: [B, 5, N] (mean, std, min, max, range) para cada punto
        """
        batch_size, channels, num_points = x.shape

        # Calcular estadísticas globales a lo largo de la dimensión espacial
        mean = x.mean(dim=2, keepdim=True)  # [B, C, 1]
        std = x.std(dim=2, keepdim=True)    # [B, C, 1]
        max_val = x.max(dim=2, keepdim=True)[0]  # [B, C, 1]
        min_val = x.min(dim=2, keepdim=True)[0]  # [B, C, 1]
        range_val = max_val - min_val  # [B, C, 1]

        # Tomar estadísticas de los primeros canales (o promediar)
        # Vamos a usar las estadísticas del canal promedio
        mean_channel = mean.mean(dim=1, keepdim=True)  # [B, 1, 1]
        std_channel = std.mean(dim=1, keepdim=True)    # [B, 1, 1]
        max_channel = max_val.mean(dim=1, keepdim=True)  # [B, 1, 1]
        min_channel = min_val.mean(dim=1, keepdim=True)  # [B, 1, 1]
        range_channel = range_val.mean(dim=1, keepdim=True)  # [B, 1, 1]

        # Concatenar todas las estadísticas
        stats = torch.cat([
            mean_channel,
            std_channel,
            max_channel,
            min_channel,
            range_channel
        ], dim=1)  # [B, 5, 1]

        # Expandir a todos los puntos
        stats = stats.expand(-1, -1, num_points)  # [B, 5, N]

        return stats


class LocalGlobalFusionGRU(nn.Module):
    """
    LocalGlobalFusionSimple with temporal GRU aggregation (like RaTrack).

    Adds multi-layer GRU to aggregate temporal information across frames,
    matching the architecture in RaTrack's FlowDecoder.

    Args:
        sample_point_num: Number of points to sample
        in_channels: Number of input feature channels
        gru_hidden_size: Hidden state size for GRU (default: 128)
        gru_num_layers: Number of GRU layers (default: 3, RaTrack uses 5)
        gru_dropout: Dropout between GRU layers (default: 0.1)
    """
    def __init__(self, sample_point_num, in_channels,
                 gru_hidden_size=128, gru_num_layers=3, gru_dropout=0.1):
        super(LocalGlobalFusionGRU, self).__init__()

        # Base feature extractor (same as LocalGlobalFusionSimple)
        self.base_extractor = LocalGlobalFusionSimple(sample_point_num, in_channels)

        # GRU for temporal aggregation (like RaTrack)
        # Input: features from base extractor [C, N, B] (permuted for GRU)
        # Output: temporally aggregated features [C, N, B]
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers

        # Project features to GRU hidden size if needed
        self.feature_dim = 128  # Output dimension from LocalGlobalFusionSimple
        if self.feature_dim != gru_hidden_size:
            self.feature_projection = nn.Conv1d(self.feature_dim, gru_hidden_size, 1)
        else:
            self.feature_projection = None

        # Multi-layer GRU for temporal modeling
        # RaTrack uses: nn.GRU(fc_inch // 2, fc_inch // 2, 5)
        self.temporal_gru = nn.GRU(
            input_size=gru_hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=gru_dropout if gru_num_layers > 1 else 0.0,
            batch_first=False  # Input shape: [seq_len, batch, features]
        )

        # Project back to original feature dimension if needed
        if self.feature_dim != gru_hidden_size:
            self.output_projection = nn.Conv1d(gru_hidden_size, self.feature_dim, 1)
        else:
            self.output_projection = None

    def forward(self, pc, features, hidden=None):
        """
        Forward pass with temporal GRU aggregation.

        Args:
            pc: [B, N, 3] point cloud coordinates
            features: [B, N, C] input features
            hidden: [num_layers, B, hidden_size] GRU hidden state from previous frame
                    If None, will be initialized to zeros

        Returns:
            l3_xyz: [B, M, 3] downsampled coordinates
            l0_points: [B, C, N] output features with temporal aggregation
            hidden: [num_layers, B, hidden_size] GRU hidden state for next frame
        """
        batch_size = pc.shape[0]

        # Step 1: Extract spatial features using base extractor
        l3_xyz, l0_points = self.base_extractor(pc, features)
        # l0_points: [B, C, N] where C=128

        # Step 2: Project features to GRU hidden size if needed
        if self.feature_projection is not None:
            l0_points = self.feature_projection(l0_points)  # [B, hidden_size, N]

        # Step 3: Temporal aggregation with GRU
        # GRU expects input shape: [seq_len, batch, features]
        # We treat each point as a sequence element
        num_points = l0_points.shape[2]

        # Reshape for GRU: [B, C, N] -> [N, B, C]
        gru_input = l0_points.permute(2, 0, 1)  # [N, B, hidden_size]

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(
                self.gru_num_layers,
                batch_size,
                self.gru_hidden_size,
                device=l0_points.device
            )

        # Apply GRU
        gru_output, hidden_new = self.temporal_gru(gru_input, hidden)
        # gru_output: [N, B, hidden_size]
        # hidden_new: [num_layers, B, hidden_size]

        # Reshape back: [N, B, C] -> [B, C, N]
        temporal_features = gru_output.permute(1, 2, 0)  # [B, hidden_size, N]

        # Step 4: Project back to original feature dimension if needed
        if self.output_projection is not None:
            temporal_features = self.output_projection(temporal_features)  # [B, C, N]

        return l3_xyz, temporal_features, hidden_new


class LocalGlobalFusionStrong(nn.Module):
    """
    Enhanced PointNet++ backbone with MORE capacity for Re-ID.

    Improvements over LocalGlobalFusionSimple:
      - 4 SA layers (instead of 3) for deeper hierarchies
      - More channels: 128 → 256 → 512 → 256
      - Attention mechanism at highest level
      - Better for discriminative Re-ID embeddings
    """
    def __init__(self, sample_point_num, in_channels):
        super(LocalGlobalFusionStrong, self).__init__()

        self.sample_point_num = sample_point_num

        # ===== ENCODER: 4 SA LAYERS (DEEPER) =====

        # SA1: Multi-scale [2, 4]m → 128 channels (increased from 64)
        self.sa1 = PointnetSAModuleMSG(
            npoint=sample_point_num,
            radii=[2, 4],
            nsamples=[4, 8],
            mlps=[[3+in_channels, 32, 32, 64], [3+in_channels, 32, 32, 64]]
        )
        self.linear1 = nn.Linear(128, 64)  # 64+64=128 → 64

        # SA2: Multi-scale [4, 8]m → 192 channels
        self.sa2 = PointnetSAModuleMSG(
            npoint=sample_point_num,
            radii=[4, 8],
            nsamples=[8, 16],
            mlps=[[3+64, 64, 64], [3+64, 64, 128]]
        )
        self.linear2 = nn.Linear(192, 128)  # 64+128=192 → 128

        # SA3: Multi-scale [8, 16]m → 256 channels
        self.sa3 = PointnetSAModuleMSG(
            npoint=sample_point_num,
            radii=[8, 16],
            nsamples=[16, 32],
            mlps=[[3+128, 128, 128], [3+128, 128, 128]]
        )
        self.linear3 = nn.Linear(256, 128)  # 128+128=256 → 128

        # SA4 (NEW): Multi-scale [16, 32]m → 256 channels (highest abstraction)
        self.sa4 = PointnetSAModuleMSG(
            npoint=sample_point_num // 2,  # Subsample to 256 points
            radii=[16, 32],
            nsamples=[16, 32],
            mlps=[[3+128, 128, 128], [3+128, 128, 128]]
        )
        self.linear4 = nn.Linear(256, 256)  # 128+128=256 → 256

        # Global statistics fusion (at highest level)
        self.global_stats = GlobalStatisticsModule()
        self.fusion_conv = nn.Conv1d(256 + 5, 256, 1)

        # ===== DECODER: FP MODULES =====
        self.fp4 = PointnetFPModule(mlp=[256 + 128, 256])  # Upsample sa4 → sa3
        self.fp3 = PointnetFPModule(mlp=[256 + 128, 128])  # Upsample sa3 → sa2
        self.fp2 = PointnetFPModule(mlp=[128 + 64, 128])   # Upsample sa2 → sa1
        self.fp1 = PointnetFPModule(mlp=[128, 128])        # Upsample sa1 → input

    def forward(self, pc, features):
        """
        Args:
            pc: [B, N, 3] point cloud coordinates
            features: [B, N, C] input features

        Returns:
            l4_xyz: [B, N/2, 3] downsampled coordinates (highest level)
            l0_points: [B, 128, N] per-point features (output)
        """
        # pc: [B, N, 3], features: [B, N, C]
        l0_xyz = pc.contiguous()
        l0_points = features.transpose(1, 2).contiguous()  # [B, N, C] -> [B, C, N]

        # ===== ENCODER: Extract hierarchical features =====

        # SA1: 512 points → 128 channels
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.linear1(l1_points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        # SA2: 512 points → 128 channels
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.linear2(l2_points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        # SA3: 512 points → 128 channels
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.linear3(l3_points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        # SA4 (NEW): 256 points → 256 channels (highest abstraction)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l4_points = self.linear4(l4_points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        # ===== GLOBAL FUSION: Add global context to highest level =====
        global_stats = self.global_stats(l4_points)  # [B, 5, N/2]
        l4_fused = torch.cat([l4_points, global_stats], dim=1)  # [B, 261, N/2]
        l4_fused = self.fusion_conv(l4_fused)  # [B, 256, N/2]

        # ===== DECODER: Upsample with skip connections =====
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_fused)  # → [B, 256, 512]
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # → [B, 128, 512]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # → [B, 128, 512]
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)       # → [B, 128, 512]

        return l4_xyz, l0_points


class SelfAttentionModule(nn.Module):
    """
    Self-Attention module for point cloud features.

    Applies multi-head attention to enhance discriminative features.
    """
    def __init__(self, in_channels, num_heads=4, dropout=0.1):
        super(SelfAttentionModule, self).__init__()

        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.qkv = nn.Conv1d(in_channels, in_channels * 3, 1)
        self.proj_out = nn.Conv1d(in_channels, in_channels, 1)
        self.dropout = nn.Dropout(dropout)

        # Layer norm
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        """
        Args:
            x: [B, C, N] point features
        Returns:
            out: [B, C, N] attention-enhanced features
        """
        B, C, N = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)  # [B, 3*C, N]
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, N)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # [B, num_heads, N, head_dim]
        out = out.permute(0, 1, 3, 2).reshape(B, C, N)  # [B, C, N]

        # Output projection
        out = self.proj_out(out)

        # Residual connection + LayerNorm
        out = out + x  # Residual
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)  # LayerNorm on [B, N, C]

        return out


class LocalGlobalFusionCLIO(nn.Module):
    """
    LocalGlobalFusionStrong + CLIO-style improvements:
      - Self-attention at highest level (l4) for discriminative features
      - Contrastive projection head for CLIO-style temporal contrastive learning
      - Motion-aware feature fusion
      - Enhanced capacity: 4 SA layers + 256 channels + attention

    This backbone is designed for:
      1. PHASE 1: Detection pretraining (box proposal)
      2. PHASE 2: Re-ID with contrastive learning (appearance + motion)
    """
    def __init__(self, sample_point_num, in_channels, contrastive_dim=256):
        super(LocalGlobalFusionCLIO, self).__init__()

        self.sample_point_num = sample_point_num
        self.contrastive_dim = contrastive_dim

        # ===== ENCODER: 4 SA LAYERS (DEEPER) =====

        # SA1: Multi-scale [2, 4]m → 128 channels
        self.sa1 = PointnetSAModuleMSG(
            npoint=sample_point_num,
            radii=[2, 4],
            nsamples=[4, 8],
            mlps=[[3+in_channels, 32, 32, 64], [3+in_channels, 32, 32, 64]]
        )
        self.linear1 = nn.Linear(128, 64)

        # SA2: Multi-scale [4, 8]m → 192 channels
        self.sa2 = PointnetSAModuleMSG(
            npoint=sample_point_num,
            radii=[4, 8],
            nsamples=[8, 16],
            mlps=[[3+64, 64, 64], [3+64, 64, 128]]
        )
        self.linear2 = nn.Linear(192, 128)

        # SA3: Multi-scale [8, 16]m → 256 channels
        self.sa3 = PointnetSAModuleMSG(
            npoint=sample_point_num,
            radii=[8, 16],
            nsamples=[16, 32],
            mlps=[[3+128, 128, 128], [3+128, 128, 128]]
        )
        self.linear3 = nn.Linear(256, 128)

        # SA4: Multi-scale [16, 32]m → 256 channels (highest abstraction)
        self.sa4 = PointnetSAModuleMSG(
            npoint=sample_point_num // 2,  # Subsample to 256 points
            radii=[16, 32],
            nsamples=[16, 32],
            mlps=[[3+128, 128, 128], [3+128, 128, 128]]
        )
        self.linear4 = nn.Linear(256, 256)

        # ===== SELF-ATTENTION (NEW) =====
        # Apply attention at highest level for discriminative features
        self.self_attention = SelfAttentionModule(
            in_channels=256,
            num_heads=8,
            dropout=0.1
        )

        # Global statistics fusion
        self.global_stats = GlobalStatisticsModule()
        self.fusion_conv = nn.Conv1d(256 + 5, 256, 1)

        # ===== DECODER: FP MODULES =====
        self.fp4 = PointnetFPModule(mlp=[256 + 128, 256])
        self.fp3 = PointnetFPModule(mlp=[256 + 128, 128])
        self.fp2 = PointnetFPModule(mlp=[128 + 64, 128])
        self.fp1 = PointnetFPModule(mlp=[128, 128])

        # ===== CONTRASTIVE PROJECTION HEAD (CLIO-style) =====
        # Projects global features to contrastive space for temporal matching
        self.contrastive_projection = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, contrastive_dim, 1)
        )

        # ===== MOTION-AWARE FUSION (optional, for Phase 2) =====
        # Fuses ego motion representation with spatial features
        self.motion_fusion = nn.Sequential(
            nn.Conv1d(128 + 3, 128, 1),  # 128 spatial + 3 motion (dx, dy, dz)
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

    def forward(self, pc, features, ego_motion=None, return_contrastive=False):
        """
        Args:
            pc: [B, N, 3] point cloud coordinates
            features: [B, N, C] input features
            ego_motion: [B, 3] ego motion vector (dx, dy, dz) - optional
            return_contrastive: bool, return contrastive projection for CLIO loss

        Returns:
            l4_xyz: [B, N/2, 3] downsampled coordinates (highest level)
            l0_points: [B, 128, N] per-point features (output)
            contrastive_proj: [B, contrastive_dim] global contrastive features (if return_contrastive=True)
        """
        B = pc.shape[0]

        # pc: [B, N, 3], features: [B, N, C]
        l0_xyz = pc.contiguous()
        l0_points = features.transpose(1, 2).contiguous()  # [B, N, C] -> [B, C, N]

        # ===== ENCODER: Extract hierarchical features =====

        # SA1: 512 points → 64 channels
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.linear1(l1_points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        # SA2: 512 points → 128 channels
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.linear2(l2_points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        # SA3: 512 points → 128 channels
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.linear3(l3_points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        # SA4: 256 points → 256 channels (highest abstraction)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l4_points = self.linear4(l4_points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        # ===== SELF-ATTENTION: Enhance discriminative features =====
        l4_points = self.self_attention(l4_points)  # [B, 256, N/2]

        # ===== GLOBAL FUSION: Add global context =====
        global_stats = self.global_stats(l4_points)  # [B, 5, N/2]
        l4_fused = torch.cat([l4_points, global_stats], dim=1)  # [B, 261, N/2]
        l4_fused = self.fusion_conv(l4_fused)  # [B, 256, N/2]

        # ===== CONTRASTIVE PROJECTION (for CLIO loss) =====
        contrastive_proj = None
        if return_contrastive:
            # Global max pooling
            global_feat = l4_fused.max(dim=2)[0]  # [B, 256]
            # Project to contrastive space
            contrastive_proj = self.contrastive_projection(global_feat.unsqueeze(2)).squeeze(2)  # [B, contrastive_dim]
            # L2 normalize for cosine similarity
            contrastive_proj = F.normalize(contrastive_proj, dim=1)

        # ===== DECODER: Upsample with skip connections =====
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_fused)  # → [B, 256, 512]
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # → [B, 128, 512]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # → [B, 128, 512]
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)       # → [B, 128, 512]

        # ===== MOTION FUSION (optional, for Phase 2 Re-ID) =====
        if ego_motion is not None:
            # Expand ego motion to all points: [B, 3] -> [B, 3, N]
            ego_motion_expanded = ego_motion.unsqueeze(2).expand(-1, -1, l0_points.shape[2])
            # Concatenate with spatial features
            l0_points_motion = torch.cat([l0_points, ego_motion_expanded], dim=1)  # [B, 131, N]
            # Fuse
            l0_points = self.motion_fusion(l0_points_motion)  # [B, 128, N]

        if return_contrastive:
            return l4_xyz, l0_points, contrastive_proj
        else:
            return l4_xyz, l0_points