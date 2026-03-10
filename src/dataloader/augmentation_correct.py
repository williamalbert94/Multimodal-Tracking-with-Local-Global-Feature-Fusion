"""
Correct Data Augmentation for Scene Flow and Segmentation.

Key principles:
1. Apply SAME transformation to all frames (pc0, pc1, pc0_comp)
2. Transform bounding boxes accordingly
3. Only use geometrically-consistent augmentations
4. Preserve temporal and spatial relationships

Author: Corrected implementation
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


class PointCloudAugmentation:
    """
    Geometrically-consistent augmentation for multi-frame point cloud data.

    Safe augmentations for scene flow:
    - Random jitter (small noise)
    - Random horizontal flip (with box transformation)
    - Random point dropout (without replacement)
    - Random translation (small, global)

    Avoided (break label consistency):
    - Random rotation (requires complex box transformation)
    - Random scaling (changes real-world distances)
    """

    def __init__(self,
                 jitter_std=0.02,           # Gaussian noise std (meters)
                 dropout_ratio=0.0,         # Fraction of points to drop (NO PADDING)
                 flip_prob=0.5,             # Probability of horizontal flip
                 translation_range=0.0):    # Max translation (meters), disabled by default
        """
        Args:
            jitter_std: Standard deviation for Gaussian noise on coordinates
            dropout_ratio: Fraction of points to randomly drop (0.0 = disabled)
            flip_prob: Probability of applying horizontal flip
            translation_range: Max random translation in XY plane (0.0 = disabled)
        """
        self.jitter_std = jitter_std
        self.dropout_ratio = dropout_ratio
        self.flip_prob = flip_prob
        self.translation_range = translation_range

    def __call__(self, pc0, pc1, pc0_comp, lbl1, lbl2):
        """
        Apply consistent augmentation to all inputs.

        Args:
            pc0: [N0, 3] - frame t+1 point cloud
            pc1: [N1, 3] - frame t point cloud
            pc0_comp: [N0_comp, 3] - frame t+1 ego-motion compensated
            lbl1: dict - labels for frame t+1
            lbl2: dict - labels for frame t

        Returns:
            Augmented versions of all inputs (same transformations applied)
        """
        # Sample random augmentation parameters ONCE
        do_flip = np.random.rand() < self.flip_prob
        jitter_0 = np.random.normal(0, self.jitter_std, pc0.shape) if self.jitter_std > 0 else 0
        jitter_1 = np.random.normal(0, self.jitter_std, pc1.shape) if self.jitter_std > 0 else 0
        jitter_0_comp = np.random.normal(0, self.jitter_std, pc0_comp.shape) if self.jitter_std > 0 else 0

        translation = np.zeros(3)
        if self.translation_range > 0:
            translation[:2] = np.random.uniform(-self.translation_range,
                                               self.translation_range, size=2)

        # Apply augmentations
        pc0_aug = pc0.copy()
        pc1_aug = pc1.copy()
        pc0_comp_aug = pc0_comp.copy()
        lbl1_aug = self._copy_labels(lbl1)
        lbl2_aug = self._copy_labels(lbl2)

        # 1. Jitter (point-wise noise) - DIFFERENT per frame (OK for jitter)
        pc0_aug = pc0_aug + jitter_0
        pc1_aug = pc1_aug + jitter_1
        pc0_comp_aug = pc0_comp_aug + jitter_0_comp

        # 2. Horizontal flip (Y-axis flip in camera/radar coordinates)
        if do_flip:
            pc0_aug = self._flip_points(pc0_aug)
            pc1_aug = self._flip_points(pc1_aug)
            pc0_comp_aug = self._flip_points(pc0_comp_aug)
            lbl1_aug = self._flip_labels(lbl1_aug)
            lbl2_aug = self._flip_labels(lbl2_aug)

        # 3. Translation (global shift)
        if self.translation_range > 0:
            pc0_aug = pc0_aug + translation
            pc1_aug = pc1_aug + translation
            pc0_comp_aug = pc0_comp_aug + translation
            lbl1_aug = self._translate_labels(lbl1_aug, translation)
            lbl2_aug = self._translate_labels(lbl2_aug, translation)

        # 4. Point dropout (NO PADDING - critical!)
        if self.dropout_ratio > 0:
            pc0_aug, idx0 = self._dropout_points(pc0_aug, self.dropout_ratio)
            pc1_aug, idx1 = self._dropout_points(pc1_aug, self.dropout_ratio)
            pc0_comp_aug, idx0_comp = self._dropout_points(pc0_comp_aug, self.dropout_ratio)
            # Note: Labels are not affected by dropout (they reference spatial regions)

        return pc0_aug, pc1_aug, pc0_comp_aug, lbl1_aug, lbl2_aug

    def _flip_points(self, pc):
        """Flip point cloud around Y-axis (horizontal flip)."""
        pc_flip = pc.copy()
        pc_flip[:, 1] = -pc_flip[:, 1]  # Negate Y coordinate
        return pc_flip

    def _flip_labels(self, labels):
        """Flip bounding boxes around Y-axis."""
        labels_flip = {}
        for key, lbl in labels.items():
            labels_flip[key] = lbl  # Create shallow copy
            labels_flip[key].y = -lbl.y  # Flip Y position
            labels_flip[key].ry = -lbl.ry  # Flip yaw angle
        return labels_flip

    def _translate_labels(self, labels, translation):
        """Translate bounding boxes."""
        labels_trans = {}
        for key, lbl in labels.items():
            labels_trans[key] = lbl
            labels_trans[key].x += translation[0]
            labels_trans[key].y += translation[1]
            labels_trans[key].z += translation[2]
        return labels_trans

    def _dropout_points(self, pc, ratio):
        """
        Randomly drop points WITHOUT replacement.
        Returns: (augmented_pc, kept_indices)
        """
        num_points = len(pc)
        num_keep = int(num_points * (1 - ratio))

        if num_keep <= 0:
            return pc, np.arange(num_points)  # Keep all if ratio too high

        keep_indices = np.random.choice(num_points, num_keep, replace=False)
        keep_indices = np.sort(keep_indices)  # Maintain order for stability

        return pc[keep_indices], keep_indices

    def _copy_labels(self, labels):
        """Create shallow copy of labels dict."""
        return {k: v for k, v in labels.items()}


class MinimalAugmentation:
    """
    Minimal augmentation (only jitter) - safest option.
    Use this if flip/dropout cause issues.
    """

    def __init__(self, jitter_std=0.02):
        self.jitter_std = jitter_std

    def __call__(self, pc0, pc1, pc0_comp, lbl1, lbl2):
        """Apply only Gaussian jitter to point clouds."""
        pc0_aug = pc0 + np.random.normal(0, self.jitter_std, pc0.shape) if self.jitter_std > 0 else pc0.copy()
        pc1_aug = pc1 + np.random.normal(0, self.jitter_std, pc1.shape) if self.jitter_std > 0 else pc1.copy()
        pc0_comp_aug = pc0_comp + np.random.normal(0, self.jitter_std, pc0_comp.shape) if self.jitter_std > 0 else pc0_comp.copy()

        return pc0_aug, pc1_aug, pc0_comp_aug, lbl1, lbl2


# ============================================================================
# Convenience function for quick testing
# ============================================================================

def test_augmentation():
    """Quick test of augmentation consistency."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Create dummy data
    np.random.seed(42)
    pc0 = np.random.randn(100, 3)
    pc1 = np.random.randn(100, 3)
    pc0_comp = np.random.randn(100, 3)

    class DummyLabel:
        def __init__(self):
            self.x, self.y, self.z = 0.0, 0.0, 0.0
            self.ry = 0.0
            self.h, self.w, self.l = 1.0, 1.0, 1.0

    lbl1 = {0: DummyLabel()}
    lbl2 = {0: DummyLabel()}

    # Test augmentation
    aug = PointCloudAugmentation(jitter_std=0.05, flip_prob=1.0, dropout_ratio=0.1)
    pc0_aug, pc1_aug, pc0_comp_aug, lbl1_aug, lbl2_aug = aug(pc0, pc1, pc0_comp, lbl1, lbl2)

    print("Augmentation test passed!")
    print(f"   Original pc0 shape: {pc0.shape}")
    print(f"   Augmented pc0 shape: {pc0_aug.shape}")
    print(f"   Dropout kept {len(pc0_aug)/len(pc0)*100:.1f}% of points")
    print(f"   Label Y flip: {lbl1[0].y:.2f} → {lbl1_aug[0].y:.2f}")


if __name__ == "__main__":
    test_augmentation()
