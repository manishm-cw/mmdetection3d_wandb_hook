"""
Custom WandB Hook for MMDetection3D
Logs 3D point clouds with detections and masks to Weights & Biases
"""

import numpy as np
import torch
from typing import Optional, Sequence
from mmengine.hooks import Hook
from mmengine.runner import Runner

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


class WandB3DDetectionHook(Hook):
    """
    Custom hook to log 3D detections with point clouds and masks to WandB.

    This hook captures predictions during validation/testing and logs them as
    3D point clouds with bounding boxes to Weights & Biases for visualization.

    Args:
        interval (int): Logging interval (log every N iterations). Default: 50
        max_samples_per_epoch (int): Maximum number of samples to log per epoch. Default: 10
        log_train (bool): Whether to log training predictions. Default: False
        log_val (bool): Whether to log validation predictions. Default: True
        log_test (bool): Whether to log test predictions. Default: True
        score_threshold (float): Minimum score threshold for logging boxes. Default: 0.3
        max_points (int): Maximum number of points to log (WandB limit is 300k). Default: 200000
    """

    priority = 'NORMAL'

    def __init__(
        self,
        interval: int = 50,
        max_samples_per_epoch: int = 10,
        log_train: bool = False,
        log_val: bool = True,
        log_test: bool = True,
        score_threshold: float = 0.3,
        max_points: int = 200000,
    ):
        super().__init__()

        if not WANDB_AVAILABLE:
            raise ImportError(
                "wandb is required for WandB3DDetectionHook. "
                "Install it with: pip install wandb"
            )

        self.interval = interval
        self.max_samples_per_epoch = max_samples_per_epoch
        self.log_train = log_train
        self.log_val = log_val
        self.log_test = log_test
        self.score_threshold = score_threshold
        self.max_points = max_points

        self.sample_count = 0

    def before_val_epoch(self, runner: Runner) -> None:
        """Reset sample count at the start of validation epoch."""
        self.sample_count = 0

    def before_test_epoch(self, runner: Runner) -> None:
        """Reset sample count at the start of test epoch."""
        self.sample_count = 0

    def after_train_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: Optional[dict] = None,
        outputs: Optional[dict] = None,
    ) -> None:
        """Log training predictions."""
        if not self.log_train:
            return

        if self.every_n_train_iters(runner, self.interval):
            if self.sample_count < self.max_samples_per_epoch:
                self._log_predictions(runner, data_batch, outputs, "train")
                self.sample_count += 1

    def after_val_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: Optional[dict] = None,
        outputs: Optional[Sequence] = None,
    ) -> None:
        """Log validation predictions."""
        if not self.log_val:
            return

        if self.every_n_inner_iters(batch_idx, self.interval):
            if self.sample_count < self.max_samples_per_epoch:
                self._log_predictions(runner, data_batch, outputs, "val")
                self.sample_count += 1

    def after_test_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: Optional[dict] = None,
        outputs: Optional[Sequence] = None,
    ) -> None:
        """Log test predictions."""
        if not self.log_test:
            return

        if self.every_n_inner_iters(batch_idx, self.interval):
            if self.sample_count < self.max_samples_per_epoch:
                self._log_predictions(runner, data_batch, outputs, "test")
                self.sample_count += 1

    def _log_predictions(
        self,
        runner: Runner,
        data_batch: dict,
        outputs: Sequence,
        phase: str,
    ) -> None:
        """
        Extract predictions and log to WandB.

        Args:
            runner: MMEngine runner
            data_batch: Input data batch
            outputs: Model predictions
            phase: One of 'train', 'val', or 'test'
        """
        if data_batch is None or outputs is None:
            return

        try:
            # Extract first sample from batch
            # MMDetection3D uses data_samples which contain predictions
            if isinstance(outputs, dict):
                # Training mode: outputs is a dict
                data_samples = outputs.get('data_samples', None)
                if data_samples is None:
                    return
            else:
                # Validation/Test mode: outputs is a sequence of data samples
                data_samples = outputs

            # Get first sample
            if isinstance(data_samples, (list, tuple)):
                if len(data_samples) == 0:
                    return
                data_sample = data_samples[0]
            else:
                data_sample = data_samples

            # Extract point cloud from inputs
            inputs = data_batch.get('inputs', None)
            if inputs is None:
                return

            # Handle different input formats
            if isinstance(inputs, dict):
                points = inputs.get('points', None)
            elif isinstance(inputs, (list, tuple)):
                points = inputs[0] if len(inputs) > 0 else None
            elif torch.is_tensor(inputs):
                points = inputs
            else:
                points = None

            if points is None:
                return

            # Convert points to numpy
            if torch.is_tensor(points):
                points_np = points.cpu().numpy()
            else:
                points_np = np.array(points)

            # Handle LiDAR point format: [N, 4] or [N, 5] (x, y, z, intensity, ...)
            if points_np.ndim > 2:
                # Batch format, take first sample
                points_np = points_np[0]

            # Subsample points if too many
            if len(points_np) > self.max_points:
                indices = np.random.choice(len(points_np), self.max_points, replace=False)
                points_np = points_np[indices]

            # Extract XYZ coordinates (first 3 columns)
            if points_np.shape[1] >= 3:
                xyz = points_np[:, :3]

                # Add intensity as color if available
                if points_np.shape[1] >= 4:
                    intensity = points_np[:, 3:4]
                    # Normalize intensity to 0-255 range for RGB
                    intensity_norm = ((intensity - intensity.min()) /
                                     (intensity.max() - intensity.min() + 1e-6) * 255)
                    # Create grayscale RGB from intensity
                    colors = np.repeat(intensity_norm, 3, axis=1).astype(np.uint8)
                    point_cloud = np.concatenate([xyz, colors], axis=1)  # [N, 6]
                else:
                    point_cloud = xyz  # [N, 3]
            else:
                return

            # Extract predictions
            boxes_3d = self._extract_boxes(data_sample)

            # Extract ground truth boxes if available
            gt_boxes_3d = self._extract_gt_boxes(data_sample)

            # Create WandB Object3D
            wandb_data = {}

            # Log predictions
            if boxes_3d is not None and len(boxes_3d) > 0:
                pred_obj3d = wandb.Object3D({
                    "type": "lidar/beta",
                    "points": point_cloud,
                    "boxes": boxes_3d,
                })
                wandb_data[f"{phase}/predictions_3d"] = pred_obj3d

            # Log ground truth
            if gt_boxes_3d is not None and len(gt_boxes_3d) > 0:
                gt_obj3d = wandb.Object3D({
                    "type": "lidar/beta",
                    "points": point_cloud,
                    "boxes": gt_boxes_3d,
                })
                wandb_data[f"{phase}/ground_truth_3d"] = gt_obj3d

            # Log to WandB
            if wandb_data:
                wandb.log(wandb_data, step=runner.iter)

        except Exception as e:
            runner.logger.warning(f"Failed to log 3D predictions to WandB: {e}")

    def _extract_boxes(self, data_sample) -> Optional[list]:
        """
        Extract predicted 3D bounding boxes from data sample.

        Returns:
            List of box dictionaries for WandB Object3D format
        """
        try:
            # Access prediction results
            if not hasattr(data_sample, 'pred_instances_3d'):
                return None

            pred_instances = data_sample.pred_instances_3d

            # Extract bounding boxes
            if not hasattr(pred_instances, 'bboxes_3d'):
                return None

            bboxes_3d = pred_instances.bboxes_3d

            # Extract scores
            scores = pred_instances.scores_3d.cpu().numpy() if hasattr(
                pred_instances, 'scores_3d') else None

            # Extract labels
            labels = pred_instances.labels_3d.cpu().numpy() if hasattr(
                pred_instances, 'labels_3d') else None

            # Convert to numpy if tensor
            if torch.is_tensor(bboxes_3d):
                bboxes_np = bboxes_3d.cpu().numpy()
            else:
                # Assume it's a BaseInstance3DBoxes object
                bboxes_np = bboxes_3d.tensor.cpu().numpy()

            boxes_list = []
            for i, bbox in enumerate(bboxes_np):
                # Filter by score threshold
                if scores is not None and scores[i] < self.score_threshold:
                    continue

                # Convert box to corners format
                # bbox format: [x, y, z, dx, dy, dz, yaw] for LiDARInstance3DBoxes
                corners = self._box_to_corners(bbox)

                box_dict = {
                    "corners": corners.tolist(),
                    "label": f"pred_{int(labels[i])}" if labels is not None else "pred",
                    "color": [0, 255, 0],  # Green for predictions
                }

                if scores is not None:
                    box_dict["score"] = float(scores[i])

                boxes_list.append(box_dict)

            return boxes_list if boxes_list else None

        except Exception as e:
            return None

    def _extract_gt_boxes(self, data_sample) -> Optional[list]:
        """
        Extract ground truth 3D bounding boxes from data sample.

        Returns:
            List of box dictionaries for WandB Object3D format
        """
        try:
            # Access ground truth
            if not hasattr(data_sample, 'gt_instances_3d'):
                return None

            gt_instances = data_sample.gt_instances_3d

            # Extract bounding boxes
            if not hasattr(gt_instances, 'bboxes_3d'):
                return None

            bboxes_3d = gt_instances.bboxes_3d

            # Extract labels
            labels = gt_instances.labels_3d.cpu().numpy() if hasattr(
                gt_instances, 'labels_3d') else None

            # Convert to numpy if tensor
            if torch.is_tensor(bboxes_3d):
                bboxes_np = bboxes_3d.cpu().numpy()
            else:
                # Assume it's a BaseInstance3DBoxes object
                bboxes_np = bboxes_3d.tensor.cpu().numpy()

            boxes_list = []
            for i, bbox in enumerate(bboxes_np):
                # Convert box to corners format
                corners = self._box_to_corners(bbox)

                box_dict = {
                    "corners": corners.tolist(),
                    "label": f"gt_{int(labels[i])}" if labels is not None else "gt",
                    "color": [255, 0, 0],  # Red for ground truth
                }

                boxes_list.append(box_dict)

            return boxes_list if boxes_list else None

        except Exception as e:
            return None

    def _box_to_corners(self, bbox: np.ndarray) -> np.ndarray:
        """
        Convert 3D bounding box to 8 corner points.

        Args:
            bbox: [x, y, z, dx, dy, dz, yaw] or [x, y, z, dx, dy, dz, yaw, vx, vy]

        Returns:
            corners: [8, 3] array of corner coordinates
        """
        # Extract box parameters
        x, y, z = bbox[0], bbox[1], bbox[2]
        dx, dy, dz = bbox[3], bbox[4], bbox[5]
        yaw = bbox[6] if len(bbox) > 6 else 0.0

        # Half dimensions
        dx_half, dy_half, dz_half = dx / 2, dy / 2, dz / 2

        # Create 8 corners in local coordinates (before rotation)
        corners_local = np.array([
            [-dx_half, -dy_half, -dz_half],
            [dx_half, -dy_half, -dz_half],
            [dx_half, dy_half, -dz_half],
            [-dx_half, dy_half, -dz_half],
            [-dx_half, -dy_half, dz_half],
            [dx_half, -dy_half, dz_half],
            [dx_half, dy_half, dz_half],
            [-dx_half, dy_half, dz_half],
        ])

        # Rotation matrix around z-axis
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rot_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])

        # Rotate and translate
        corners = corners_local @ rot_matrix.T
        corners[:, 0] += x
        corners[:, 1] += y
        corners[:, 2] += z

        return corners
