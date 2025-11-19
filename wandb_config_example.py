"""
Example configuration file for MMDetection3D with WandB 3D Detection Hook

This shows how to integrate the custom WandB hook into your training config.
"""

# =============================================================================
# Base Configuration
# =============================================================================
_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_kitti.py',  # Your model config
    '../_base_/datasets/kitti-3d-3class.py',            # Your dataset config
    '../_base_/schedules/cyclic-40e.py',                # Your schedule config
    '../_base_/default_runtime.py'                       # Runtime config
]

# =============================================================================
# WandB Visualization Backend Configuration
# =============================================================================
vis_backends = [
    dict(
        type='LocalVisBackend'  # Keep local visualization
    ),
    dict(
        type='TensorboardVisBackend'  # Optional: TensorBoard
    ),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='mmdet3d-lidar-detection',
            name='pointpillars-kitti-experiment',
            entity='your-wandb-username',  # Replace with your WandB username/team
            config={
                'model': 'PointPillars',
                'dataset': 'KITTI',
                'batch_size': 6,
            },
            tags=['lidar', '3d-detection', 'pointpillars'],
        ),
    ),
]

visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# =============================================================================
# Custom Hook Configuration
# =============================================================================
custom_hooks = [
    # WandB 3D Detection Visualization Hook
    dict(
        type='WandB3DDetectionHook',
        interval=50,                    # Log every 50 iterations
        max_samples_per_epoch=10,       # Log max 10 samples per epoch
        log_train=False,                # Don't log training (can be slow)
        log_val=True,                   # Log validation predictions
        log_test=True,                  # Log test predictions
        score_threshold=0.3,            # Only log boxes with score > 0.3
        max_points=200000,              # Max points per cloud (WandB limit: 300k)
    ),
]

# =============================================================================
# Training Configuration
# =============================================================================
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=80,
    val_interval=2  # Validate every 2 epochs
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# =============================================================================
# Other Settings
# =============================================================================
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=5,
        save_best='auto',
    ),
    logger=dict(
        type='LoggerHook',
        interval=50  # Log metrics every 50 iterations
    ),
)

# Load checkpoint
load_from = None  # Or path to pretrained model
resume = False

# =============================================================================
# Alternative: Minimal Configuration (if you use _base_ configs)
# =============================================================================
"""
If you want a minimal config that extends an existing one:

_base_ = ['./pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py']

# Add WandB backend to existing visualizer
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='mmdet3d-lidar',
            name='my-experiment',
        )
    )
]

visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
)

# Add custom hook
custom_hooks = [
    dict(
        type='WandB3DDetectionHook',
        interval=50,
        max_samples_per_epoch=10,
    )
]
"""
