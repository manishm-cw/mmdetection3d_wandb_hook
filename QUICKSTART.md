# Quick Start: MMDetection3D + WandB 3D Logging

Get up and running in 5 minutes!

## Step 1: Install Dependencies (1 min)

```bash
pip install mmdet3d mmcv-full mmengine wandb
wandb login
```

## Step 2: Copy Files (30 seconds)

Copy these files to your MMDetection3D project:
```
your_project/
├── wandb_3d_hook.py          # The custom hook
├── train_with_wandb.py       # Training script
└── configs/
    └── your_config.py
```

## Step 3: Update Your Config (2 min)

Add to your config file (e.g., `configs/pointpillars/my_config.py`):

```python
# Add WandB backend
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='my-3d-detection-project',
            name='experiment-1',
        )
    )
]

visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
)

# Add custom hook for 3D logging
custom_hooks = [
    dict(
        type='WandB3DDetectionHook',
        interval=50,
        max_samples_per_epoch=10,
    )
]
```

## Step 4: Train! (30 seconds)

```bash
python train_with_wandb.py configs/pointpillars/my_config.py
```

Or use the standard training script:

```bash
python tools/train.py configs/pointpillars/my_config.py
```

*(Make sure to register the hook in tools/train.py first - see README)*

## Step 5: View Results

Open your browser and go to: https://wandb.ai/your-username/my-3d-detection-project

You'll see:
- 📊 Training/validation metrics
- 🎨 3D point cloud visualizations with bounding boxes
- 🔄 Ground truth vs predictions comparison
- 📈 Real-time training progress

## Common Use Cases

### Fine-tune a Pretrained Model

```python
# In your config
load_from = 'checkpoints/pointpillars_pretrained.pth'
```

### Train on Custom LiDAR Dataset

```python
# Point your config to custom data
data_root = 'data/my_lidar_dataset/'
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='train.pkl',
    )
)
```

### Adjust Logging Frequency

```python
custom_hooks = [
    dict(
        type='WandB3DDetectionHook',
        interval=100,              # Log less frequently
        max_samples_per_epoch=5,   # Log fewer samples
        score_threshold=0.5,       # Only high-confidence boxes
    )
]
```

## What Gets Logged?

### Automatically Logged
- Loss curves (every iteration)
- Learning rate schedule
- Validation metrics (mAP, NDS, etc.)
- Training time and speed

### 3D Visualizations (via custom hook)
- Point clouds with intensity colors
- Predicted 3D bounding boxes (green)
- Ground truth boxes (red)
- Confidence scores
- Class labels

## Troubleshooting

**Hook not found error?**
```python
# Add to train script before Runner
from mmdet3d.registry import HOOKS
from wandb_3d_hook import WandB3DDetectionHook
HOOKS.register_module(module=WandB3DDetectionHook, force=True)
```

**No 3D visualizations?**
- Check `interval` isn't too large
- Verify `log_val=True` in hook config
- Make sure validation is running (check `val_interval`)

**Out of memory?**
```python
# Reduce logging
custom_hooks = [
    dict(
        type='WandB3DDetectionHook',
        interval=200,
        max_samples_per_epoch=3,
        max_points=100000,  # Reduce point cloud size
    )
]
```

## Next Steps

- Read the full [README_WANDB_INTEGRATION.md](README_WANDB_INTEGRATION.md) for advanced usage
- Check [wandb_config_example.py](wandb_config_example.py) for more config options
- Experiment with different models: PointPillars, VoxelNet, PointNet++, SECOND

## Example Commands

```bash
# Basic training
python train_with_wandb.py configs/pointpillars/my_config.py

# Custom work directory
python train_with_wandb.py configs/pointpillars/my_config.py --work-dir work_dirs/exp1

# Resume training
python train_with_wandb.py configs/pointpillars/my_config.py --resume

# Override config options
python train_with_wandb.py configs/pointpillars/my_config.py \
    --cfg-options "custom_hooks=[dict(type='WandB3DDetectionHook', interval=100)]"

# Enable mixed precision training (faster)
python train_with_wandb.py configs/pointpillars/my_config.py --amp
```

## Popular Model Configs

### PointPillars (Fast, Good for Real-time)
```bash
python train_with_wandb.py configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py
```

### VoxelNet (Good Accuracy)
```bash
python train_with_wandb.py configs/voxelnet/voxelnet_8xb1-80e_kitti-3d-3class.py
```

### SECOND (Balanced Speed/Accuracy)
```bash
python train_with_wandb.py configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-3class.py
```

## Resources

- [MMDetection3D Docs](https://mmdetection3d.readthedocs.io/)
- [WandB Docs](https://docs.wandb.ai/)
- [Example Configs](wandb_config_example.py)
- [Full README](README_WANDB_INTEGRATION.md)

---

**Need help?** Check the full documentation or open an issue with your config and error message.
