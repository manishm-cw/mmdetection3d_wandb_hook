# MMDetection3D + WandB 3D Detection Logging

Complete guide for logging LiDAR-based 3D detections with masks to Weights & Biases.

## Overview

This integration provides:
- ✅ 3D point cloud visualization with bounding boxes
- ✅ Predicted vs ground truth comparison
- ✅ Score-based filtering
- ✅ Automatic logging during training/validation/testing
- ✅ Support for intensity-colored point clouds

## Prerequisites

```bash
# Install required packages
pip install mmdet3d
pip install mmcv-full
pip install mmengine
pip install wandb

# Login to WandB
wandb login
```

## Installation

### Step 1: Copy the Custom Hook

Place `wandb_3d_hook.py` in your MMDetection3D project:

```bash
# Option A: In your project root
your_project/
├── configs/
├── wandb_3d_hook.py  # <-- Place here
└── tools/

# Option B: In mmdet3d custom hooks directory
mmdet3d/
├── engine/
│   └── hooks/
│       └── wandb_3d_hook.py  # <-- Or here
```

### Step 2: Register the Hook

If placing in your project root, register it in your training script:

**Method 1: In your training script (e.g., `tools/train.py`)**

```python
# Add before cfg = Config.fromfile(args.config)
from mmdet3d.registry import HOOKS
import sys
sys.path.insert(0, '.')  # Add project root to path
from wandb_3d_hook import WandB3DDetectionHook

# Register the hook
HOOKS.register_module(module=WandB3DDetectionHook, force=True)
```

**Method 2: In a custom `__init__.py`**

Create `custom_hooks/__init__.py`:

```python
from .wandb_3d_hook import WandB3DDetectionHook
__all__ = ['WandB3DDetectionHook']
```

Then in your training script:
```python
from mmdet3d.registry import HOOKS
from custom_hooks import WandB3DDetectionHook

HOOKS.register_module(module=WandB3DDetectionHook, force=True)
```

### Step 3: Configure Your Training

Update your config file (e.g., `configs/pointpillars/my_config.py`):

```python
# Base configurations
_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic-40e.py',
    '../_base_/default_runtime.py'
]

# Add WandB visualization backend
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='mmdet3d-lidar-detection',
            name='experiment-name',
            entity='your-wandb-username',
            tags=['lidar', '3d-detection'],
        )
    )
]

visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
)

# Add custom WandB 3D detection hook
custom_hooks = [
    dict(
        type='WandB3DDetectionHook',
        interval=50,                # Log every 50 iterations
        max_samples_per_epoch=10,   # Log up to 10 samples per epoch
        log_train=False,            # Skip training (faster)
        log_val=True,               # Log validation
        log_test=True,              # Log testing
        score_threshold=0.3,        # Only show boxes with confidence > 0.3
        max_points=200000,          # Subsample to 200k points
    )
]
```

## Usage

### Basic Training

```bash
# Standard training
python tools/train.py configs/pointpillars/my_config.py

# With WandB offline mode
WANDB_MODE=offline python tools/train.py configs/pointpillars/my_config.py

# Resume training
python tools/train.py configs/pointpillars/my_config.py --resume
```

### Testing with Logging

```bash
python tools/test.py \
    configs/pointpillars/my_config.py \
    checkpoints/latest.pth \
    --cfg-options "custom_hooks=[dict(type='WandB3DDetectionHook', log_test=True)]"
```

### Fine-tuning a Pretrained Model

```python
# In your config
load_from = 'checkpoints/pointpillars_pretrained.pth'

# Or PointNet, VoxelNet, etc.
load_from = 'checkpoints/your_pretrained_model.pth'
```

## Configuration Options

### Hook Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `interval` | int | 50 | Log every N iterations |
| `max_samples_per_epoch` | int | 10 | Max samples to log per epoch |
| `log_train` | bool | False | Log during training |
| `log_val` | bool | True | Log during validation |
| `log_test` | bool | True | Log during testing |
| `score_threshold` | float | 0.3 | Min confidence for boxes |
| `max_points` | int | 200000 | Max points per cloud |

### WandB Backend Options

```python
dict(
    type='WandbVisBackend',
    init_kwargs=dict(
        project='project-name',           # WandB project
        entity='username-or-team',        # WandB account
        name='experiment-name',           # Run name
        tags=['tag1', 'tag2'],           # Tags for organization
        group='experiment-group',         # Group related runs
        notes='Experiment description',   # Run notes
        config={                          # Log hyperparameters
            'model': 'PointPillars',
            'dataset': 'KITTI',
            'lr': 0.001,
        },
        resume='allow',                   # Resume mode
    )
)
```

## Example Configs for Popular Models

### PointPillars on KITTI

```python
_base_ = './pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py'

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(
        project='mmdet3d-pointpillars',
        name='kitti-baseline',
    ))
]

custom_hooks = [
    dict(type='WandB3DDetectionHook', interval=100)
]
```

### VoxelNet on nuScenes

```python
_base_ = './voxelnet_8xb1-80e_kitti-3d-3class.py'

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(
        project='mmdet3d-voxelnet',
        name='nuscenes-experiment',
    ))
]

custom_hooks = [
    dict(
        type='WandB3DDetectionHook',
        interval=50,
        score_threshold=0.5,  # Higher threshold for nuScenes
    )
]
```

### PointNet++ Fine-tuning

```python
_base_ = './pointnet2_msg_2x_kitti-3d-3class.py'

# Load pretrained weights
load_from = 'checkpoints/pointnet2_pretrained.pth'

vis_backends = [
    dict(type='WandbVisBackend', init_kwargs=dict(
        project='mmdet3d-pointnet-finetuning',
        tags=['finetune', 'pointnet++'],
    ))
]

custom_hooks = [
    dict(
        type='WandB3DDetectionHook',
        interval=25,
        log_train=True,  # Log training for fine-tuning analysis
    )
]
```

## Viewing Results in WandB

After training, your WandB dashboard will show:

1. **Point Cloud Visualizations**
   - Navigate to the "Media" tab
   - Look for `val/predictions_3d` and `val/ground_truth_3d`
   - Interactive 3D viewer with:
     - Green boxes = predictions
     - Red boxes = ground truth
     - Score-based filtering

2. **Metrics**
   - Training/validation loss curves
   - mAP, NDS, and other detection metrics
   - Learning rate schedules

3. **System Metrics**
   - GPU utilization
   - Memory usage
   - Training speed

## Troubleshooting

### Issue: Hook not found

```
KeyError: 'WandB3DDetectionHook is not in the HOOKS registry'
```

**Solution:** Make sure to register the hook before creating the runner:

```python
from mmdet3d.registry import HOOKS
from wandb_3d_hook import WandB3DDetectionHook
HOOKS.register_module(module=WandB3DDetectionHook, force=True)
```

### Issue: No point clouds logged

**Possible causes:**
1. `interval` is too large (increase frequency)
2. `max_samples_per_epoch` is too small
3. Data format incompatibility

**Debug:**
```python
# Add to hook's _log_predictions method
print(f"Data batch keys: {data_batch.keys()}")
print(f"Outputs type: {type(outputs)}")
```

### Issue: WandB 300k point limit

If you see warnings about point limits:

```python
# Reduce max_points
custom_hooks = [
    dict(
        type='WandB3DDetectionHook',
        max_points=100000,  # Reduce from default 200k
    )
]
```

### Issue: Out of memory

Logging 3D visualizations uses extra memory:

```python
# Solutions:
# 1. Reduce logging frequency
dict(type='WandB3DDetectionHook', interval=200)

# 2. Disable training logging
dict(type='WandB3DDetectionHook', log_train=False)

# 3. Reduce samples per epoch
dict(type='WandB3DDetectionHook', max_samples_per_epoch=5)
```

## Advanced: Custom Data Format

If your dataset uses a different format, modify the `_log_predictions` method:

```python
def _log_predictions(self, runner, data_batch, outputs, phase):
    # Your custom data extraction logic
    points = self._extract_custom_points(data_batch)
    boxes = self._extract_custom_boxes(outputs)

    # Convert to WandB format
    wandb_obj = wandb.Object3D({
        "type": "lidar/beta",
        "points": points,  # [N, 3] or [N, 6]
        "boxes": boxes,    # List of box dicts
    })

    wandb.log({f"{phase}/predictions": wandb_obj})
```

## Performance Tips

1. **Log only validation**: Set `log_train=False` to save time
2. **Increase interval**: Use `interval=100` or higher for large datasets
3. **Reduce samples**: Set `max_samples_per_epoch=5` for quick checks
4. **Async logging**: WandB logs asynchronously, but consider `wandb.finish()` at end

## Dataset-Specific Notes

### KITTI
- Works out of the box
- Point format: [N, 4] (x, y, z, intensity)
- Classes: Car, Pedestrian, Cyclist

### nuScenes
- May need adjustment for 10 classes
- Point format varies by sensor
- Consider higher `score_threshold`

### Waymo
- Large point clouds - use aggressive subsampling
- Set `max_points=50000` for faster logging

### Custom Datasets
- Ensure data_batch contains 'inputs' with 'points'
- Verify outputs contain 'pred_instances_3d'

## References

- [MMDetection3D Documentation](https://mmdetection3d.readthedocs.io/)
- [WandB 3D Visualization](https://docs.wandb.ai/guides/track/log/media#3d-visualizations)
- [MMEngine Hooks](https://mmengine.readthedocs.io/en/latest/design/hook.html)

## Support

For issues:
1. Check MMDetection3D version: `python -c "import mmdet3d; print(mmdet3d.__version__)"`
2. Check WandB version: `wandb --version`
3. Enable debug logging: Add `print()` statements in hook methods
4. Open issue with: config file, error message, MMDet3D version

## License

This hook follows MMDetection3D's Apache 2.0 license.
