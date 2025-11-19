# MMDetection3D WandB Hook

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![MMDetection3D](https://img.shields.io/badge/MMDetection3D-1.0%2B-orange)](https://github.com/open-mmlab/mmdetection3d)
[![WandB](https://img.shields.io/badge/WandB-Integration-yellow)](https://wandb.ai/)

Custom hook for logging 3D LiDAR detections with point clouds, bounding boxes, and masks to Weights & Biases (WandB) during MMDetection3D training.

![3D Detection Visualization](https://img.shields.io/badge/Visualize-3D%20Point%20Clouds-green)

## ✨ Features

- 🎯 **3D Point Cloud Visualization** - Interactive 3D viewer in WandB
- 📦 **Bounding Box Logging** - Predicted (green) and ground truth (red) boxes
- 🎨 **Intensity Coloring** - LiDAR intensity mapped to point colors
- 🎚️ **Score Filtering** - Confidence-based box filtering
- 📊 **Training Metrics** - Automatic loss, mAP, and performance tracking
- 🔧 **Easy Integration** - Drop-in hook for any MMDetection3D model
- 🚀 **Production Ready** - Tested with PointPillars, VoxelNet, SECOND, PointNet++

## 🚀 Quick Start

### Installation

```bash
pip install mmdet3d mmcv-full mmengine wandb
wandb login
```

### Usage

**1. Copy the hook to your project:**
```bash
git clone https://github.com/manishm-cw/mmdetection3d_wandb_hook.git
cd mmdetection3d_wandb_hook
cp wandb_3d_hook.py /path/to/your/mmdet3d/project/
```

**2. Update your config file:**
```python
# Add WandB backend
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='my-3d-detection',
            name='experiment-1',
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
        interval=50,              # Log every 50 iterations
        max_samples_per_epoch=10, # Log up to 10 samples per epoch
        score_threshold=0.3,      # Min confidence for boxes
    )
]
```

**3. Train your model:**
```bash
python train_with_wandb.py configs/pointpillars/my_config.py
```

**4. View results at:** https://wandb.ai/your-username/my-3d-detection

## 📁 Repository Structure

```
mmdetection3d_wandb_hook/
├── wandb_3d_hook.py              # Main hook implementation
├── train_with_wandb.py           # Ready-to-use training script
├── wandb_config_example.py       # Example configuration
├── QUICKSTART.md                 # 5-minute quick start guide
├── README_WANDB_INTEGRATION.md   # Comprehensive documentation
└── README.md                     # This file
```

## 📖 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[README_WANDB_INTEGRATION.md](README_WANDB_INTEGRATION.md)** - Full documentation with troubleshooting
- **[wandb_config_example.py](wandb_config_example.py)** - Configuration examples

## 🎯 What Gets Logged?

### Automatic Metrics
- Training/validation loss curves
- mAP, NDS, and other detection metrics
- Learning rate schedules
- GPU utilization and training speed

### 3D Visualizations (via hook)
- Point clouds with intensity coloring
- 3D bounding boxes (8 corner points)
- Predicted boxes (green) with confidence scores
- Ground truth boxes (red)
- Class labels

## 🔧 Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `interval` | int | 50 | Log every N iterations |
| `max_samples_per_epoch` | int | 10 | Max samples per epoch |
| `log_train` | bool | False | Log training predictions |
| `log_val` | bool | True | Log validation predictions |
| `log_test` | bool | True | Log test predictions |
| `score_threshold` | float | 0.3 | Min confidence for boxes |
| `max_points` | int | 200000 | Max points per cloud |

## 📦 Supported Models

Works with all MMDetection3D models:
- ✅ PointPillars
- ✅ VoxelNet
- ✅ SECOND
- ✅ PointNet / PointNet++
- ✅ MVXNet
- ✅ FCOS3D
- ✅ And more...

## 📊 Supported Datasets

- ✅ KITTI
- ✅ nuScenes
- ✅ Waymo Open Dataset
- ✅ Lyft Level 5
- ✅ Custom LiDAR datasets

## 🎓 Example Use Cases

### Fine-tune a Pretrained Model
```python
load_from = 'checkpoints/pointpillars_pretrained.pth'
```

### Train on Custom Dataset
```python
data_root = 'data/my_lidar_data/'
```

### Adjust Logging Frequency
```python
custom_hooks = [
    dict(
        type='WandB3DDetectionHook',
        interval=100,
        max_samples_per_epoch=5,
    )
]
```

## 🐛 Troubleshooting

**Hook not found?**
```python
from mmdet3d.registry import HOOKS
from wandb_3d_hook import WandB3DDetectionHook
HOOKS.register_module(module=WandB3DDetectionHook, force=True)
```

**Out of memory?**
```python
custom_hooks = [
    dict(
        type='WandB3DDetectionHook',
        interval=200,              # Reduce frequency
        max_samples_per_epoch=3,   # Fewer samples
        max_points=100000,         # Smaller point clouds
    )
]
```

See [README_WANDB_INTEGRATION.md](README_WANDB_INTEGRATION.md) for more troubleshooting.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project follows MMDetection3D's Apache 2.0 license.

## 🙏 Acknowledgements

- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) - OpenMMLab's 3D detection toolbox
- [Weights & Biases](https://wandb.ai/) - Experiment tracking and visualization
- [MMEngine](https://github.com/open-mmlab/mmengine) - OpenMMLab's training foundation

## 📧 Contact

For issues or questions:
- Open an [issue](https://github.com/manishm-cw/mmdetection3d_wandb_hook/issues)
- Check the [documentation](README_WANDB_INTEGRATION.md)

## ⭐ Star History

If you find this useful, please star the repository!

---

**Made with ❤️ for the 3D detection community**
