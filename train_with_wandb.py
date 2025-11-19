"""
Example training script for MMDetection3D with WandB 3D Detection Hook

Usage:
    python train_with_wandb.py configs/pointpillars/my_config.py
    python train_with_wandb.py configs/pointpillars/my_config.py --work-dir ./work_dirs/exp1
    python train_with_wandb.py configs/pointpillars/my_config.py --resume
"""

import argparse
import os
import os.path as osp
import sys

# Add current directory to path to import custom hook
sys.path.insert(0, osp.dirname(__file__))

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

# Import and register custom hook
from mmdet3d.registry import HOOKS
from wandb_3d_hook import WandB3DDetectionHook

# Register the custom hook
HOOKS.register_module(module=WandB3DDetectionHook, force=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a 3D detector with WandB logging')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training'
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.'
    )
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher'
    )
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)

    # Merge cfg-options
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Set work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # Default work_dir
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # Enable automatic mixed precision training
    if args.amp:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # Resume training
    cfg.resume = args.resume

    # Ensure WandB hook is in custom_hooks
    if not hasattr(cfg, 'custom_hooks'):
        cfg.custom_hooks = []

    # Check if WandB hook already exists in config
    has_wandb_hook = any(
        hook.get('type') == 'WandB3DDetectionHook'
        for hook in cfg.custom_hooks
    )

    # Add default WandB hook if not present
    if not has_wandb_hook:
        print("Adding default WandB3DDetectionHook to custom_hooks")
        cfg.custom_hooks.append(
            dict(
                type='WandB3DDetectionHook',
                interval=50,
                max_samples_per_epoch=10,
                log_train=False,
                log_val=True,
                log_test=True,
                score_threshold=0.3,
                max_points=200000,
            )
        )

    # Ensure WandB backend is configured
    if not hasattr(cfg, 'visualizer'):
        cfg.visualizer = dict(
            type='Det3DLocalVisualizer',
            vis_backends=[dict(type='LocalVisBackend')],
        )

    if not hasattr(cfg.visualizer, 'vis_backends'):
        cfg.visualizer.vis_backends = [dict(type='LocalVisBackend')]

    # Check if WandB backend exists
    has_wandb_backend = any(
        backend.get('type') == 'WandbVisBackend'
        for backend in cfg.visualizer.vis_backends
    )

    if not has_wandb_backend:
        print("Adding WandbVisBackend to visualizer")
        # Extract project name from config path
        project_name = osp.splitext(osp.basename(args.config))[0]
        cfg.visualizer.vis_backends.append(
            dict(
                type='WandbVisBackend',
                init_kwargs=dict(
                    project='mmdet3d-lidar',
                    name=project_name,
                )
            )
        )

    # Build the runner
    runner = Runner.from_cfg(cfg)

    # Start training
    print(f"Starting training with work_dir: {cfg.work_dir}")
    print(f"WandB logging enabled: Check https://wandb.ai for results")
    runner.train()


if __name__ == '__main__':
    main()
