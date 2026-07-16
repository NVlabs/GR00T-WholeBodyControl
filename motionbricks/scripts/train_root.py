"""Root model training script using synthetic data.

Demonstrates how the root backbone training pipeline works without
requiring the actual motion dataset. Loads the saved model config from
the checkpoint directory and trains on randomly generated motion tensors.

The root model does not require a pretrained VQVAE — it directly
predicts continuous root motion values.

Usage:
    python scripts/train_root.py --max_steps 100
"""

import argparse
import copy
import os

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from motionbricks.data.synthetic_dataset import SyntheticMotionDataset, collate_batch
from motionbricks.helper.pl_util import load_motion_rep


def load_config(result_dir: str, max_steps: int):
    """Load and patch hparams.yaml for single-GPU training."""
    version_dir = os.path.join(result_dir, "motionbricks_root", "version_1")
    hparams_path = os.path.join(version_dir, "hparams.yaml")
    conf = OmegaConf.load(hparams_path)

    with open_dict(conf):
        # resolve data paths to the version directory (where skeleton/stats live)
        conf.data = {"folder": version_dir, "text_embeddings": None}
        conf.skeleton.folder = os.path.join(version_dir, "skeleton")
        conf.motion_rep.stats.folder = os.path.join(version_dir, "stats", "motion")

        # single-GPU training overrides
        conf.trainer.devices = 1
        conf.trainer.num_nodes = 1
        conf.trainer.max_steps = max_steps
        conf.trainer.accelerator = "auto"
        conf.trainer.strategy = "auto"
        conf.trainer.enable_progress_bar = True
        conf.trainer.log_every_n_steps = 10
        conf.trainer.val_check_interval = max_steps
        conf.trainer.num_sanity_val_steps = 0

        # resolve ${trainer.max_steps} in scheduler
        conf.model.scheduler.num_training_steps = max_steps

        # remove keys with unresolvable ${hydra:...} interpolations
        conf.id = "synthetic"
        conf.run_dir = "."
        conf.out_dir = result_dir

    # resolve all ${} interpolations, then re-wrap as DictConfig
    resolved = OmegaConf.to_container(conf, resolve=True)
    conf = OmegaConf.create(resolved)

    return conf, version_dir


def main():
    parser = argparse.ArgumentParser(description="Root model training")
    parser.add_argument("--result_dir", type=str, default="./out",
                        help="Directory containing pretrained checkpoints")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of synthetic samples in dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    conf, version_dir = load_config(args.result_dir, args.max_steps)

    # instantiate skeleton and motion representation
    motion_rep = load_motion_rep(conf)
    feat_dim = len(motion_rep.indices['all'])

    # create synthetic dataset
    dataset = SyntheticMotionDataset(
        feat_dim=feat_dim,
        num_samples=args.num_samples,
        min_frames=200,
        max_frames=400,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_batch,
        persistent_workers=True,
    )

    # instantiate networks and model
    model_conf = copy.deepcopy(conf.model)
    with open_dict(model_conf):
        # instantiate backbone network (needs full motion_rep for dual_rep access)
        backbone_net = instantiate(
            model_conf.backbone_network,
            motion_rep=motion_rep,
            _recursive_=False,
        )

        # build optimizer and scheduler as partials
        optimizer_fn = instantiate(model_conf.optimizer)
        scheduler_fn = instantiate(model_conf.scheduler) if model_conf.scheduler else None

        model = instantiate(
            model_conf,
            pose_vqvae_network=None,
            root_vqvae_network=None,
            backbone_network=backbone_net,
            motion_rep=motion_rep,
            optimizer=optimizer_fn,
            scheduler=scheduler_fn,
            _recursive_=False,
        )

    # create trainer (no callbacks)
    trainer = pl.Trainer(
        max_steps=conf.trainer.max_steps,
        devices=conf.trainer.devices,
        num_nodes=conf.trainer.num_nodes,
        accelerator=conf.trainer.accelerator,
        strategy=conf.trainer.strategy,
        precision=conf.trainer.precision,
        gradient_clip_val=conf.trainer.gradient_clip_val,
        enable_progress_bar=conf.trainer.enable_progress_bar,
        log_every_n_steps=conf.trainer.log_every_n_steps,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        logger=False,
    )

    print(f"Starting root model training for {args.max_steps} steps...")
    print(f"  Feature dim: {feat_dim}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Dataset size: {args.num_samples}")
    trainer.fit(model, train_dataloaders=dataloader)
    print("Training complete.")


if __name__ == "__main__":
    main()
