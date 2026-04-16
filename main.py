import logging
import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from common_utils.image import load_structdiff_image_and_mask
from config import Config, log_config, parse_cmdline_args_to_config
from datasets.cropset import CropSet
from diffusion.diffusion import Diffusion
from models.arfnet import ARFNet
from models.positional_encoding import PositionalEncoding3D


def configure_runtime_logging():
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning").setLevel(logging.WARNING)


def train_image_diffusion(cfg):
    training_steps = 50_000

    image, foreground_mask, image_path, mask_path, used_full_foreground_mask = load_structdiff_image_and_mask(
        cfg.image_name,
        resize_max_side=cfg.resize_max_side,
    )

    head_grid = None
    if cfg.use_positional_encoding:
        head_grid = PositionalEncoding3D()(image, foreground_mask, fg_bg=True)

    crop_size = int(min(image[0].shape[-2:]) * 0.95)
    train_dataset = CropSet(
        mask_image=foreground_mask,
        image=image,
        head_grid=head_grid,
        crop_size=crop_size,
        use_flip=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)

    model = ARFNet(in_channels=3, filters_per_layer=cfg.network_filters, depth=cfg.network_depth)
    diffusion = Diffusion(
        model,
        training_target="x0",
        timesteps=cfg.diffusion_timesteps,
        auto_sample=True,
        sample_size=image[0].shape[-2:],
        head_grid=head_grid,
        uses_positional_encoding=cfg.use_positional_encoding,
    )

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            filename="single-level-{step}",
            save_last=True,
            save_top_k=3,
            monitor="train_loss",
            mode="min",
        )
    ]
    logger = pl.loggers.TensorBoardLogger("lightning_logs/", name=cfg.image_name, version=cfg.run_name)
    trainer_gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(
        max_steps=training_steps,
        gpus=trainer_gpus,
        auto_select_gpus=bool(trainer_gpus),
        logger=logger,
        log_every_n_steps=10,
        callbacks=callbacks,
        enable_model_summary=False,
    )

    print(f"Training image: {image_path}")
    if used_full_foreground_mask:
        print("Foreground mask: <not found in data/masks/default, using full-image foreground>")
    else:
        print(f"Foreground mask: {mask_path}")

    trainer.fit(diffusion, train_loader)


def main():
    cfg = parse_cmdline_args_to_config(Config())

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.available_gpus

    configure_runtime_logging()
    log_config(cfg, context="train")

    if cfg.task != "image":
        raise ValueError(f"Unknown task: {cfg.task}")

    train_image_diffusion(cfg)


if __name__ == "__main__":
    main()
