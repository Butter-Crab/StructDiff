# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

import sam2


if os.path.isdir(os.path.join(sam2.__path__[0], "sam2")):
    raise RuntimeError(
        "You're likely running Python from the parent directory of the sam2 repository. "
        "Please run StructDiff from its own repository root instead."
    )


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=None,
    apply_postprocessing=True,
    **kwargs,
):
    hydra_overrides_extra = [] if hydra_overrides_extra is None else list(hydra_overrides_extra)

    if apply_postprocessing:
        hydra_overrides_extra += [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]

    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is None:
        raise ValueError("A local SAM2 checkpoint path is required.")

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
    missing_keys, unexpected_keys = model.load_state_dict(sd)
    if missing_keys:
        logging.error(missing_keys)
        raise RuntimeError("SAM2 checkpoint load failed because keys were missing.")
    if unexpected_keys:
        logging.error(unexpected_keys)
        raise RuntimeError("SAM2 checkpoint load failed because unexpected keys were found.")
    logging.info("Loaded SAM2 checkpoint successfully.")
