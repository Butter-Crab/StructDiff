# StructDiff

<p align="center">
  <strong>StructDiff: A Structure-Preserving and Spatially Controllable Diffusion Model for Single-Image Generation</strong>
</p>

<p align="center">
  Yinxi He<sup>1</sup>, Kang Liao<sup>2</sup>, Chunyu Lin<sup>1</sup>, Tianyi Wei<sup>2</sup>, Yao Zhao<sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>Beijing Jiaotong University &nbsp;&nbsp; <sup>2</sup>Nanyang Technological University
</p>

<p align="center">
  Accepted by <strong>IEEE Transactions on Multimedia 2026</strong>
</p>

<p align="center">
  <a href="https://butter-crab.github.io/StructDiff/">Project Page</a> ·
  <a href="https://arxiv.org/pdf/2604.12575">Paper PDF</a> ·
  <a href="https://arxiv.org/abs/2604.12575">arXiv</a> ·
  <a href="https://github.com/Butter-Crab/StructDiff">GitHub</a> ·
  <a href="https://huggingface.co/datasets/ButterCrab0122/Mulmini-NL">Dataset</a>
</p>

![StructDiff teaser](assets/teaser.png)

StructDiff is a single-image diffusion framework for structure-preserving generation and editing. This release focuses on a clean, image-only pipeline with:

- diverse generation by default
- optional 3D positional encoding for spatial control
- text-guided, reference-guided, and outpainting sampling modes
- an interactive SAM2-based mask tool for foreground mask creation

## Installation

### Core environment

Use a fresh Python 3.9 environment for training and sampling:

```bash
pip install -r requirements.txt
```

### Mask tool environment

The SAM2 mask editor uses a separate environment. The full setup and launch instructions are documented in [docs/mask_tool/README.md](docs/mask_tool/README.md).

## Quick Start

Train the default model without positional encoding:

```bash
python main.py --image_name face_3.jpg --run_name structdiff_default
```

Sample diverse results from the default model:

```bash
python sample.py --image_name face_3.jpg --run_name structdiff_default --sample_mode diverse
```

Train with positional encoding enabled:

```bash
python main.py --image_name face_3.jpg --run_name structdiff_pe --use_positional_encoding
```

Run spatially controlled sampling with a positional-encoding checkpoint:

```bash
python sample.py --image_name face_3.jpg --run_name structdiff_pe --use_positional_encoding --sample_mode control --control_action shift
```

For detailed data layout, mask editing, and additional sampling modes, see the guides below.

## Documentation

- [Data Guide](docs/data/README.md)
- [Mask Tool Guide](docs/mask_tool/README.md)

## Citation

```bibtex
@article{he2026structdiff,
  title={StructDiff: A Structure-Preserving and Spatially Controllable Diffusion Model for Single-Image Generation},
  author={He, Yinxi and Liao, Kang and Lin, Chunyu and Wei, Tianyi and Zhao, Yao},
  journal={IEEE Transactions on Multimedia},
  year={2026}
}
```

## Acknowledgements

This project builds on or is inspired by several open-source efforts, including [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch), [CLIP](https://github.com/openai/CLIP), [Text2LIVE](https://github.com/omerbt/Text2LIVE), [SAM2](https://github.com/facebookresearch/sam2), and [GeoDiffuser](https://github.com/RahulSajnani/GeoDiffuser).
