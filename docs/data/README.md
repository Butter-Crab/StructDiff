## StructDiff Data Layout

- `./data/images/`: all repo images used by training, sampling, and the mask tool
- `./data/masks/default/`: default masks named `<image_stem>_mask.png`
- `./data/masks/variants/<image_stem>/`: optional edited masks such as `_mask_change*` or `whole_mask` variants

### Naming

- Image file: keep the original image name, for example `blackswan.jpg`
- Default mask: `<image_stem>_mask.png`
- Variant mask: keep the original descriptive suffix, for example `face_3_whole_mask.png` or `tennis_mask_change1.png`

### Loader Behavior

- Training and sampling search `./data/images/`
- Default masks are loaded only from `./data/masks/default/`
- If the default mask is missing, StructDiff prints a notice and treats the whole image as foreground
- By default, training and sampling resize the loaded image and mask so the longest side is at most `256`. Use `--resize_max_side 0` to disable that behavior
