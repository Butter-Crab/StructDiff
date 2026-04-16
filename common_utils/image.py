import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import ipywidgets as widgets
    from IPython.display import display

    __IMSHOW_ENABLED = True
except ImportError:
    __IMSHOW_ENABLED = False


__all__ = [
    'to_numpy',
    'np2pt',
    'pt2np',
    'imread',
    'imwrite',
    'imshow',
    'load_structdiff_image',
    'resolve_structdiff_image_path',
    'resolve_structdiff_mask_variant_path',
    'load_structdiff_image_and_mask',
    'load_structdiff_mask_variant',
]

_BOUNDS = (0.0, 1.0)

try:
    _PIL_BILINEAR = Image.Resampling.BILINEAR
    _PIL_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    _PIL_BILINEAR = Image.BILINEAR
    _PIL_NEAREST = Image.NEAREST


def to_numpy(tensor, clone=True):
    tensor = tensor.detach()
    tensor = tensor.clone() if clone else tensor
    return tensor.cpu().numpy()


def np2pt(arr):
    # return torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).contiguous()



def pt2np(tensor):
    return to_numpy(tensor.squeeze(0).permute(1, 2, 0))


def _img_to_float32(image, bounds):
    vmin, vmax = bounds
    image = np.asarray(image, dtype=np.float32) / 255.0
    image = np.clip((vmax - vmin) * image + vmin, vmin, vmax)
    return image


def _torch_to_np(image):
    image = to_numpy(image)
    if image.ndim == 3:
        image = image.transpose((1, 2, 0))
    elif image.ndim == 4:
        image = image.transpose((0, 2, 3, 1))
    else:
        raise ValueError()
    return image


def _img_to_uint8(image, bounds):
    if isinstance(image, torch.Tensor):
        image = _torch_to_np(image)

    if image.dtype != np.uint8:
        vmin, vmax = bounds
        image = (image.astype(np.float32) - vmin) / (vmax - vmin)
        image = (image * 255.0).round().clip(0, 255).astype(np.uint8)
    return image


def _check_path(path):
    path = os.path.abspath(path)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    return path


## Image
def imread(fname, bounds=_BOUNDS, mode='RGB', pt=True, **kwargs):
    # image = image_to_array(load_img(fname, **kwargs))
    image = Image.open(fname, **kwargs).convert(mode=mode)
    image = _img_to_float32(image, bounds)
    if pt:
        image = np2pt(image)
    return image


def _structdiff_data_root(data_root=None):
    return Path(data_root) if data_root is not None else Path(__file__).resolve().parents[1] / 'data'


def _normalize_resize_max_side(resize_max_side):
    if resize_max_side is None:
        return None
    resize_max_side = int(resize_max_side)
    return resize_max_side if resize_max_side > 0 else None


def _compute_resized_size(image_size, resize_max_side):
    resize_max_side = _normalize_resize_max_side(resize_max_side)
    width, height = image_size
    if resize_max_side is None or max(width, height) <= resize_max_side:
        return width, height

    scale = resize_max_side / float(max(width, height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return resized_width, resized_height


def _structdiff_image_search_paths(image_name, data_root):
    return (
        data_root / 'images' / image_name,
        data_root / 'images' / 'paired' / image_name,
        data_root / 'images' / 'examples' / image_name,
    )


def _load_resized_pil_image(path, mode, resize_to=None, resample=_PIL_BILINEAR, **kwargs):
    image = Image.open(path, **kwargs).convert(mode=mode)
    if resize_to is not None and image.size != resize_to:
        image = image.resize(resize_to, resample=resample)
    return image


def _pil_image_to_array(image, bounds, pt):
    image = _img_to_float32(image, bounds)
    if pt:
        image = np2pt(image)
    return image


def resolve_structdiff_image_path(image_name, data_root=None):
    candidate = Path(image_name).expanduser()
    if candidate.exists():
        return candidate.resolve()

    data_root = _structdiff_data_root(data_root)
    searched = []
    for candidate in _structdiff_image_search_paths(image_name, data_root):
        searched.append(str(candidate))
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f'Could not find image "{image_name}". Searched: {", ".join(searched)}')


def load_structdiff_image(image_name, bounds=_BOUNDS, mode='RGB', pt=True, data_root=None, resize_max_side=None, **kwargs):
    image_path = resolve_structdiff_image_path(image_name, data_root=data_root)
    with Image.open(str(image_path), **kwargs) as pil_image:
        resize_to = _compute_resized_size(pil_image.size, resize_max_side)
    image = _pil_image_to_array(
        _load_resized_pil_image(str(image_path), mode=mode, resize_to=resize_to, resample=_PIL_BILINEAR, **kwargs),
        bounds=bounds,
        pt=pt,
    )
    return image, str(image_path)


def resolve_structdiff_mask_variant_path(image_name, variant_name, data_root=None):
    if variant_name is None:
        return None

    candidate = Path(variant_name).expanduser()
    if candidate.exists():
        return candidate.resolve()

    data_root = _structdiff_data_root(data_root)
    image_stem = Path(image_name).stem
    variant_dir = data_root / 'masks' / 'variants' / image_stem

    candidates = [variant_dir / variant_name]
    if Path(variant_name).suffix == '':
        candidates.append(variant_dir / f'{variant_name}.png')

    for path in candidates:
        if path.exists():
            return path

    searched = ', '.join(str(path) for path in candidates)
    raise FileNotFoundError(f'Could not find mask variant "{variant_name}" for "{image_name}". Searched: {searched}')


def load_structdiff_mask_variant(image_name, variant_name, bounds=_BOUNDS, mode='RGB', pt=True, data_root=None, resize_max_side=None, **kwargs):
    mask_path = resolve_structdiff_mask_variant_path(image_name, variant_name, data_root=data_root)
    image_path = resolve_structdiff_image_path(image_name, data_root=data_root)
    with Image.open(str(image_path), **kwargs) as pil_image:
        resize_to = _compute_resized_size(pil_image.size, resize_max_side)
    mask = _pil_image_to_array(
        _load_resized_pil_image(str(mask_path), mode=mode, resize_to=resize_to, resample=_PIL_NEAREST, **kwargs),
        bounds=bounds,
        pt=pt,
    )
    return mask, str(mask_path)


def load_structdiff_image_and_mask(image_name, bounds=_BOUNDS, mode='RGB', pt=True, data_root=None, resize_max_side=None, **kwargs):
    data_root = _structdiff_data_root(data_root)
    image_path = resolve_structdiff_image_path(image_name, data_root=data_root)
    with Image.open(str(image_path), **kwargs) as pil_image:
        original_image_size = pil_image.size
        resize_to = _compute_resized_size(original_image_size, resize_max_side)

    image = _pil_image_to_array(
        _load_resized_pil_image(str(image_path), mode=mode, resize_to=resize_to, resample=_PIL_BILINEAR, **kwargs),
        bounds=bounds,
        pt=pt,
    )

    default_mask_path = data_root / 'masks' / 'default' / f'{image_path.stem}_mask.png'
    if default_mask_path.exists():
        try:
            with Image.open(str(default_mask_path), **kwargs) as pil_mask:
                if pil_mask.size == original_image_size:
                    mask = _pil_image_to_array(
                        _load_resized_pil_image(
                            str(default_mask_path),
                            mode=mode,
                            resize_to=resize_to,
                            resample=_PIL_NEAREST,
                            **kwargs,
                        ),
                        bounds=bounds,
                        pt=pt,
                    )
                    used_full_foreground_mask = False
                else:
                    mask = torch.ones_like(image) if pt else np.ones_like(image, dtype=np.float32)
                    used_full_foreground_mask = True
                    default_mask_path = None
        except Exception:
            mask = torch.ones_like(image) if pt else np.ones_like(image, dtype=np.float32)
            used_full_foreground_mask = True
            default_mask_path = None
    else:
        mask = torch.ones_like(image) if pt else np.ones_like(image, dtype=np.float32)
        used_full_foreground_mask = True

    return (
        image,
        mask,
        str(image_path),
        (str(default_mask_path) if default_mask_path is not None and default_mask_path.exists() else None),
        used_full_foreground_mask,
    )


def imwrite(fname, image, bounds=_BOUNDS, **kwargs):
    fname = _check_path(fname)
    image = _img_to_uint8(image, bounds)
    image = Image.fromarray(image)
    image.save(fname, **kwargs)


if __IMSHOW_ENABLED:
    def _imshow(image, bounds, *, fname=None):
        fd = None
        if fname is None:
            fd, fname = tempfile.mkstemp(suffix='.png')
        imwrite(fname, image, bounds)
        if fd is not None:
            os.close(fd)

        img_output = widgets.Image.from_file(fname)
        display(img_output)

else:
    def _imshow(image, bounds, *, fname=None):
        raise NotImplementedError()


def imshow(img, bounds=_BOUNDS, **kwargs):
    if len(img.shape) == 4 and img.shape[0] == 1:
        img = img.squeeze(0)
    if len(img.shape) == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)

    return _imshow(img, bounds, **kwargs)


def tensor2npimg(x, vmin=-1, vmax=1, normmaxmin=False, to_numpy=True):
    """tensor in [-1,1] (1x3xHxW) --> numpy image ready to plt.imshow"""
    if normmaxmin:
        vmin = x.min().item()
        vmax = x.max().item()
    final = x[0].add(-vmin).div(vmax-vmin).mul(255).add(0.5).clamp(0, 255)

    if to_numpy:
        final = final.permute(1, 2, 0)
        # if input has 1-channel, pass grayscale to numpy
        if final.shape[-1] == 1:
            final = final[:,:,0]
        return final.to('cpu', torch.uint8).numpy()
    else:
        return final.to('cpu', torch.uint8)


torch255tonpimg = lambda x: x[0].add(0.5).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
