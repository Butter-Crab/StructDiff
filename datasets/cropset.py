import torch
from torch.utils.data import Dataset
import random
from torchvision import transforms

class RandomCropWithCoord(transforms.RandomCrop):
    """
    Custom RandomCrop to return the same cropping for both image and mask.
    """
    def forward(self, img):
        # Get crop parameters
        i, j, h, w = self.get_params(img, self.size)
        return transforms.functional.crop(img, i, j, h, w), (i, j, h, w)

class CropSet(Dataset):
    """
    A dataset comprised of crops of a single image or several images.
    """
    def __init__(self, mask_image, image, head_grid, crop_size, use_flip=True, dataset_size=5000):
        """
        Args:
            mask_image (torch.tensor): The mask to generate crops from. Same shape as image.
            image (torch.tensor): The image to generate crops from. Can be of shape (C,H,W) or (B,C,H,W).
            crop_size (tuple(int, int)): The spatial dimensions of the crops to be taken.
            use_flip (bool): Whether to use horizontal flips of the image.
            dataset_size (int): Number of crops in a single epoch of training.
        """
        self.crop_size = crop_size
        self.dataset_size = dataset_size
        self.mask_image = mask_image
        self.head_grid = head_grid

        self.flip = transforms.RandomHorizontalFlip() if use_flip else None
        self.random_crop = RandomCropWithCoord(self.crop_size)

        self.img = image

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        img = self.img if len(self.img.shape) == 3 else random.choice(self.img)
        mask = self.mask_image if len(self.mask_image.shape) == 3 else random.choice(self.mask_image)
        grid = None if self.head_grid is None else (self.head_grid if len(self.mask_image.shape) == 3 else random.choice(self.head_grid))

        if self.flip:
            img = self.flip(img)
            mask = self.flip(mask)

        img_crop, crop_params = self.random_crop(img)
        mask_crop = transforms.functional.crop(mask, *crop_params)

        img_crop = (img_crop[:3, ] * 2) - 1

        if grid is None:
            return {'IMG': torch.cat((img_crop, mask_crop))}

        grid_crop = transforms.functional.crop(grid, *crop_params)
        return {'IMG': torch.cat((img_crop, mask_crop, grid_crop))}
