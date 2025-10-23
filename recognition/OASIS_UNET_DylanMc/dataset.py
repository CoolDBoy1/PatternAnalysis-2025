import os
from torch.utils.data import Dataset
from PIL import Image

class OASIS2DDataset(Dataset):
    """
    Dataset for 2D OASIS MRI slices.
    Expects directories:
        keras_png_slices_[train/val/test]/       -> images
        keras_png_slices_seg_[train/val/test]/   -> masks
    """

    def __init__(self, image_dir, mask_dir, transform_img=None, transform_mask=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform_img = transform_img
        self.transform_mask = transform_mask

        # Collect and sort all files
        self.image_files = sorted([
            f for f in os.listdir(image_dir) if f.endswith(".nii.png")
        ])
        self.mask_files = sorted([
            f for f in os.listdir(mask_dir) if f.endswith(".nii.png")
        ])

        # Ensure the number of images matches the labels
        assert len(self.image_files) == len(self.mask_files), \
            f"Number of images and masks do not match: {len(self.image_files)} vs {len(self.mask_files)}"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        img = Image.open(img_path).convert("L")  # grayscale
        mask = Image.open(mask_path)
        
        if self.transform_img:
            img = self.transform_img(img)
        if self.transform_mask:
            mask = self.transform_mask(mask)
        
        return img, mask
