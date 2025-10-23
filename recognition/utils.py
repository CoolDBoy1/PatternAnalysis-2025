# utils.py
# Helper functions (e.g., metrics, visualization)

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

class preprocessing():       
    def squeeze_channel(x):
        return x.squeeze(0)

    def label_map(x):
        return x // 85

    def to_long(x):
        return x.long()
    
    transform_img = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((256,256), interpolation=Image.NEAREST),
        transforms.PILToTensor(),
        transforms.Lambda(squeeze_channel),
        transforms.Lambda(label_map),
        transforms.Lambda(to_long)
    ])
    
    OASIS2Dtransforms = transform_img, transform_mask

class DiceLoss(nn.Module):
    """Dice Loss for multi-class segmentation."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, outputs, targets):
        """
        outputs: [B, C, H, W] raw logits
        targets: [B, H, W] integer class labels
        """
        num_classes = outputs.shape[1]
        # Hard one-hot encoding of targets
        targets_onehot = F.one_hot(targets, num_classes).permute(0,3,1,2).float()

        # Probabilities
        probs = F.softmax(outputs, dim=1)

        # Compute per-class Dice
        dims = (0,2,3)  # sum over batch + H + W
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality = torch.sum(probs + targets_onehot, dims)
        dice_per_class = (2 * intersection + self.eps) / (cardinality + self.eps)

        # Average over classes
        return 1 - dice_per_class.mean()
    
class metrics():
    @staticmethod
    def compute_dice(model, dataloader, device, output_dir=None, visualize_fn=None, save_preds=True, max_vis=3, amp=True):
        """
        Computes per-class Dice coefficient over a dataset and optionally saves predictions.

        Args:
            model: torch.nn.Module, the trained model.
            dataloader: DataLoader for the dataset to evaluate.
            device: torch.device.
            output_dir: Optional[str], directory to save predictions as .npy.
            visualize_fn: Optional[callable], function(images, preds, masks, output_dir, idx) for visualization.
            save_preds: bool, whether to save predictions as .npy.
            max_vis: int, maximum number of batches to visualize.
            amp: bool, whether to use automatic mixed precision.

        Returns:
            avg_dice: float, mean Dice over all classes and batches.
        """
        model.eval()
        dice_scores = []

        device_type = "cuda" if "cuda" in str(device) else "cpu"

        with torch.no_grad():
            for i, (images, masks) in enumerate(tqdm(dataloader, desc="Computing Dice")):
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

                if amp:
                    with torch.autocast(device_type=device_type):
                        outputs = model(images)
                else:
                    outputs = model(images)

                preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)

                # Compute per-class Dice
                num_classes = outputs.shape[1]
                for cls in range(num_classes):
                    pred_cls = (preds == cls).float()
                    true_cls = (masks == cls).float()
                    intersection = (pred_cls * true_cls).sum()
                    dice = (2. * intersection) / (pred_cls.sum() + true_cls.sum() + 1e-8)
                    dice_scores.append(dice.item())

                # Optional save
                if save_preds and output_dir:
                    np.save(f"{output_dir}/pred_{i}.npy", preds.cpu().numpy())

                # Optional visualize
                if visualize_fn and i < max_vis:
                    visualize_fn(images, preds, masks, output_dir, i)

        return float(np.mean(dice_scores))
        
class visualize():
    @staticmethod
    def save_prediction_mask(images, preds, masks, output_dir, idx):
        image = images[0].cpu().squeeze()
        pred = preds[0].cpu().squeeze()
        mask = masks[0].cpu().squeeze()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image, cmap="gray"); axes[0].set_title("Input")
        axes[1].imshow(pred, cmap="viridis"); axes[1].set_title("Prediction")
        axes[2].imshow(mask, cmap="viridis"); axes[2].set_title("Ground Truth")
        [a.axis("off") for a in axes]
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"vis_{idx}.png"))
        plt.close()