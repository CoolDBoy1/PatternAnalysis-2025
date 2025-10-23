# utils.py
# Helper functions (e.g., metrics, visualization)

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class preprocessing():
    def squeeze_channel(x):
        return x.squeeze(0)

    def label_map(x):
        return x // 85

    def to_long(x):
        return x.long()
    

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
    def dice_coeff():
        pass
        
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