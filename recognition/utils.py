# utils.py
# Helper functions (e.g., metrics, visualization)

import torch
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt

class preprocessing():
    def squeeze_channel(x):
        return x.squeeze(0)

    def label_map(x):
        return x // 85

    def to_long(x):
        return x.long()

class metrics():
    @staticmethod
    def dice_coeff(pred, target, eps=1e-6):
        """
        pred: [B, C, H, W] (raw logits or probabilities)
        target: [B, H, W] (integer class labels)
        """
        # Apply softmax if not yet normalized
        if pred.shape[1] > 1:
            pred = F.softmax(pred, dim=1)
        
        # One-hot encode target
        num_classes = pred.shape[1]
        target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        # Compute Dice per class
        dims = (0, 2, 3)
        intersection = torch.sum(pred * target_onehot, dims)
        cardinality = torch.sum(pred + target_onehot, dims)

        dice_per_class = (2. * intersection + eps) / (cardinality + eps)
        return dice_per_class.mean()  # mean over classes

    @staticmethod
    def dice_loss(pred, target, eps=1e-6):
        return 1 - metrics.dice_coeff(pred, target, eps)
        
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