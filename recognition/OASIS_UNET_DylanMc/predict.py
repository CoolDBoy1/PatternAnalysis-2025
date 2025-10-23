import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from PIL import Image

from dataset import OASIS2DDataset
from modules import ImprovedUNet
from utils import visualize, preprocessing

def predict_loop(
    data_dir="PatternAnalysis-2025/recognition/OASIS",
    model_path="PatternAnalysis-2025/recognition/OASIS_UNET_DylanMc/checkpoints/best_model.pth",
    output_dir="PatternAnalysis-2025/recognition/OASIS_UNET_DylanMc/predictions",
    num_classes=4,
    in_channels=1,
    batch_size=24,
):
    """
    Run inference using the best trained model and compute Dice score on test set.

    Args:
        data_dir (str): Path to dataset containing test/images and test/labels.
        model_path (str): Path to trained model checkpoint.
        output_dir (str): Where to save predicted masks.
        num_classes (int): Number of segmentation labels.
        in_channels (int): Number of input channels.
        batch_size (int): Batch size for inference.

    Returns:
        avg_dice (float): Average Dice coefficient across the test set.
    """
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)

    os.makedirs(output_dir, exist_ok=True)
    
    # Transforms
    test_transform_img = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])

    test_transform_mask = transforms.Compose([
        transforms.Resize((256,256), interpolation=Image.NEAREST),
        transforms.PILToTensor(),
        transforms.Lambda(preprocessing.squeeze_channel),
        transforms.Lambda(preprocessing.label_map),
        transforms.Lambda(preprocessing.to_long)
    ])

    # Dataset and Loader
    test_dataset = OASIS2DDataset(
        image_dir=os.path.join(data_dir, "keras_png_slices_test"),
        mask_dir=os.path.join(data_dir, "keras_png_slices_seg_test"),
        transform_img=test_transform_img,
        transform_mask=test_transform_mask
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=6, pin_memory=True)

    # Load model
    model = ImprovedUNet(in_channels=in_channels, out_channels=num_classes).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    dice_scores = []

    # Inference loop
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc="Predicting")):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

            with autocast(device_type=device_type):
                outputs = model(images)
                preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)

                for cls in range(outputs.shape[1]):
                    pred_cls = (preds == cls).float()
                    true_cls = (masks == cls).float()
                    intersection = (pred_cls * true_cls).sum()
                    dice = (2. * intersection) / (pred_cls.sum() + true_cls.sum() + 1e-8)
                    dice_scores.append(dice.item())

            # Optional save
            np.save(os.path.join(output_dir, f"pred_{i}.npy"), preds.cpu().numpy())

            # Optional: visualize a few masks
            if i < 3:  # only first few for sanity check
                visualize.save_prediction_mask(images, preds, masks, output_dir, i)

    avg_dice = torch.tensor(dice_scores).mean().item()
    print(f"\nPrediction complete. Average Dice: {avg_dice:.4f}")

    return avg_dice

if __name__ == "__main__":
    predict_loop()
