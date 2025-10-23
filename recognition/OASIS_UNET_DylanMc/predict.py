import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader

from dataset import OASIS2DDataset
from modules import ImprovedUNet
from utils import visualize, preprocessing, metrics

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
    test_transform_img, test_transform_mask = preprocessing.OASIS2Dtransforms

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
    
    # Eval model
    avg_dice = metrics.compute_dice(
        model,
        test_loader,
        device,
        output_dir=output_dir,
        visualize_fn=visualize.save_prediction_mask
    )
    print(f"\nPrediction complete. Average Dice: {avg_dice:.4f}")

    return avg_dice

if __name__ == "__main__":
    predict_loop()
