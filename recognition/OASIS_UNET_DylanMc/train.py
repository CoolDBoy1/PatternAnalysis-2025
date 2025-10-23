import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import OASIS2DDataset
from modules import ImprovedUNet
from utils import DiceLoss, preprocessing, metrics

def train_loop(
    data_dir="PatternAnalysis-2025/recognition/OASIS",
    save_dir="PatternAnalysis-2025/recognition/OASIS_UNET_DylanMc/checkpoints",
    num_classes=4,
    in_channels=1,
    batch_size=24,
    num_epochs=10,
    learning_rate=1e-3,
):
    """
    Train an Improved UNet on the 2D OASIS dataset.

    Args:
        data_dir (str): Path to dataset containing train/val folders.
        save_dir (str): Directory to save checkpoints.
        num_classes (int): Number of segmentation labels.
        in_channels (int): Number of input channels (1 for grayscale MRI).
        batch_size (int): Batch size.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for optimizer.

    Returns:
        model (torch.nn.Module): Trained model.
        best_val_dice (float): Highest Dice coefficient achieved.
    """

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    
    os.makedirs(save_dir, exist_ok=True)

    # Training transforms with augmentation
    transform_img, transform_mask = preprocessing.OASIS2Dtransforms

    # Datasets and Loaders
    train_dataset = OASIS2DDataset(
        image_dir=os.path.join(data_dir, "keras_png_slices_train"),
        mask_dir=os.path.join(data_dir, "keras_png_slices_seg_train"),
        transform_img=transform_img,
        transform_mask=transform_mask
    )

    val_dataset = OASIS2DDataset(
        image_dir=os.path.join(data_dir, "keras_png_slices_validate"),
        mask_dir=os.path.join(data_dir, "keras_png_slices_seg_validate"),
        transform_img=transform_img,
        transform_mask=transform_mask
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=12, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=6, pin_memory=True)

    # Model, Loss, Optimizer
    model = ImprovedUNet(in_channels=in_channels, out_channels=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = DiceLoss()
    scaler = GradScaler(device=device)

    best_val_dice = 0.0

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

        for images, masks in loop:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type=device_type):
                outputs = model(images)
                loss = criterion(outputs, masks.long())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss = loss.item()
            running_loss += loss
            loop.set_postfix(loss=loss)

        avg_train_loss = running_loss / len(train_loader)

        # Validation
        avg_val_dice = metrics.compute_dice(model, val_loader, device)
        scheduler.step(avg_val_dice)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Dice={avg_val_dice:.4f}")

        # Save best model
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"Saved best model with Dice={best_val_dice:.4f}")

    print("Training complete.")
    return model, best_val_dice


if __name__ == "__main__":
    train_loop()