# train.py
import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from tqdm import tqdm
import random

from dataset import get_datasets
from modules import ConvNeXt

def add_gaussian_noise(x, std=0.03):
    """Add small Gaussian noise to tensor image."""
    noise = torch.randn_like(x) * std
    return torch.clamp(x + noise, 0.0, 1.0)

def random_gamma(x, gamma_min=0.9, gamma_max=1.1):
    """Apply random gamma adjustment."""
    gamma = random.uniform(gamma_min, gamma_max)
    return torch.clamp(x ** gamma, 0.0, 1.0)

def add_intensity_jitter(x, scale=0.02):
    """Add slight random intensity offset (simulate scanner differences)."""
    offset = torch.randn(1).item() * scale
    return torch.clamp(x + offset, 0.0, 1.0)

def train_loop(
    save_dir="PatternAnalysis-2025/recognition/ADNI_ConvNeXt_DylanMc/checkpoints",
    num_epochs=20,
    num_workers=8, 
    batch_size=8, 
    in_chans=1,
    num_classes=2,
    lr=1e-4,
    drop_rate=0.1
):
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    
    os.makedirs(save_dir, exist_ok=True)
        
    # Training transforms
    train_transform = transforms.Compose([
        transforms.Pad(padding=(0, 0, 16, 0)),  # pad width 240 -> 256
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.0324], [1.0224]),

        # ---- photometric ----
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.RandomApply([transforms.Lambda(add_gaussian_noise)], p=0.3),
        transforms.RandomApply([transforms.Lambda(random_gamma)], p=0.4),
        transforms.RandomAdjustSharpness(sharpness_factor=1.1, p=0.3),
        transforms.RandomApply([transforms.Lambda(add_intensity_jitter)], p=0.3),

        # ---- spatial ----
        transforms.RandomAffine(
            degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02)
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=0.3),
    ])
    
    train_set, val_set = get_datasets("train", num_slices=20, split_data=True, transform=train_transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                            num_workers=int(num_workers/12), pin_memory=True)
    
    # Model, Loss, Optimizer
    model = ConvNeXt(in_chans=in_chans, num_classes=num_classes, drop_rate=drop_rate).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    scaler = GradScaler(device=device)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    best_val_acc = 0.0
    
    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for x, y in loop:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            with autocast(device_type=device_type):
                outputs = model(x)
                loss = criterion(outputs, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            loop.set_postfix(loss=running_loss/total, acc=correct/total)

        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]  ", leave=False)
            for x, y in val_loop:
                x = x.to(device)
                y = y.to(device)
                
                with autocast(device_type=device_type):
                    outputs = model(x)
                    loss = criterion(outputs, y)
                val_loss += loss.item() * x.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
                
                val_loop.set_postfix(loss=val_loss/val_total, acc=val_correct/val_total)

        val_loss /= val_total
        val_acc = val_correct / val_total
        
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"New best model saved with val acc: {best_val_acc:.4f}")
            
    print(f"Training complete. Best val accuracy: {best_val_acc:.4f}")
    return model, best_val_acc

if __name__ == "__main__":
    # preds
    num_epochs=30
    num_workers=12
    batch_size=32
    in_chans=1
    num_classes=2
    lr=5e-4
    drop_rate=0.4
    train_loop(num_epochs=num_epochs, num_workers=num_workers, batch_size=batch_size, 
               in_chans=in_chans, num_classes=num_classes, lr=lr, drop_rate=drop_rate)