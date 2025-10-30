# train.py
import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from tqdm import tqdm

from dataset import get_datasets
from modules import ConvNeXt

def train_loop(
    save_dir="PatternAnalysis-2025/recognition/ADNI_ConvNeXt_DylanMc/checkpoints",
    num_epochs=20,
    num_workers=8, 
    batch_size=8, 
    in_chans=1,
    num_classes=2,
    lr=1e-4,
):
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Training transforms
    train_transform = transforms.Compose([
        transforms.Pad(padding=(0, 0, 16, 0)),  # pad width 240 to 256
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.0324], [1.0224])
    ])
    
    train_set, val_set = get_datasets("train", num_slices=20, split_data=True)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                            num_workers=int(num_workers/2), pin_memory=True)
    
    # Model, Loss, Optimizer
    model = ConvNeXt(in_chans=in_chans, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    scaler = GradScaler(device=device)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, 
                                  threshold=1e-4, threshold_mode='rel', cooldown=1, min_lr=1e-6)

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
        
        scheduler.step(val_loss)
        
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
    batch_size=20
    in_chans=1
    num_classes=2
    lr=1e-4
    train_loop(num_epochs=num_epochs, num_workers=num_workers, batch_size=batch_size, 
               in_chans=in_chans, num_classes=num_classes, lr=lr)