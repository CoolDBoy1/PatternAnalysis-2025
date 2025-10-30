# dataset.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np

class ADNIDataset(Dataset):
    def __init__(self, brains, labels, num_slices=20, transform=None):
        self.brains = brains
        self.labels = labels
        self.num_slices = num_slices
        self.transform = transform

    def __len__(self):
        return len(self.brains)

    def __getitem__(self, idx):
        slice_paths = self.brains[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Sort slices to ensure consistent order (1, 2, 3, ...)
        slice_paths = sorted(slice_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[1]))

        slices = []
        for path in slice_paths:
            img = Image.open(path).convert("L")  # grayscale
            if self.transform:
                img = self.transform(img)
            slices.append(img)
        
        if len(slices) != self.num_slices:
            raise AssertionError(f"should be {self.num_slices} slices")

        volume = torch.stack(slices, dim=1)
        return volume, label  # [1, D, H, W] for 3D CNN input

def collect_brains(folder_path, label_value = None):
    brains = {}
    for fname in os.listdir(folder_path):
        if not fname.endswith(".jpeg"):
            continue
        brain_id = fname.split("_")[0]
        if brain_id not in brains:
            brains[brain_id] = []
        brains[brain_id].append(os.path.join(folder_path, fname))
    
    if label_value is None:
        return brains
    return list(brains.values()), [label_value] * len(brains)

def get_datasets(option="train", num_slices=20, split_data=False, transform=None):
    root = os.path.join("PatternAnalysis-2025/recognition/ADNI/AD_NC", option)
    ad_brains, ad_labels = collect_brains(os.path.join(root, "AD"), label_value=1)
    nc_brains, nc_labels = collect_brains(os.path.join(root, "NC"), label_value=0)

    all_brains = ad_brains + nc_brains
    all_labels = ad_labels + nc_labels

    base_transform = transforms.Compose([
        transforms.Pad(padding=(0, 0, 16, 0)),  # pad width 240 to 256
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.0324], [1.0224])
    ])

    if transform == None:
        transform = base_transform

    if split_data:
        # Split into train/val (ensures no brain overlap)
        t_brains, val_brains, t_labels, val_labels = train_test_split(
            all_brains, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        val_set = ADNIDataset(val_brains, val_labels, num_slices=num_slices, transform=base_transform)
    else:
        t_brains, t_labels = all_brains, all_labels
        val_set = None
    
    t_set = ADNIDataset(t_brains, t_labels, num_slices=num_slices, transform=transform)

    return t_set, val_set

def slice_stats(train_root="PatternAnalysis-2025/recognition/ADNI/AD_NC/train"):
    ad_brains = collect_brains(os.path.join(train_root, "AD"))
    nc_brains = collect_brains(os.path.join(train_root, "NC"))

    all_brains = {**ad_brains, **nc_brains}

    slice_counts = [len(slices) for slices in all_brains.values()]

    print(f"Total brains: {len(all_brains)}")
    print(f"Min slices: {np.min(slice_counts)}")
    print(f"Max slices: {np.max(slice_counts)}")
    print(f"Mean slices: {np.mean(slice_counts):.2f}")
    print(f"Median slices: {np.median(slice_counts)}")
    
if __name__ == "__main__":
    slice_stats()
    
    train_set, val_set = get_datasets("train", num_slices=20, split_data=True)
    print(f"Train samples: {len(train_set)}")
    print(f"Val samples: {len(val_set)}")
    
    test_set, _ = get_datasets("test", num_slices=20)
    print(f"Test samples: {len(test_set)}")

    for set in (train_set, val_set, test_set):
        loader = DataLoader(set, batch_size=10, shuffle=True)
        x, y = next(iter(loader))
        print("Batch shape:", x.shape)
        print("Batch labels:", y)