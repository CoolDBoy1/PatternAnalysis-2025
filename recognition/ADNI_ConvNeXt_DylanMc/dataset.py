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

        # If fewer than num_slices, pad by repeating last slice
        if len(slice_paths) < self.num_slices:
            slice_paths += [slice_paths[-1]] * (self.num_slices - len(slice_paths))
        elif len(slice_paths) > self.num_slices:
            # uniform sampling
            idxs = np.linspace(0, len(slice_paths)-1, self.num_slices).astype(int)
            slice_paths = [slice_paths[i] for i in idxs]

        slices = []
        for path in slice_paths:
            img = Image.open(path).convert("L")  # grayscale
            if self.transform:
                img = self.transform(img)
            slices.append(img)

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

def get_datasets(train_root="PatternAnalysis-2025/recognition/ADNI/AD_NC/train", test_root="PatternAnalysis-2025/recognition/ADNI/AD_NC/test", num_slices=20):
    ad_brains, ad_labels = collect_brains(os.path.join(train_root, "AD"), label_value=1)
    nc_brains, nc_labels = collect_brains(os.path.join(train_root, "NC"), label_value=0)

    all_brains = ad_brains + nc_brains
    all_labels = ad_labels + nc_labels

    # Split into train/val (ensures no brain overlap)
    train_brains, val_brains, train_labels, val_labels = train_test_split(
        all_brains, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_set = ADNIDataset(train_brains, train_labels, num_slices=num_slices, transform=transform)
    val_set = ADNIDataset(val_brains, val_labels, num_slices=num_slices, transform=transform)

    # Prepare test set
    test_brains_ad, test_labels_ad = collect_brains(os.path.join(test_root, "AD"), label_value=1)
    test_brains_nc, test_labels_nc = collect_brains(os.path.join(test_root, "NC"), label_value=0)
    test_brains = test_brains_ad + test_brains_nc
    test_labels = test_labels_ad + test_labels_nc
    test_set = ADNIDataset(test_brains, test_labels, num_slices=num_slices, transform=transform)

    return train_set, val_set, test_set

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
    
    train_set, val_set, test_set = get_datasets(num_slices=20)
    print(f"Train samples: {len(train_set)}")
    print(f"Val samples: {len(val_set)}")
    print(f"Test samples: {len(test_set)}")

    loader = DataLoader(train_set, batch_size=10, shuffle=True)
    x, y = next(iter(loader))
    print("Batch shape:", x.shape)
    print("Batch labels:", y)
