# utils.py
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import get_datasets

def compute_dataset_stats(dataset, num_workers=8, batch_size=4, max_batches=None):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    n_pixels = 0
    mean = 0.0
    M2 = 0.0  # for variance accumulation (Welford's method)

    for i, (volumes, _) in enumerate(tqdm(loader, desc="Computing mean/std")):
        # volumes: [B, 1, D, H, W]
        x = volumes.view(-1)  # flatten everything
        batch_n = x.numel()
        batch_mean = x.mean().item()
        batch_var = x.var().item()

        delta = batch_mean - mean
        n_pixels += batch_n
        mean += delta * batch_n / n_pixels
        M2 += batch_var * batch_n + delta**2 * (batch_n * (n_pixels - batch_n) / n_pixels)

        if max_batches and i >= max_batches:
            break

    var = M2 / (n_pixels - 1)
    std = var**0.5
    return mean, std


if __name__ == "__main__":
    from torchvision import transforms
    base_transform = transforms.Compose([
        transforms.Pad(padding=(0, 0, 16, 0)),  # pad width 240 to 256
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    train_set, _ = get_datasets("train", num_slices=20, split_data=True)
    mean, std = compute_dataset_stats(train_set)
    print(f"Dataset mean={mean:.4f}, std={std:.4f}")
