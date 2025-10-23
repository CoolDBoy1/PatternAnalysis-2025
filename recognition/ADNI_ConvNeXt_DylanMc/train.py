import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import ADNIDataset
from modules import ConvNeXt
from utils import preprocessing, metrics

def train_loop(
    data_dir="PatternAnalysis-2025/recognition/ADNI",
    save_dir="PatternAnalysis-2025/recognition/ADNI_ConvNeXt_DylanMc/checkpoints",
    
):
    pass


if __name__ == "__main__":
    train_loop()