import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader

from dataset import OASIS2DDataset
from modules import ImprovedUNet
from utils import visualize, preprocessing, metrics

def predict_loop(
    data_dir="PatternAnalysis-2025/recognition/ADNI",
    model_path="PatternAnalysis-2025/recognition/ADNI_ConvNeXt_DylanMc/checkpoints/best_model.pth",
    output_dir="PatternAnalysis-2025/recognition/ADNI_ConvNeXt_DylanMc/predictions",

):
    pass

if __name__ == "__main__":
    predict_loop()
