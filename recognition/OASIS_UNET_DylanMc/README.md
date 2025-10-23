# Image Segmentation using U-Net (Dylan M)

## Overview / Problem Description
The **Improved U-Net** is a deep learning model designed for **multi-class segmentation of 2D brain MRI slices** from the OASIS dataset. The goal is to automatically delineate different anatomical regions, enabling precise analysis of brain structure. Accurate segmentation is essential for medical research, diagnostics, and tracking structural changes across patients. This algorithm solves the problem of identifying and separating these regions in grayscale MRI images where boundaries between tissues may be subtle and complex.

## How It Works
The algorithm follows a **U-Net architecture** with an encoder-decoder structure and skip connections. The **encoder path** extracts hierarchical features from the input image via convolutional layers and downsampling, capturing semantic information at multiple scales. The **decoder path** reconstructs the segmentation mask by upsampling and combining the encoder features through **skip connections**, which preserve spatial details and fine structures. Each block includes convolutional layers with **BatchNorm, ReLU activations, optional Dropout, and residual connections**, improving training stability and convergence. The network is trained using a **multi-class Dice loss**, directly optimizing overlap between predicted masks and ground truth, allowing the model to focus on accurate boundary alignment and overall region segmentation.

## Files
- `modules.py` : model components
- `dataset.py` : data loading and preprocessing
- `train.py` : training, validation, and testing
- `predict.py` : example usage of trained model on test data

## Usage
running train.py with:
    num_classes=4,
    in_channels=1,
    batch_size=24,
    num_epochs=10,
    learning_rate=1e-3,
    num_workers=12 (for train dataloader)
    num_workers=6 (for val dataloader)

Epoch [10/10]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 403/403 [03:31<00:00,  1.90it/s, loss=0.0241]
Computing Dice: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 47/47 [00:23<00:00,  2.03it/s] 
Epoch 10: Train Loss=0.0292, Val Dice=0.9672

running predict.py on the saved model gave:
Computing Dice: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 23/23 [00:21<00:00,  1.10it/s]
Prediction complete. Average Dice: 0.9707

For an example of the visualisation of test data:
https://github.com/CoolDBoy1/PatternAnalysis-2025/blob/topic-recognition/recognition/OASIS_UNET_DylanMc/vis_0.png
https://github.com/CoolDBoy1/PatternAnalysis-2025/blob/topic-recognition/recognition/OASIS_UNET_DylanMc/vis_1.png
https://github.com/CoolDBoy1/PatternAnalysis-2025/blob/topic-recognition/recognition/OASIS_UNET_DylanMc/vis_2.png

## Dependencies
| Package          | Version | Purpose                                                            |
| :--------------- | :------ | :----------------------------------------------------------------- |
| `torch`          | 2.3+    | Deep learning framework (PyTorch) for model training and inference |
| `torchvision`    | 0.18+   | Image transforms and utilities                                     |
| `numpy`          | 1.26+   | Numerical operations and array manipulation                        |
| `matplotlib`     | 3.8+    | Plotting training curves and visualizations                        |
| `tqdm`           | 4.66+   | Progress bar for training and inference loops                      |
| `Pillow` (`PIL`) | 10.0+   | Image file I/O and processing                                      |
| `sys`, `os`      | —       | System and path handling (Python standard library)                 |
