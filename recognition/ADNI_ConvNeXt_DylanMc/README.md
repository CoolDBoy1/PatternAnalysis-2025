# 3D ConvNeXt for Volumetric MRI Classification (Dylan M)

## Overview / Problem Description
This project implements a **3D ConvNeXt convolutional neural network** to classify 3D medical MRI scans, based on the ADNI dataset.  
It extends the ConvNeXt architecture (Liu et al., 2022) from 2D to 3D by adapting its convolutional, normalization, and downsampling layers to volumetric data.  
The algorithm addresses the challenge of **recognizing spatial patterns across three dimensions** in medical imaging — such as brain structure changes — for diagnostic classification.

## How It Works
The model builds a hierarchical feature extractor with multiple ConvNeXt stages:
1. **Patch Embedding:** A 3D convolution downsamples the input along spatial dimensions.  
2. **ConvNeXt Blocks:** Each stage consists of several depthwise 3D convolution blocks with layer normalization, GELU activation, linear expansion, and residual scaling (`gamma` parameter).  
3. **Downsampling Layers:** Between stages, 3D convolutions with stride (1,2,2) reduce feature map size while increasing channel depth.  
4. **Global Average Pooling:** Collapses spatial dimensions to a single vector.  
5. **Fully Connected Head:** Outputs class probabilities for binary classification.

Training uses **mixed precision (AMP)** for efficiency and **Cosine Annealing LR scheduling** for stable convergence.  
Loss is computed using **CrossEntropyLoss with label smoothing** and optimization is handled by **AdamW**.

## Hyperparameter Choices and Justifications
This section outlines the major hyperparameters used in training and explains the reasoning behind each choice.

### Model Architecture
| Parameter  | Value  | Justification |
|------------|--------|---------------|
| **Depths** | [2, 2, 4, 2] | A moderate depth distribution inspired by ConvNeXt-Tiny. Early stages (2–2) capture low-level spatial features, while deeper stages (4–2) allow hierarchical representation learning without over-parameterization. Balances expressivity and computational cost for 3D data. |
| **Dims** | [16, 32, 64, 128] | Progressively doubles feature channels per stage, enabling increased representational capacity as spatial resolution decreases. Using a smaller initial dimension (16) reduces memory usage for volumetric data compared to the standard 96 used in 2D ConvNeXt. |

### Optimization
| Parameter  | Value  | Justification |
|------------|--------|---------------|
| **Learning Rate** | 5e-4 | Chosen as an empirically stable starting point for AdamW on mid-sized networks. Too high causes divergence with mixed precision; too low slows convergence. Tuned after small-scale validation sweeps. |
| **Weight Decay** | 2e-4 | Standard regularization strength for ConvNeXt-style networks (per Loshchilov & Hutter, 2019). Controls overfitting without excessive penalization, especially beneficial for small datasets like ADNI. |
| **Drop Rate** | 0.4 | Applied progressively via `DropPath` (stochastic depth). 0.4 provides sufficient regularization at deeper stages, improving generalization and preventing co-adaptation between residual paths. |

### Learning Rate Scheduling
| Scheduler  | Parameter(s)  | Justification |
|------------|---------------|---------------|
| **CosineAnnealingLR** | `T_max = num_epochs`, `eta_min = lr × 0.01` | Cosine annealing gradually reduces the learning rate, promoting smoother convergence and avoiding sharp drops in validation performance. It cyclically explores larger steps early and fine-tunes near convergence. Empirically more stable than StepLR for medium-run experiments. |

The combination of **AdamW** and **cosine annealing** has been shown to achieve excellent convergence on modern architectures (Liu et al., 2022; Loshchilov & Hutter, 2019).  
Overall, these settings represent a **balance between stability, speed, and generalization** for high-dimensional 3D data, where model overfitting and GPU memory constraints are primary concerns.

### Data Augmentation Choices and Justifications
| Augmentation  | Parameters  | Purpose & Justification |
|---------------|-------------|--------------------------|
| **Padding (`transforms.Pad(padding=(0, 0, 16, 0))`)** | Adds extra pixels along one dimension | Ensures consistent spatial dimensions across subjects without geometric distortion. Maintains brain structure alignment along the depth axis. |
| **Center Crop (224 × 224)** | Standardized crop | Focuses the model on the central brain region (where most pathology occurs) and keeps aspect ratio consistent. Prevents edge artefacts from dominating learning. |
| **Normalization (`mean=0.0324, std=1.0224`)** | Z-score normalization | Standardizes voxel intensity distribution across subjects and scanners, improving numerical stability and convergence speed. |
| **ColorJitter (`brightness=0.05`, `contrast=0.05`)** | Mild contrast and brightness shifts | Simulates lighting or acquisition variability, ensuring the model remains invariant to minor global intensity changes. |
| **Add Gaussian Noise (`p=0.3`)** | Randomly applied noise perturbation | Mimics sensor or environmental noise, enhancing robustness to imperfect acquisition and preventing overfitting to clean training data. |
| **Random Gamma (`p=0.4`)** | Random nonlinear brightness transformation | Emulates exposure variation or scanner response differences. Encourages resilience to contrast changes across samples. |
| **Adjust Sharpness (`sharpness_factor=1.1`, `p=0.3`)** | Slight sharpening or softening | Models focus variation between captures. Prevents sensitivity to fine texture differences while retaining structural features. |
| **Intensity Jitter (`p=0.3`)** | Random additive intensity fluctuation | Accounts for small global scaling or baseline intensity shifts across datasets. Reduces dependence on fixed pixel distributions. |
| **RandomAffine (`degrees=0`, `translate=(0.02, 0.02)`, `scale=(0.98, 1.02)`)** | Subtle spatial translation and scaling | Introduces realistic positional and size variance without distorting semantic content. Avoids rotations or flips that could alter class meaning. |
| **Gaussian Blur (`kernel_size=3`, `sigma=(0.1, 1.0)`, `p=0.3`)** | Controlled spatial smoothing | Mimics slight motion blur or focus variability. Reduces high-frequency overfitting while maintaining overall structure. |

Each transformation was selected to increase the model’s robustness to minor acquisition and intensity variations while preserving anatomical fidelity — a crucial consideration for volumetric MRI data where structural orientation carries diagnostic meaning.

## Files
- `modules.py` : model components
- `dataset.py` : data loading and preprocessing
- `train.py` : training, validation, and testing
- `predict.py` : example usage of trained model on test data
- `utils.py` : functions which were useful in learning about the data and optimisation

## Usage
A batch of 3D MRI tensors (after preprocessing):
shape: [batch_size, 1, depth, height, width]
example: [16, 1, 20, 224, 224]

example inputs into the train loop:
    num_epochs=30
    num_workers=12
    batch_size=32
    in_chans=1
    num_classes=2
    lr=5e-4
    drop_rate=0.4

example epoch:
Epoch 6/30 [Train]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 27/27 [01:59<00:00,  4.41s/it, acc=0.592, loss=0.678] 
Epoch [6/30] Train Loss: 0.6779 Train Acc: 0.5919 Val Loss: 0.6414 Val Acc: 0.7083                                                                                                                                                                                                                   
New best model saved with val acc: 0.7083

example predict:
Evaluating on Test Set: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 113/113 [01:11<00:00,  1.59it/s]

=== Classification Report ===
              precision    recall  f1-score   support

          NC     0.8190    0.7974    0.8080       227
          AD     0.7991    0.8206    0.8097       223

    accuracy                         0.8089       450
   macro avg     0.8091    0.8090    0.8089       450
weighted avg     0.8092    0.8089    0.8089       450

Test Accuracy: 80.89%

![Confusion Matrix](https://github.com/CoolDBoy1/PatternAnalysis-2025/blob/topic-recognition/recognition/ADNI_ConvNeXt_DylanMc/confusion_matrix.png)


## Dependencies
| Package          | Version | Purpose                                                            |
| :--------------- | :------ | :----------------------------------------------------------------- |
| `torch`          | 2.3+    | Deep learning framework (PyTorch) for model training and inference |
| `torchvision`    | 0.18+   | Image transforms and utilities                                     |
| `numpy`          | 1.26+   | Numerical operations and array manipulation                        |
| `matplotlib`     | 3.8+    | Plotting training curves and visualizations                        |
| `tqdm`           | 4.66+   | Progress bar for training and inference loops                      |
| `Pillow` (`PIL`) | 10.0+   | Image file I/O and processing                                      |
| scikit-learn     | 1.3.0   | For splitting data between training and validation                 |
| timm             | 1.0.3   | Use the function drop path as used by the code released            |
| `sys`, `os`      | —       | System and path handling (Python standard library)                 |

## reproducability
Consistency of results varied greatly with it being very common for the model to get stuck in 65% accuracy range for the test data, this is also due to how random seeds were used and on thos of that the results may fluctuate due to stochastic data augmentations and DropPath

## References
[Liu et al., 2022](https://arxiv.org/abs/2201.03545)
[Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101)