# predict.py
import sys, os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm

from dataset import get_datasets
from modules import ConvNeXt

def predict_loop(
    model_path="PatternAnalysis-2025/recognition/ADNI_ConvNeXt_DylanMc/checkpoints/best_model.pth",
    batch_size=4,
    num_workers=8,
    in_chans=1,
    num_classes=2,
    num_slices=20,
):
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)

    # Load test dataset
    test_set, _ = get_datasets("test", num_slices=num_slices)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Load model
    model = ConvNeXt(in_chans=in_chans, num_classes=num_classes).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict=state_dict)
    model.eval()

    all_labels = []
    all_preds = []

    # Inference loop
    with torch.no_grad():
        loop = tqdm(test_loader, desc="Evaluating on Test Set")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["NC", "AD"], digits=4)

    print("\n=== Classification Report ===")
    print(report)

    # Plot confusion matrix with TP/TN/FP/FN
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred NC", "Pred AD"],
                yticklabels=["True NC", "True AD"])
    plt.title("Confusion Matrix (TP, TN, FP, FN)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    
    out_path = os.path.join(os.path.dirname(model_path), "confusion_matrix.png")
    plt.savefig(out_path)

    acc = np.mean(all_labels == all_preds)
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"Confusion matrix saved to {out_path}")

    return acc, cm, report

if __name__ == "__main__":
    # preds
    model_path="PatternAnalysis-2025/recognition/ADNI_ConvNeXt_DylanMc/checkpoints/best_model.pth"
    batch_size=4
    num_workers=8
    in_chans=1
    num_classes=2
    num_slices=20
        
    predict_loop(
        model_path=model_path,
        batch_size=batch_size,
        num_workers=num_workers,
        in_chans=in_chans,
        num_classes=num_classes,
        num_slices=num_slices
    )