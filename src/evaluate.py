# src/evaluate.py
import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from src.dataset import make_dataloaders
from src.model import build_model
from src.utils import load_checkpoint

def evaluate_model(model_path,
                   data_dir='/kaggle/input/chest-x-ray/chest_xray',
                   batch_size=32,
                   image_size=224,
                   device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader, classes = make_dataloaders(
        data_dir, batch_size=batch_size, image_size=image_size
    )
    model = build_model(num_classes=len(classes), backbone='resnet18', pretrained=False)
    model.to(device)
    ckpt = load_checkpoint(model_path, model, device=device)
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            out = model(imgs)
            preds = out.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())
    print("Classes:", classes)
    print(classification_report(y_true, y_pred, target_names=classes))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    return classes
