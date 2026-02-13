# src/train.py
import os
import time
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from src.dataset import make_dataloaders
from src.model import build_model
from src.utils import set_seed, save_checkpoint

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    preds_all = []
    targets_all = []
    for images, targets in tqdm(loader, desc="Train"):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        preds_all.extend(preds)
        targets_all.extend(targets.cpu().numpy())
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    acc = accuracy_score(targets_all, preds_all) if preds_all else 0.0
    return avg_loss, acc

def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            preds_all.extend(preds)
            targets_all.extend(targets.cpu().numpy())
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    acc = accuracy_score(targets_all, preds_all) if preds_all else 0.0
    return avg_loss, acc

def main(data_dir='/kaggle/input/chest-x-ray/chest_xray',
         out_dir='/kaggle/working/models',
         epochs=8,
         batch_size=32,
         lr=1e-4,
         image_size=224,
         backbone='resnet18',
         seed=42):
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    train_loader, val_loader, test_loader, classes = make_dataloaders(
        data_dir, batch_size=batch_size, image_size=image_size
    )
    print("Classes:", classes)
    model = build_model(num_classes=len(classes), backbone=backbone, pretrained=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    os.makedirs(out_dir, exist_ok=True)
    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}/{epochs}")
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        t1 = time.time()
        print(f"Train loss: {train_loss:.4f} acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f} acc: {val_acc:.4f}")
        print("Epoch time: {:.1f}s".format(t1 - t0))
        last_path = os.path.join(out_dir, 'last_checkpoint.pth')
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'classes': classes
        }, last_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(out_dir, 'best_model.pth')
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'classes': classes
            }, best_path)
            print(f"Saved new best at {best_path}")
    print("Training finished. Models saved in", out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/kaggle/input/chest-x-ray/chest_xray')
    parser.add_argument('--out-dir', default='/kaggle/working/models')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--backbone', type=str, default='resnet18')
    args = parser.parse_args()
    main(data_dir=args.data_dir, out_dir=args.out_dir, epochs=args.epochs,
         batch_size=args.batch_size, lr=args.lr, image_size=args.image_size,
         backbone=args.backbone)
