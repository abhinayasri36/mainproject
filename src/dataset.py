# src/dataset.py
import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

def is_pneumonic_folder(name: str) -> bool:
    n = name.lower()
    return ('pneu' in n) or ('viral' in n) or ('bacter' in n) or ('pneumonia' in n) or ('pneumonic' in n)

def collect_image_paths(root_dir: str, split_name: str = 'train'):
    split_dir = os.path.join(root_dir, split_name)
    image_paths = []
    labels = []
    if not os.path.isdir(split_dir):
        return image_paths, labels
    exts = ('*.png','*.jpg','*.jpeg','*.bmp')
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(split_dir, '**', e), recursive=True))
    for f in files:
        parts = os.path.normpath(f).split(os.sep)
        assigned = None
        for p in reversed(parts[:-1]):
            pn = p.lower()
            if 'normal' in pn:
                assigned = 0
                break
            if is_pneumonic_folder(pn):
                assigned = 1
                break
        if assigned is None:
            fname = os.path.basename(f).lower()
            if 'normal' in fname:
                assigned = 0
            elif 'bacter' in fname or 'viral' in fname or 'pneu' in fname:
                assigned = 1
            else:
                assigned = 1
        image_paths.append(f)
        labels.append(assigned)
    return image_paths, labels

class SimpleChestXrayDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.paths, self.labels = collect_image_paths(root_dir, split_name=split)
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under {os.path.join(root_dir, split)}")
        self.classes = ['NORMAL', 'PNEUMONIC']

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

def get_transforms(image_size=224):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(int(image_size*1.1)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return train_transform, val_transform

def make_dataloaders(root_dir,
                     batch_size=32,
                     image_size=224,
                     val_split_ratio=0.15,
                     num_workers=2):
    train_t, val_t = get_transforms(image_size)
    train_ds = SimpleChestXrayDataset(root_dir, split='train', transform=train_t)
    n_total = len(train_ds)
    n_val = int(n_total * val_split_ratio)
    if n_val <= 0:
        n_val = max(1, int(0.05 * n_total))
    n_train = n_total - n_val
    train_subset, val_subset = random_split(train_ds, [n_train, n_val])
    val_subset.dataset.transform = val_t
    try:
        test_ds = SimpleChestXrayDataset(root_dir, split='test', transform=val_t)
    except RuntimeError:
        test_ds = val_subset
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    classes = train_ds.classes
    return train_loader, val_loader, test_loader, classes
