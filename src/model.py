# src/model.py
import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes: int = 3,
                backbone: str = 'resnet18',
                pretrained: bool = True,
                dropout_p: float = 0.5) -> nn.Module:
    """
    Build a ResNet backbone and replace the final classifier with:
        nn.Sequential(nn.Dropout(dropout_p), nn.Linear(in_feats, num_classes))

    Default `num_classes=3` to support: [NORMAL, BACTERIAL, VIRAL].

    Args:
        num_classes: number of output classes (3 for NORMAL/BACTERIAL/VIRAL).
        backbone: one of 'resnet18', 'resnet34', 'resnet50'.
        pretrained: whether to use ImageNet pretrained weights for the backbone.
        dropout_p: dropout probability before the final linear layer.

    Returns:
        model: a torchvision ResNet with a replaced final head.
    """
    if backbone == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif backbone == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
    elif backbone == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_feats, num_classes)
    )

    return model


def build_model_from_checkpoint(checkpoint_path: str,
                                backbone: str = 'resnet18',
                                map_location: str = 'cpu',
                                dropout_p: float = 0.5) -> (nn.Module, list):
    """
    Utility to build a model whose final layer size matches the checkpoint.
    Loads checkpoint to detect number of output classes (if `classes` key present
    or by inspecting fc weight shape) and returns (model, classes_list).

    This is handy if you have a checkpoint and want to create the correct
    architecture automatically for NORMAL/BACTERIAL/VIRAL (3 classes) or other counts.

    Args:
        checkpoint_path: path to a .pth/.pt checkpoint containing 'model_state_dict'
                         and optionally 'classes'.
        backbone: backbone name for building the model.
        map_location: device mapping used when loading checkpoint.
        dropout_p: dropout probability for the final head.

    Returns:
        model: model loaded with state_dict from checkpoint (if shapes match).
        classes: list of class names inferred from the checkpoint or a default list.
    """
    ckpt = torch.load(checkpoint_path, map_location=map_location)

    # Prefer explicit classes stored in checkpoint
    classes = ckpt.get('classes', None)

    if classes is None:
        # try to infer from state_dict final layer weight shape
        msd = ckpt.get('model_state_dict', ckpt if isinstance(ckpt, dict) else None)
        out_features = None
        if isinstance(msd, dict):
            # find fc weight key (common names)
            candidate_keys = [k for k in msd.keys() if (k.endswith('.weight') and ('fc' in k or 'classifier' in k or 'head' in k))]
            if candidate_keys:
                # take the first candidate and inspect shape
                w = msd[candidate_keys[0]]
                try:
                    out_features = tuple(w.shape)[0]
                except Exception:
                    out_features = None
        if out_features is None:
            # default to 3 classes (NORMAL, BACTERIAL, VIRAL)
            out_features = 3
        # create generic class names
        if out_features == 3:
            classes = ['NORMAL', 'BACTERIAL', 'VIRAL']
        elif out_features == 2:
            classes = ['NORMAL', 'PNEUMONIC']
        else:
            classes = [f'CLASS_{i}' for i in range(out_features)]

    num_classes = len(classes)
    model = build_model(num_classes=num_classes, backbone=backbone, pretrained=False, dropout_p=dropout_p)

    # load weights if shapes match (otherwise user may need to re-train / fine-tune)
    if 'model_state_dict' in ckpt:
        try:
            model.load_state_dict(ckpt['model_state_dict'])
        except RuntimeError as e:
            # typically arises when checkpoint head size != desired num_classes
            # raise a helpful error
            raise RuntimeError(
                f"Failed to load checkpoint into model (probably mismatch in final layer size): {e}\n"
                f"Checkpoint classes inferred: {classes} (num_classes={num_classes}).\n"
                "If you want to use the checkpoint trained for a different number of classes, "
                "either load it with matching num_classes or fine-tune a new head."
            )

    return model, classes
