# app.py
import os
import glob
import cv2
import tempfile
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.model import build_model

# Defaults
DEFAULT_MODEL_PATH = "./model.pth"
DEFAULT_TEST_GLOB = "/kaggle/input/chest-x-ray/chest_xray/test/**/*.*"
IMAGE_SIZE = 224

st.set_page_config(page_title="Chest X-ray Multi-class Classifier", layout="centered")

@st.cache_resource
def load_model_and_classes(model_path: str, device: str):
    """
    Load checkpoint and build model. Handles a few checkpoint shapes:
      - checkpoint may contain 'model_state_dict' or 'state_dict'
      - checkpoint may contain 'classes' (list). If not present, tries to infer
        number of outputs from final linear layer weights.
      - strips 'module.' prefixes if present.
    Returns (model, classes) or (None, None) on failure.
    """
    if not os.path.exists(model_path):
        return None, None

    # load checkpoint safely to device
    try:
        ckpt = torch.load(model_path, map_location=device)
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        return None, None

    # find state_dict in many common shapes
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            # maybe the ckpt itself is a state_dict (no wrapper)
            # detect by checking for tensor-like values
            if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                state = ckpt
            else:
                # unknown format
                state = {}
    else:
        state = {}

    # classes: try to read from checkpoint top-level 'classes' or 'label_names'
    classes = None
    if isinstance(ckpt, dict):
        classes = ckpt.get('classes', None) or ckpt.get('label_names', None)

    # If classes not present, try to infer from final linear weight size
    if classes is None:
        out_features = None
        # try to find final linear in state dict
        for k, v in state.items():
            if k.endswith('.weight') and ('fc' in k or 'classifier' in k or 'head' in k):
                out_features = v.shape[0]
                break
        if out_features is not None:
            if out_features == 3:
                classes = ['NORMAL', 'BACTERIAL', 'VIRAL']  # common mapping (user-trained ordering may differ)
            elif out_features == 2:
                classes = ['NORMAL', 'PNEUMONIC']
            else:
                classes = [f"CLASS_{i}" for i in range(out_features)]
        else:
            # fallback default
            classes = ['NORMAL', 'BACTERIAL', 'VIRAL']

    num_classes = len(classes)

    # build model and load weights
    try:
        model = build_model(num_classes=num_classes, backbone='resnet18', pretrained=False)
    except Exception as e:
        st.error(f"Failed to build model: {e}")
        return None, None

    # handle DataParallel 'module.' prefix in state dict keys
    if state:
        new_state = {}
        for k, v in state.items():
            new_k = k.replace('module.', '') if k.startswith('module.') else k
            new_state[new_k] = v
        try:
            model.load_state_dict(new_state, strict=False)
        except Exception as e:
            # try direct load (some checkpoints store entire model object)
            try:
                model.load_state_dict(state, strict=False)
            except Exception as e2:
                st.warning(f"Could not strictly load state_dict: {e2}. Attempting non-strict load.")
                # still proceed; model may partially load
    else:
        # if no state dict present, maybe the checkpoint is a whole model object (rare)
        if isinstance(ckpt, torch.nn.Module):
            model = ckpt
        else:
            st.warning("No state_dict found in checkpoint; model will be uninitialized.")

    model.to(device)
    model.eval()
    return model, classes


def preprocess_image(pil_img, image_size=IMAGE_SIZE):
    tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),  # slightly larger then center crop (common training pattern)
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf(pil_img).unsqueeze(0)


def predict(model, tensor, device):
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.activations = None
        self.gradients = None

        # attach forward hook
        target_layer.register_forward_hook(self._forward)

        # attach backward hook: prefer register_full_backward_hook if available (PyTorch >1.8)
        if hasattr(target_layer, "register_full_backward_hook"):
            target_layer.register_full_backward_hook(self._backward)
        else:
            target_layer.register_backward_hook(self._backward)  # deprecated but fallback

    def _forward(self, module, inp, out):
        # out can be tuple or tensor
        if isinstance(out, tuple):
            out = out[0]
        self.activations = out.detach()

    def _backward(self, module, grad_in, grad_out):
        # grad_out may be tuple
        go = grad_out[0] if isinstance(grad_out, tuple) else grad_out
        self.gradients = go.detach()

    def __call__(self, input_tensor, class_idx=None):
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        # zero grads
        self.model.zero_grad()
        out = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(out.argmax(dim=1).item())
        score = out[0, class_idx]
        score.backward(retain_graph=True)

        # global-average-pool grads
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # shape (N, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        # normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def overlay_on_pil(pil_img, cam_mask, alpha=0.4):
    img = np.array(pil_img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    h, w, _ = img.shape
    mask = cv2.resize(cam_mask, (w, h))
    heat = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    over = img * (1 - alpha) + heat * alpha
    return np.uint8(np.clip(over, 0, 255))


# --- UI ---
st.title("Chest X-ray Classifier — NORMAL / BACTERIAL / VIRAL")
st.write("This app supports model checkpoints with 2 or 3 output classes. For 3-class prediction the model must be trained to output [NORMAL, BACTERIAL, VIRAL] in the same order as your checkpoint (or include a `classes` list).")

with st.sidebar:
    model_path = st.text_input("Model path (local)", DEFAULT_MODEL_PATH)
    uploaded_ckpt = st.file_uploader("Or upload model checkpoint (.pth/.pt)", type=["pth", "pt"])
    use_gpu = st.checkbox("Use GPU if available", value=True)
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    st.write("Device:", device)
    image_size = st.slider("Image size", min_value=128, max_value=512, value=IMAGE_SIZE, step=32)

# Determine checkpoint source
ckpt_source = None
if uploaded_ckpt is not None:
    # write upload to a stable temp path
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, uploaded_ckpt.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_ckpt.read())
    ckpt_source = tmp_path
else:
    ckpt_source = model_path if model_path.strip() != "" else DEFAULT_MODEL_PATH

# Load model
with st.spinner("Loading model..."):
    model, classes = load_model_and_classes(ckpt_source, device)

if model is None:
    st.error(f"Model not found or failed to load at {ckpt_source}. Place checkpoint and reload, or upload via the sidebar.")
    st.stop()

# Adjust UI based on number of classes
num_classes = len(classes)
if num_classes == 3:
    lower = [c.lower() for c in classes]
    if ('normal' not in lower) or (('bacter' not in "".join(lower)) and ('viral' not in "".join(lower))):
        st.info(f"Model classes detected: {classes}. App will display them as-is. If you trained for [NORMAL, BACTERIAL, VIRAL], ensure checkpoint classes match that order.")
else:
    st.warning(f"Model has {num_classes} classes: {classes}. For NORMAL/BACTERIAL/VIRAL predictions you need a 3-class model trained to output those labels. This app will still show the model's classes.")

# Top-K slider (capped by num_classes)
top_k = st.sidebar.slider("Top K", min_value=1, max_value=min(5, num_classes), value=min(3, num_classes))

col1, col2 = st.columns([2, 1])
with col2:
    uploaded = st.file_uploader("Upload chest X-ray image", type=["png", "jpg", "jpeg"], key="img_upload")
    st.markdown("Or select a sample image from your test folder:")
    if st.checkbox("Show test list"):
        glob_path = st.text_input("Test glob (supports glob patterns)", value=DEFAULT_TEST_GLOB)
        files = glob.glob(glob_path, recursive=True)
        images = [p for p in files if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            choice = st.selectbox("Choose image", images, format_func=lambda p: os.path.basename(p))
        else:
            st.write("No images found at that glob.")
            choice = None

with col1:
    pil_img = None
    if uploaded:
        try:
            pil_img = Image.open(uploaded).convert('RGB')
            st.image(pil_img, caption="Uploaded image", use_column_width=True)
        except Exception as e:
            st.error(f"Failed to open uploaded image: {e}")
            st.stop()
    elif 'choice' in locals() and choice:
        try:
            pil_img = Image.open(choice).convert('RGB')
            st.image(pil_img, caption=choice, use_column_width=True)
        except Exception as e:
            st.error(f"Failed to open selected image: {e}")
            st.stop()
    else:
        st.info("Please upload an image or select one from the test set to continue.")

if pil_img is None:
    st.stop()

if st.button("Predict"):
    tensor = preprocess_image(pil_img, image_size=image_size)
    probs = predict(model, tensor, device)
    top_idxs = probs.argsort()[::-1][:top_k]
    st.subheader("Top predictions")
    for rank, i in enumerate(top_idxs):
        st.write(f"{rank + 1}. **{classes[i]}** — {probs[i] * 100:.2f}%")

    pred_idx = int(probs.argmax())
    st.success(f"Final prediction: **{classes[pred_idx]}** ({probs[pred_idx] * 100:.2f}%)")

    if num_classes == 2:
        st.warning("Your loaded model outputs 2 classes (likely NORMAL vs PNEUMONIC). To predict NORMAL/BACTERIAL/VIRAL you must load a 3-class model trained accordingly.")

    # Grad-CAM
    if st.checkbox("Show Grad-CAM overlay", value=True):
        # Try to locate a sensible last conv layer for ResNet variants
        target_layer = None
        try:
            # common for ResNet-18/34
            target_layer = model.layer4[-1].conv2
        except Exception:
            import torch.nn as nn
            # fallback: find last Conv2d in model
            for m in reversed(list(model.modules())):
                if isinstance(m, nn.Conv2d):
                    target_layer = m
                    break

        if target_layer is None:
            st.error("Could not find a convolutional layer for Grad-CAM in this model.")
        else:
            try:
                gradcam = GradCAM(model, target_layer)
                cam_mask, used_idx = gradcam(tensor, class_idx=pred_idx)
                overlay = overlay_on_pil(pil_img, cam_mask, alpha=0.45)
                st.image(overlay, caption=f"Grad-CAM overlay — predicted: {classes[pred_idx]}", use_column_width=True)
            except Exception as e:
                st.error(f"Grad-CAM generation failed: {e}")

st.caption(f"Loaded model classes: {classes} | Model path: {ckpt_source}")
