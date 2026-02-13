# src/gradcam_visualize.py
import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from src.model import build_model
from src.utils import load_checkpoint

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        self.model.zero_grad()
        out = self.model(input_tensor)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        score = out[0, class_idx]
        score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

def preprocess(img_path, image_size=224):
    transform = transforms.Compose([
        transforms.Resize(int(image_size*1.1)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    return img, transform(img).unsqueeze(0)

def overlay_cam(img_pil, cam_mask, alpha=0.4):
    img = np.array(img_pil)  # RGB
    h, w, _ = img.shape
    mask = cv2.resize(cam_mask, (w, h))
    heat = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = img * (1 - alpha) + heat * alpha
    overlay = np.uint8(np.clip(overlay, 0, 255))
    return overlay

def main(image_path,
         model_path='/mnt/data/c4b04791-9abe-460e-a13d-35a44845f934.pth',
         out_path='/kaggle/working/results/gradcam.png',
         image_size=224):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(model_path, map_location=device)
    classes = ckpt.get('classes', ['NORMAL','PNEUMONIC'])
    model = build_model(num_classes=len(classes), backbone='resnet18', pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    try:
        target_layer = model.layer4[-1].conv2
    except Exception:
        target_layer = None
        for m in reversed(list(model.modules())):
            import torch.nn as nn
            if isinstance(m, nn.Conv2d):
                target_layer = m
                break
    gradcam = GradCAM(model, target_layer)
    orig_img, tensor = preprocess(image_path, image_size=image_size)
    tensor = tensor.to(device)
    cam_mask, pred_idx = gradcam(tensor)
    overlayed = overlay_cam(orig_img, cam_mask)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(overlayed).save(out_path)
    print(f"Saved Grad-CAM to {out_path}; predicted: {classes[pred_idx]}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--model', default='/mnt/data/c4b04791-9abe-460e-a13d-35a44845f934.pth')
    parser.add_argument('--out', default='/kaggle/working/results/gradcam.png')
    parser.add_argument('--image-size', type=int, default=224)
    args = parser.parse_args()
    main(args.image, model_path=args.model, out_path=args.out, image_size=args.image_size)
