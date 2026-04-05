from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def _get_last_conv_layer(model: nn.Module) -> nn.Module:
    target_layer = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            target_layer = module
    if target_layer is None:
        raise ValueError("Could not locate a convolution layer for Grad-CAM")
    return target_layer


def _encode_overlay(image: Image.Image, cam: np.ndarray) -> str:
    heatmap = Image.fromarray(np.uint8(cam * 255.0), mode="L").resize(image.size, Image.BILINEAR)
    heatmap_np = np.asarray(heatmap, dtype=np.float32) / 255.0

    base_np = np.asarray(image.convert("RGB"), dtype=np.float32)
    color_map = np.stack(
        [
            heatmap_np,
            np.sqrt(heatmap_np),
            np.clip(heatmap_np - 0.25, 0.0, 1.0),
        ],
        axis=-1,
    )
    overlay = np.clip((0.58 * base_np) + (0.42 * color_map * 255.0), 0, 255).astype(np.uint8)

    buffer = BytesIO()
    Image.fromarray(overlay).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_gradcam_overlay(
    image: Image.Image,
    model: nn.Module,
    tensor: torch.Tensor,
    target_index: int,
    model_kwargs: dict | None = None,
) -> str:
    target_layer = _get_last_conv_layer(model)
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}
    model_kwargs = model_kwargs or {}

    def forward_hook(_, __, output):
        activations["value"] = output.detach()

    def backward_hook(_, __, grad_output):
        gradients["value"] = grad_output[0].detach()

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        outputs = model(tensor, **model_kwargs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        score = logits[:, target_index].sum()
        score.backward()

        acts = activations["value"]
        grads = gradients["value"]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1)).squeeze(0)
        cam -= cam.min()
        cam /= cam.max().clamp(min=1e-8)
        cam_np = cam.detach().cpu().numpy()
    finally:
        handle_forward.remove()
        handle_backward.remove()
        model.zero_grad(set_to_none=True)

    return _encode_overlay(image, cam_np)
