"""Image features for the mean-effect cosine heuristic.

    CLIP     openai/clip-vit-base-patch32 image embeddings, L2-normalized
    DINOv2   facebook/dinov2-large CLS token
    VAE      posterior mean of an AutoencoderKL, flattened

All functions take uint8 images of shape (N, 3, H, W). CLIP and DINOv2 run
their Hugging Face image processors on the images and use bf16 autocast. The
VAE maps pixels to [-1, 1] and runs in float32.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

CLIP_MODEL = "openai/clip-vit-base-patch32"
DINO_MODEL = "facebook/dinov2-large"


def _pil(images: np.ndarray) -> list[Image.Image]:
    return [Image.fromarray(np.ascontiguousarray(x.transpose(1, 2, 0))) for x in images]


class Encoder:
    """A frozen feature encoder: call it on (N, 3, H, W) uint8 images."""

    def __init__(self, name: str, *, device, vae_path: str | None = None):
        self.device = device
        if name == "CLIP":
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
            self.processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL)
            model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL)
            self.fn = lambda px: torch.nn.functional.normalize(
                model(pixel_values=px).image_embeds.float(), dim=-1)
        elif name == "DINOv2":
            from transformers import AutoImageProcessor, AutoModel
            self.processor = AutoImageProcessor.from_pretrained(DINO_MODEL)
            model = AutoModel.from_pretrained(DINO_MODEL)
            self.fn = lambda px: model(pixel_values=px).last_hidden_state.float()[:, 0]
        elif name == "VAE":
            from diffusers import AutoencoderKL
            self.processor = None
            model = AutoencoderKL.from_pretrained(vae_path)
            self.fn = lambda x: model.encode(
                x.float().div(127.5).sub(1.0)).latent_dist.mode().float().flatten(1)
        else:
            raise ValueError(f"unknown encoder {name!r}")
        self.model = model.to(device).eval()

    @torch.inference_mode()
    def __call__(self, images: np.ndarray, batch_size: int) -> np.ndarray:
        """(N, 3, H, W) uint8 images -> (N, D) float32 features."""
        features = []
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            if self.processor is not None:
                px = self.processor(images=_pil(batch), return_tensors="pt")["pixel_values"]
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=self.device.type == "cuda"):
                    f = self.fn(px.to(self.device))
            else:
                f = self.fn(torch.from_numpy(np.asarray(batch)).to(self.device))
            features.append(f.cpu().numpy().astype(np.float32))
        return np.concatenate(features)


def abs_cosine(delta_1: np.ndarray, delta_2: np.ndarray) -> float:
    """Unsigned cosine between two mean-effect vectors."""
    return abs(float(np.dot(delta_1, delta_2)
                     / (np.linalg.norm(delta_1) * np.linalg.norm(delta_2))))
