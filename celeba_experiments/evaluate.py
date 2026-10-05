"""Score generated CelebA compositions with the FaceXFormer attribute head.

FaceXFormer outputs a sigmoid probability for each of the 40 CelebA
attributes. An attribute counts as present when its probability is at least
0.5. For each sample file this prints the joint success rate (both attributes
of the pair present) and the two marginal rates.

The FaceXFormer network code is copied into facexformer/ (MIT License). The
released checkpoint is downloaded from the Hugging Face hub
(kartiknarayan/facexformer, ckpts/model.pt).

    python -m celeba_experiments.evaluate --pair FC samples/FC_ID_naive.pt samples/FC_ID_fkc16.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch.nn import functional as F

from celeba_experiments.data import ATTRIBUTES, PAIRS
from celeba_experiments.facexformer.model import FaceXFormer

FACEXFORMER_REPO_ID = "kartiknarayan/facexformer"
FACEXFORMER_FILENAME = "ckpts/model.pt"
THRESHOLD = 0.5


def load_facexformer(device: torch.device) -> FaceXFormer:
    """The FaceXFormer network with its released checkpoint."""
    checkpoint = hf_hub_download(repo_id=FACEXFORMER_REPO_ID, filename=FACEXFORMER_FILENAME)
    model = FaceXFormer()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict_backbone"], strict=True)
    return model.eval().to(device)


@torch.inference_mode()
def facexformer_probabilities(
    model: torch.nn.Module, images: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Standard-order CelebA attribute probabilities from BCHW uint8 images.

    The images are already aligned face crops, so they are resized directly
    to 224x224 (bicubic) and ImageNet-normalized, without a face-detection crop.
    """
    values = images.to(device=device, dtype=torch.float32) / 255
    values = F.interpolate(values, size=(224, 224), mode="bicubic",
                           align_corners=False, antialias=True)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device)[None, :, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), device=device)[None, :, None, None]
    values = (values - mean) / std
    tasks = torch.full((len(values),), 3, dtype=torch.long, device=device)
    attribute_logits = model(values, None, tasks)[2]
    return attribute_logits.sigmoid()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples", type=Path, nargs="+",
                        help="sample.py or sample_ldm.py output files")
    parser.add_argument("--pair", choices=tuple(PAIRS), required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)

    a, b = PAIRS[args.pair]
    a_index, b_index = ATTRIBUTES.index(a), ATTRIBUTES.index(b)
    model = load_facexformer(device)

    print(f"| Samples | Joint | {a} | {b} |")
    print("|---|---:|---:|---:|")
    for path in args.samples:
        images = torch.load(path, map_location="cpu", weights_only=False)["images"]
        probabilities = torch.cat([
            facexformer_probabilities(model, images[start:start + args.batch_size], device).cpu()
            for start in range(0, len(images), args.batch_size)
        ])
        has_a = probabilities[:, a_index] >= THRESHOLD
        has_b = probabilities[:, b_index] >= THRESHOLD
        joint = float((has_a & has_b).float().mean())
        print(f"| {path.name} | {joint:.3f} | {float(has_a.float().mean()):.3f} | "
              f"{float(has_b.float().mean()):.3f} |")


if __name__ == "__main__":
    main()
