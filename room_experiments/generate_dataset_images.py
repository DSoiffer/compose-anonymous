"""Generate the raw room images for each prompt in training_prompts.yaml with FLUX.1-schnell.

Images are written to <output_root>/<category>/<item name>/image_<index>.png.
"""
import argparse
import os

import torch
import yaml
from diffusers import DiffusionPipeline
from sd_embed.embedding_funcs import get_weighted_text_embeddings_flux1
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda"
BATCH_SIZE = 20


def get_next_index(folder):
    """Return max existing image index + 1, or 1 if the folder is empty/absent."""
    if not os.path.isdir(folder):
        return 1
    indices = []
    for f in os.listdir(folder):
        if f.startswith("image_") and f.endswith(".png"):
            try:
                indices.append(int(f[6:-4]))
            except ValueError:
                pass
    return max(indices) + 1 if indices else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pretrained FLUX.1-schnell model.")
    parser.add_argument("--output_root", type=str, default="generated_images",
                        help="Root folder for generated images, with subfolders per category and item.")
    parser.add_argument("--only", nargs="+", default=None,
                        help="If set, generate only these item names (exact keys from the YAML). "
                             "Quote multi-word names, e.g. --only 'Couch' 'Couch+Framed painting'")
    parser.add_argument("--n", type=int, default=2000, help="Images per prompt")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for image generation")
    parser.add_argument("--append", action="store_true",
                        help="Start image indices after the highest existing index in each folder "
                             "instead of overwriting from 1.")
    args = parser.parse_args()

    with open(os.path.join(SCRIPT_DIR, "training_prompts.yaml"), "r") as f:
        prompts = yaml.safe_load(f)

    if args.only is not None:
        prompts = {k: v for k, v in prompts.items() if k in args.only}
        print(f"Filtered to {len(prompts)} item(s): {list(prompts.keys())}")

    generator = torch.Generator(device=DEVICE).manual_seed(args.seed)

    pipe = DiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )
    print("Moving to GPU...")
    pipe = pipe.to(DEVICE)

    # Prepare jobs
    jobs = []
    for item_name, attrs in tqdm(prompts.items(), desc="Preparing jobs"):
        category = attrs["category"]
        prompt = attrs["prompt"]
        folder = os.path.join(args.output_root, category, item_name.replace(" ", "_").lower())
        start = get_next_index(folder) if args.append else 1
        if args.append:
            print(f"  {item_name}: starting from index {start}")
        prompt_embeds, pooled_prompt_embeds = get_weighted_text_embeddings_flux1(pipe, prompt=prompt)
        for i in range(args.n):
            jobs.append((item_name, category, prompt_embeds, pooled_prompt_embeds, start + i))

    # Batched inference
    for i in tqdm(range(0, len(jobs), BATCH_SIZE), desc="Generating images"):
        batch = jobs[i:i + BATCH_SIZE]
        batch_prompt_embeds = torch.cat([e for (_, _, e, _, _) in batch], dim=0)
        batch_pooled_prompt_embeds = torch.cat([pe for (_, _, _, pe, _) in batch], dim=0)

        images = pipe(
            prompt_embeds=batch_prompt_embeds,
            pooled_prompt_embeds=batch_pooled_prompt_embeds,
            num_inference_steps=6,
            width=256,
            height=256,
            guidance_scale=7.5,
            generator=generator,
        ).images

        for (item_name, category, _, _, img_index), image in zip(batch, images):
            folder = os.path.join(args.output_root, category, item_name.replace(" ", "_").lower())
            os.makedirs(folder, exist_ok=True)
            image.save(os.path.join(folder, f"image_{img_index}.png"))


if __name__ == "__main__":
    main()
