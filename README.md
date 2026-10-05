# Code for When is Compositional Generation Feasible? Distributional Estimation Error and Inference-time Approximation Error in Diffusion Models

This repository contains the code for the experiments in the paper. Each directory has its own README with instructions, please see thoes for further instructions.

| Directory | Contents | Requires |
|---|---|---|
| `gaussian_experiments` | 2D Gaussian and Gaussian mixture experiments: training, FKC sampling sweeps, and the out-of-distribution support table. | Nothing. Runs on its own. |
| `room_experiments` | Room image experiments: dataset construction, pixel-space and latent diffusion models, composition sampling, and the VLM judge. | FLUX.1-schnell and hand-labeled images to build the dataset. Later steps use the dataset and models built here. |
| `celeba_experiments` | CelebA experiments: pixel-space and latent diffusion models, composition sampling, and FaceXFormer evaluation. | Nothing. The dataset downloads automatically. |
| `factorization` | The mean-effect cosine heuristic for how factorized the room and CelebA conditions are. | The room dataset and trained autoencoders from `room_experiments` and `celeba_experiments`. |
| `sensitivity_metric` | The rESS sensitivity metric: Gaussian families with their two figures, and density-ratio estimation with latent diffusion models. | `gaussian/`: nothing. `image/`: trained latent diffusion models from `room_experiments` and `celeba_experiments`, and held-out room images. |

Each README starts with what must be run before its code.

## Setup

Install the dependencies with `uv sync` from this directory. `uv` can be installed [here](https://docs.astral.sh/uv/getting-started/installation/).

Depending on your GPU's CUDA compatibility, you may need to change all instances of `cu130` in `pyproject.toml` to an earlier version (for example, `cu129` for CUDA 12.9), and rerun `uv sync`. Gaussian experiments can also run on the CPU, but more slowly.

Models and datasets are downloaded from the Hugging Face hub: FLUX.1-schnell (room dataset), DINOv2-large (room dataset filtering and factorization), CLIP ViT-B/32 (factorization), the CelebA dataset `flwrlabs/celeba`, and the FaceXFormer checkpoint. The room judge uses the OpenAI API and reads the `OPENAI_API_KEY` environment variable, which you must set to your own OpenAI API key.


## Running scripts

Each directory is a Python package. Run every script as a module from this directory, which is the root of the repository. For example:

```
uv run python -m room_experiments.generate_compositions --help
```

All commands in the READMEs are written this way, with paths relative to this directory. 

