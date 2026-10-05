# Room Experiments

This directory contains code for replicating the dataset and experimental results for the room experiments: the dataset, pixel-space and latent diffusion models, composition sampling with FKC, and the VLM judge used to score the generated images.


## Running
Before running, ensure you have installed all dependencies with `uv sync` from the root directory. `uv` can be installed [here](https://docs.astral.sh/uv/getting-started/installation/).

In order to generate images for the dataset, you will also need to download the [FLUX.1-schnell model](https://huggingface.co/black-forest-labs/FLUX.1-schnell).

Run all commands below from the repository root (the parent of this directory).

Each step needs the output of the steps before it:

| Step | Needs |
|---|---|
| Creating the Dataset | FLUX.1-schnell, and a set of hand-labeled images per class |
| Training the diffusion model (pixel space) | The dataset |
| Latent diffusion models | The dataset |
| Generating composition images | Trained models (pixel-space, latent, or separate per-condition models) |
| Judging the images | Generated composition images and an OpenAI API key |


## Creating the Dataset
Creating the dataset is broken down into several steps.

### 1. Create training prompts

```
python -m room_experiments.create_prompts
```
creates prompts from the `perturbations.yaml` file for an empty (`control`) room, a room with only one specified object in it, or a room with exactly one each of two specified objects.


### 2. Generate raw images per prompt
Now use these prompts to generate `n` images per class with a text-to-image model. This requires you to supply the path to your downloaded text-to-image model (in this case, FLUX-schnell).

```
python -m room_experiments.generate_dataset_images \
  --model_path /path/to/FLUX.1-schnell \
  --output_root /path/to/gen_images \
  --n 2000 \
  --batch_size 20
```

Images are written to `<output-root>/<category>/<item_name_lower>/image_<idx>.png` (so single-object classes land under e.g. `<output-root>/furniture/couch/` and pairs under `<output-root>/two_objects/couch+framed_painting/`).

The `--only` argument restricts image generation to a subset of the prompts, e.g. `--only "Couch" "Couch+Framed painting"`. `--append` continues numbering past the highest existing index instead of overwriting from `image_1.png`.


### 3. Label a subset of the images
For each class, label some of the images manually as either accept/reject, based on whether or not they satisfy the prompt. This is to ensure that rooms contain ONLY the specified objects. Mirroring the generated dataset layout, place the labelled images in a new directory, with the structure `<dir>/accept/<classes>/<images_for_class>` and `<dir>/reject/<classes>/<images_for_class>`. The unlabeled images will be fed through classifiers trained on the labeled images.

Experimenting with 1000 labeled images per class and withholding a validation set, we find that it is helpful to label at least 200 images per class, and that increasing the number of labeled images beyond this point has diminishing returns on classifier accuracy. Hence, we suggest manually labeling at least 200 images per class.

### 4. Train classifiers, build the per-class dataset

To train the classifiers to label the remaining images and construct the dataset, run

```
python -m room_experiments.build_dataset \
  --labeled_dir /path/to/labeled \
  --gen_dir /path/to/gen_images \
  --out_dir /path/to/dataset_out \
  --classifier_dir /path/to/classifiers
```

where `out_dir` is where you would like the dataset to be written to, and `classifier_dir` is where you would like classifier checkpoints to be saved to. (Classifier checkpoints can be safely deleted afterwards if you do not want to keep them.)

For each class in `--classes` (defaults to
`control coffee_table couch framed_painting couch+coffee_table
couch+framed_painting`), the script:

  1. Embeds `--labeled_dir/{accept,reject}/<class>/*.png` with DINOv2
     and fits a per-class logistic-regression accept/reject classifier
     (saved to `--classifier_dir/<class>.pkl`).
  2. Copies the hand-labeled accepts into `--out_dir/<class>/`.
  3. Runs the classifier on the remaining unlabeled images located at
     `--gen_dir/<category>/<class>/*.png` (the layout written in step 2),
     and copies the classifier-accepted images into `--out_dir/<class>/`.


Be careful when generating the dataset that accept/reject image file names do not overlap with the files to be classified (within the same class), as this can cause files to be overwritten. This should not occur if you do not rename any generated image files.



## Training the diffusion model

To run, there are two primary options: single conditional model (the main mode), or separate models for each condition. You can run using conditions and data mixtures defined in `room_experiments/train_configs`. For example, to train a single conditional model on the Factorized Conditional In-Distribution conditional mixture as described in the paper:
```
python -m room_experiments.train \
  --data_dir /path/to/dataset_out \
  --output_dir /path/to/checkpoints \
  --conditions room_experiments/train_configs/FC_ID.yaml
```

To train separate models for each condition, run the same command once per condition, with ``--conditions`` set to each of ``room_experiments/train_configs/FC_ID_sep_base.yaml``, ``room_experiments/train_configs/FC_ID_sep_couch.yaml``, and ``room_experiments/train_configs/FC_ID_sep_framed_painting.yaml``. The other settings use ``FC_OOD_sep_*``, ``NFC_ID_sep_*``, and ``NFC_OOD_sep_*``.


Each checkpoint directory contains the base weights, an `ema_model/`
copy, the saved scheduler, `normalize.json`, `classes.json` (the
condition names), and `conditions.yaml`.

It is also recommended that you run training with `accelerate launch` instead of `python` to speed up training or to utilize multiple GPUs, e.g.
```
accelerate launch --mixed_precision bf16 --multi_gpu --num_processes <number of GPUs> -m room_experiments.train ...
```
(omit `--multi_gpu` if training on a single GPU).



## Latent diffusion models

**Requires:** the dataset from "Creating the Dataset". Train the autoencoder first, as the latent diffusion model needs it.

Each of the four settings (`FC_ID`, `FC_OOD`, `NFC_ID`, `NFC_OOD`) has its own autoencoder and latent diffusion model, trained on the same training config.

### 1. Train the autoencoder

```
accelerate launch --mixed_precision bf16 --multi_gpu --num_processes 2 -m room_experiments.train_vae \
  --data_dir /path/to/dataset_out \
  --output_dir /path/to/checkpoints/vae_FC_ID \
  --conditions room_experiments/train_configs/FC_ID.yaml \
  --num_epochs 50 --batch_size 16 --compile \
  --latent_channels 2
```

### 2. Train the latent diffusion model

```
accelerate launch --mixed_precision bf16 --multi_gpu --num_processes 2 -m room_experiments.train_ldm \
  --data_dir /path/to/dataset_out \
  --vae /path/to/checkpoints/vae_FC_ID/checkpoint-epoch50 \
  --output_dir /path/to/checkpoints/ldm_FC_ID \
  --conditions room_experiments/train_configs/FC_ID.yaml \
  --num_epochs 50 --batch_size 16
```

The checkpoint's `vae.json` records the autoencoder path, so keep the autoencoder checkpoint where it was trained. The CelebA latent models use the same two scripts (see `celeba_experiments/README.md`).


## Generating composition images for evaluation

**Requires:** a trained model: a pixel-space checkpoint from "Training the diffusion model", a latent diffusion checkpoint from "Latent diffusion models", or one pixel-space checkpoint per condition (separate models).

`generate_compositions.py` writes one PNG per image it generates. `--n_particles 1` runs naive composition and `--n_particles K` runs FKC with K particles. An interrupted run can be restarted with the same command.

The compositions use `--classes couch framed_painting base` for the FC settings and `--classes couch coffee_table base` for the NFC settings, with `--betas 1 1 -1`. For example, naive composition for FC + ID with a pixel-space model:

```
python -m room_experiments.generate_compositions \
  --checkpoint /path/to/checkpoints/FC_ID/checkpoint-epoch50 \
  --classes couch framed_painting base --betas 1 1 -1 \
  --n_particles 1 --n_output 500 --batch_size 50 --seed 0 --bf16 \
  --out_dir /path/to/images/FC_ID/K1
```

- FKC: set `--n_particles 16` and a smaller `--batch_size`, since the model batch is the batch size times the number of particles.
- Latent diffusion models: pass the latent checkpoint and drop `--bf16`.
- Stopping FKC after 90 of the 100 steps: add `--fkc_max_step 90`.
- Separate models: pass the three checkpoints in the order of `--betas` (couch, second object, base) and drop `--classes`.


## Evaluating the images with a VLM judge

**Requires:** composition images for all four settings, in the layout `<root>/<FC_ID|FC_OOD|NFC_ID|NFC_OOD>/<K1|K16>/` that the commands above write, and an OpenAI API key set in the `OPENAI_API_KEY` environment variable. Pass other sampler directory names with `--samplers`.

`judge_room_images.py` asks an OpenAI vision model (`gpt-5.6-luna`) whether a couch and the second object are present in each image and for a 0 to 3 warping score. Each request includes the labeled calibration images in `judge_calibration/`.

```
python -m room_experiments.judge_room_images \
  --input_root /path/to/images \
  --output_dir /path/to/judge_results \
  --concurrency 5
```

Per-image results go to `judgments.jsonl` in the output directory. `summary.json` and the printed table give, for each setting and sampler, the joint presence rate (both objects present), the per-object presence rates, and the mean warping score.
