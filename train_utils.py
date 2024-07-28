
from pathlib import Path
import os
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from accelerate.utils import ProjectConfiguration, set_seed
import diffusers
from diffusers.utils.torch_utils import is_compiled_module
import wandb
import logging
import math
# import common.loras as loras
# from loras import patch_lora
import sys
sys.path.append('..')
import random
from diffusers import DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
import copy
from tqdm import tqdm

import numpy as np

import pandas as pd
from pipeline import StableDiffusionRLCFGPipeline


def collate_fn(examples):
    text = [example["text"] for example in examples]
    pixel_values = [example["pixel_values"] for example in examples]

    batch = {
        "text": text,
        "pixel_values":  torch.stack(pixel_values),
    }

    return batch



def save_model(unet, accelerator, save_path, args, logger):
    unwrapped = unwrap_model(accelerator, unet)
    state_dict = {"token": unwrapped.reward_projector}
    torch.save(state_dict, save_path)
    logger.info(f"Saved state to {save_path}")



class PandasDataset(Dataset):

    def __init__(self, dataset_path, size, center_crop=True, image_col='filename', synth_prompt_col="re_caption", prompt_col='org_caption'):
        self.df = pd.read_parquet(dataset_path)
        self.df = self.df.dropna(subset=[image_col, prompt_col])
        self.df = self.df.reset_index(drop=True)
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.image_paths = self.df[image_col].tolist()
        self.prompts = self.df[prompt_col].tolist()
        self.synth_prompts = self.df[synth_prompt_col].tolist()

        print("image_paths", len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)
    
    def get_image(self, idx):
        img_path = self.image_paths[idx]
        if not os.path.exists(img_path):
            return None
        img = Image.open(img_path).convert("RGB")
        img = self.image_transforms(img)

        return img


    def __getitem__(self, idx):
        for i in range(500):
            try:
                img = self.get_image(idx)
                if img is None:
                    idx = random.randint(0, len(self.image_paths)-1)
                    continue

                example = {
                    "pixel_values": img,
                    "text": self.prompts[idx],
                    "synth_text": self.synth_prompts[idx]
                }

                return example
            except Exception as e:
                idx = random.randint(0, len(self.image_paths)-1)
                print(e)
                continue
        else:
            raise Exception("Could not load image")




def unwrap_model(accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def log_validation(
    unet,
    text_encoder,
    weight_dtype,
    args,
    accelerator,
    epoch,
    logger,
    is_final_validation=False,
):

    pipeline = StableDiffusionRLCFGPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unwrap_model(accelerator, unet),
        text_encoder=unwrap_model(accelerator, text_encoder),
        torch_dtype=weight_dtype,
    )

    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipeline.safety_checker = None
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    images = []
    with torch.cuda.amp.autocast():
        images.extend(pipeline(prompt=args.validation_prompt, height=args.resolution, width=args.resolution, generator=generator).images)

    for tracker in accelerator.trackers:
        if args.use_wandb:
            phase_name = "test" if is_final_validation else "validation"
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
            if tracker.name == "wandb":
                tracker.log(
                    {
                        phase_name: [
                            wandb.Image(image, caption=f"{i}: {args.validation_prompt[i]}") for i, image in enumerate(images)
                        ]
                    }
                )

    del pipeline
    torch.cuda.empty_cache()

    return images


def init_train_basics(args, logger):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Enable TF32 for faster training on Ampere GPUs,
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    return accelerator, weight_dtype


def load_models(args, accelerator, weight_dtype):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = transformers.CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
    ).requires_grad_(False).to(accelerator.device, dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", 
    ).requires_grad_(False).to(accelerator.device, dtype=weight_dtype)

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet",
    ).requires_grad_(False).to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()


    return tokenizer, noise_scheduler, text_encoder, vae, unet


def get_optimizer(args, params_to_optimize, accelerator):
    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    optimizer_class = torch.optim.AdamW
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_class = bnb.optim.AdamW8bit

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    return optimizer, lr_scheduler


def get_dataset(args, tokenizer):

    # Dataset and DataLoaders creation:
    train_dataset = PandasDataset(
        dataset_path=args.dataset_path,
        size=args.resolution,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    return train_dataset, train_dataloader, num_update_steps_per_epoch


default_arguments = dict(
    pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
    dataset_path="/home/ubuntu/NoveltyGAN/data/combined.parquet",
    num_validation_images=4,
    output_dir="model-output",
    seed=124,
    resolution=512,
    train_batch_size=8,
    max_train_steps=5000,
    validation_steps=250,
    checkpointing_steps=500,
    resume_from_checkpoint=None,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    learning_rate=1.0e-3,
    lr_scheduler="linear",
    lr_warmup_steps=50,
    lr_num_cycles=1,
    lr_power=1.0,
    dataloader_num_workers=4,
    use_8bit_adam=False,
    adam_beta1=0.85,
    adam_beta2=0.98,
    adam_weight_decay=1e-2,
    adam_epsilon=1e-08,
    max_grad_norm=1.0,
    report_to="wandb",
    mixed_precision="bf16",
    allow_tf32=True,
    logging_dir="logs",
    local_rank=-1,
    num_processes=1,
    use_wandb=True,
    num_tokens=16
)


def resume_model(unet, path, accelerator):
    accelerator.print(f"Resuming from checkpoint {path}")
    global_step = int(path.split("-")[-1])
    state_dict = torch.load(path, map_location="cpu")

    unet.reward_emb = torch.nn.Parameter(state_dict["token"].to(accelerator.device))

    return global_step


def more_init(accelerator, args, train_dataloader, train_dataset, logger, num_update_steps_per_epoch, global_step, wandb_name="diffusion_lora"):
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        if args.use_wandb:
            accelerator.init_trackers(wandb_name, config=tracker_config)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    initial_global_step = global_step
    first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    return global_step, first_epoch, progress_bar