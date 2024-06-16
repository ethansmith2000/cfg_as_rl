#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import copy
import math
import os
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate.logging import get_logger
from tqdm.auto import tqdm

import train_utils
from train_utils import (
    collate_fn,
    init_train_basics,
    log_validation,
    save_model,
    unwrap_model,
    default_arguments,
    load_models,
    get_optimizer,
    get_dataset,
    more_init,
    resume_model,
)
from types import SimpleNamespace

def train(args):
    logger = get_logger(__name__)
    args = SimpleNamespace(**args)
    args.validation_prompt = [f"majestic fantasy painting", f"a comic book drawing", f"HD cinematic photo", f"oil painting"]
    accelerator, weight_dtype = init_train_basics(args, logger)

    tokenizer, noise_scheduler, text_encoder, vae, unet = load_models(args, accelerator, weight_dtype)

    # Optimizer creation
    optimizer, lr_scheduler = get_optimizer(args, [unet.special_token], accelerator)
    train_dataset, train_dataloader, num_update_steps_per_epoch = get_dataset(args, tokenizer)

    # Prepare everything with our `accelerator`.
    unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    if args.resume_from_checkpoint:
        global_step = resume_model(unet, args.resume_from_checkpoint, accelerator)

    global_step, first_epoch, progress_bar = more_init(accelerator, args, train_dataloader, 
                                                        train_dataset, logger, num_update_steps_per_epoch, global_step, wandb_name="cfg_as_rl")

    grad_norm = 0
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                with torch.no_grad():
                    model_input = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                    noise = torch.randn_like(model_input)
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (noise.shape[0],), device=model_input.device
                    ).long()
                    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                    # Get the text embedding for conditioning
                    input_ids = tokenizer(batch["text"], return_tensors="pt", padding="max_length", truncation=True, max_length=77).input_ids
                    encoder_hidden_states = text_encoder(input_ids.to(text_encoder.device),return_dict=False,)[0]

                encoder_hidden_states = torch.cat([encoder_hidden_states, unet.special_token.expand(encoder_hidden_states.shape[0], -1, -1)], dim=1)

                # Predict the noise residual
                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    encoder_hidden_states,
                    return_dict=False,
                )[0]

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(unet.special_token, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if step % 25 == 0:
                    print(unet.special_token)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        save_model(unet,accelerator,save_path, args, logger)
                        

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "grad_norm": grad_norm}
            progress_bar.set_postfix(**logs)
            if args.use_wandb:
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process:
                if args.validation_prompt is not None and global_step % args.validation_steps == 0 and global_step > 0:
                    images = log_validation(
                        unet,
                        text_encoder,
                        weight_dtype,
                        args,   
                        accelerator,
                        epoch=epoch,
                        logger=logger,
                    )

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        save_model(unet, accelerator, save_path, args, logger)

    accelerator.end_training()


if __name__ == "__main__":
    train(default_arguments)