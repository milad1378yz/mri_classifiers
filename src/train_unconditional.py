import argparse
from dataclasses import dataclass
import os
import torch
from torchvision import transforms
from datasets import load_dataset
from diffusers import (
    UNet2DModel,
    DDPMScheduler,
    get_cosine_schedule_with_warmup,
    DDPMPipeline,
)
from torch.utils.data import DataLoader
import torch.nn.functional as F
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
from PIL import Image


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--class_MRI", type=str, default="MildDemented", help="MRI class"
    )
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    return parser.parse_args()


def load_and_transform_dataset(class_MRI, image_size):
    dataset = load_dataset(
        "./",
        data_dir=f"./data/Alzheimer_MRI_4_classes_dataset/{class_MRI}",
        split="train",
    )

    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("L")) for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)
    return dataset


@dataclass
class TrainingConfig:
    class_MRI: str
    num_epochs: int
    image_size: int = 128
    train_batch_size: int = 16
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 10
    save_model_epochs: int = 30
    mixed_precision: str = "fp16"
    output_dir: str = f"../diffiusion_models/{class_MRI}"
    push_to_hub: bool = False
    overwrite_output_dir: bool = True
    seed: int = 0


def create_model_and_optimizer(config):
    model = UNet2DModel(
        sample_size=config.image_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    return model, optimizer, device


def create_data_loader(dataset, config):
    return DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)


def create_scheduler(optimizer, config, train_dataloader):
    return get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )


def make_image_grid(images, rows=None, cols=None, fill=(0, 0, 0)):
    """
    Arrange images in a grid.

    Parameters:
    - images (list of PIL.Image): The images to arrange in a grid.
    - rows (int): Number of rows in the grid.
    - cols (int): Number of columns in the grid.
    - fill (tuple): Background color for the grid in RGB.

    Returns:
    - A PIL.Image object representing the grid.
    """
    if not rows and not cols:
        raise ValueError("Rows or columns must be specified")
    if not rows:
        rows = (len(images) + cols - 1) // cols
    if not cols:
        cols = (len(images) + rows - 1) // rows

    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images)

    grid_width = cols * max_width
    grid_height = rows * max_height

    grid_image = Image.new("RGB", (grid_width, grid_height), color=fill)
    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        grid_image.paste(image, (col * max_width, row * max_height))

    return grid_image


def evaluate(config, epoch, pipeline):
    images = pipeline(
        batch_size=config.eval_batch_size, generator=torch.manual_seed(config.seed)
    ).images
    image_grid = make_image_grid(images, rows=4, cols=4)
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


def train_loop(config, model, optimizer, train_dataloader, lr_scheduler):
    """
    Performs the training loop for the given model, optimizer, and data loader.

    Args:
        config (TrainingConfig): Configuration object containing training parameters.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
    """
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            clean_images, noise, timesteps, noisy_images = prepare_training_step(
                batch, model, noise_scheduler, accelerator, accelerator.device
            )

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if step % config.gradient_accumulation_steps == 0:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            pipeline = DDPMPipeline(
                unet=accelerator.unwrap_model(model), scheduler=noise_scheduler
            )
            if (
                epoch + 1
            ) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)
            if (
                epoch + 1
            ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                model_path = os.path.join(config.output_dir, f"model_epoch_{epoch}")
                accelerator.save_state(model_path)


def prepare_training_step(batch, model, noise_scheduler, accelerator, device):
    """
    Prepares the training step by processing the input batch and generating noisy images.

    Args:
        batch (dict): A batch from the DataLoader.
        model (torch.nn.Module): The model being trained.
        noise_scheduler (DDPMScheduler): The noise scheduler for the diffusion process.
        accelerator (Accelerator): The Accelerator object for device placement and scaling.
        device (str): The device to use ('cuda' or 'cpu').

    Returns:
        tuple: Tuple containing clean_images, noise, timesteps, and noisy_images tensors.
    """
    clean_images = batch["images"].to(device)
    noise = torch.randn(clean_images.shape).to(device)
    bs = clean_images.size(0)
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (bs,), device=device
    ).long()
    noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
    return clean_images, noise, timesteps, noisy_images


def main():
    args = parse_arguments()

    config = TrainingConfig(class_MRI=args.class_MRI, num_epochs=args.num_epochs)

    dataset = load_and_transform_dataset(config.class_MRI, config.image_size)
    train_dataloader = create_data_loader(dataset, config)

    model, optimizer, device = create_model_and_optimizer(config)

    lr_scheduler = create_scheduler(optimizer, config, train_dataloader)

    train_loop(config, model, optimizer, train_dataloader, lr_scheduler, device)

    # Optionally, you can add a final evaluation phase here, outside the training loop
    # This is useful for evaluating the model after training completes
    # For example:
    # noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    # pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    # evaluate(config, config.num_epochs, pipeline)


if __name__ == "__main__":
    main()
