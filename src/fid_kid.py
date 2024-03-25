import argparse
import os
import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from typing import Tuple


def parse_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate FID and KID metrics for images."
    )
    parser.add_argument(
        "--true_images_dir",
        type=str,
        default="data/Alzheimer_MRI_4_classes_dataset",
        help="Directory path of true images.",
    )
    parser.add_argument(
        "--generated_images_dir",
        type=str,
        default="data/generated",
        help="Directory path of generated images.",
    )
    return parser.parse_args()


def load_and_process_images(
    image_directory: str, number_of_images: int = 100
) -> torch.Tensor:
    """Loads images from a specified directory, processes, and converts them into a PyTorch tensor.

    Args:
        image_directory (str): The directory containing the images to load.
        number_of_images (int): The number of images to load and process.

    Returns:
        torch.Tensor: A tensor containing the processed images.
    """
    processed_images = []
    image_filenames = random.sample(os.listdir(image_directory), number_of_images)
    for filename in image_filenames:
        with Image.open(os.path.join(image_directory, filename)) as image:
            resized_image = transforms.Resize((128, 128))(image.convert("RGB"))
            processed_images.append(np.array(resized_image))
    processed_images_array = np.array(processed_images).transpose(
        (0, 3, 1, 2)
    )  # Rearrange axis for PyTorch compatibility
    images_tensor = torch.tensor(processed_images_array, dtype=torch.uint8)
    return images_tensor


def compute_image_metrics(
    true_images: torch.Tensor, generated_images: torch.Tensor
) -> Tuple[float, Tuple[float, float]]:
    """Calculates FID and KID metrics to compare sets of true and generated images.

    Args:
        true_images (torch.Tensor): Tensor containing true images.
        generated_images (torch.Tensor): Tensor containing generated images.

    Returns:
        Tuple[float, Tuple[float, float]]: FID score and KID mean and std deviation.
    """
    fid_metric = FrechetInceptionDistance(feature=64)
    fid_metric.update(true_images, real=True)
    fid_metric.update(generated_images, real=False)
    fid_score = fid_metric.compute()

    kid_metric = KernelInceptionDistance(subset_size=50)
    kid_metric.update(true_images, real=True)
    kid_metric.update(generated_images, real=False)
    kid_mean, kid_std = kid_metric.compute()

    return fid_score, (kid_mean, kid_std)


def main():
    args = parse_command_line_arguments()

    random.seed(0)  # Ensure reproducibility

    true_image_tensor = load_and_process_images(
        args.true_images_dir, number_of_images=100
    )
    generated_image_tensor = load_and_process_images(
        args.generated_images_dir, number_of_images=1000
    )

    fid_score, (kid_mean, kid_std) = compute_image_metrics(
        true_image_tensor, generated_image_tensor
    )
    print(f"FID: {fid_score}")
    print(f"KID: Mean = {kid_mean}, Std = {kid_std}")


if __name__ == "__main__":
    main()
