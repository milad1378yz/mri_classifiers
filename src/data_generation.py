import torch
from diffusers import DiffusionPipeline
import argparse
import os
from typing import List


def generate_data(classes: List[str], numbers_of_augmentation: int) -> None:
    """
    Generates augmented data for a list of classes using a diffusion model.

    This function iterates through a list of class names, loads a pre-trained diffusion model
    for each class, and generates a specified number of augmented images. The augmented images
    are saved in a structured directory format.

    Parameters:
    - classes (List[str]): A list of class names to generate data for.
    - numbers_of_augmentation (int): The number of augmented images to generate per class.

    Returns:
    - None
    """
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    shapes = (176, 208)

    for class_mri in classes:
        num_of_augmentations = numbers_of_augmentation
        print("Load model for", class_mri)
        pipeline = DiffusionPipeline.from_pretrained(
            f"diffusion_models/{class_mri}"
        ).to(device)

        os.makedirs(f"data/generated/{class_mri}", exist_ok=True)
        # Use a batch size of 64 for efficiency
        batch_size = 64
        for i in range((num_of_augmentations // batch_size) + 1):
            print(f"Augmenting {class_mri} {i+1}/{num_of_augmentations // batch_size}")

            augmented_images = pipeline(batch_size=batch_size).images
            for j in range(len(augmented_images)):
                augmented_images[j].resize(shapes).save(
                    f"data/generated/{class_mri}/{i * batch_size + j}.jpg"
                )


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for data generation.

    This function specifies and parses the command-line arguments required for the
    script, facilitating the customization of the number of images to generate.

    Returns:
    - args (argparse.Namespace): The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Data Generation")
    parser.add_argument(
        "--number_of_images", type=int, help="Number of images to generate"
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    # Main function to parse arguments and generate data
    classes = [
        "VeryMildDemented",
        "NonDemented",
        "MildDemented",
        "ModerateDemented",
    ]
    number_of_images = args.number_of_images
    generate_data(classes, number_of_images)


if __name__ == "__main__":
    args = parse_args()
    main(args)
