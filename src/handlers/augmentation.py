import os
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random
import copy
from typing import Dict, List, Tuple, Union

# Global variable to store original and augmented image pairs for visualization
data_show: Dict[str, List[np.ndarray]] = {"Original": [], "Augmented": []}


def augment_data(
    folder_dict: Dict[str, List[np.ndarray]], do_cnn: bool
) -> Dict[str, List[Union[Image.Image, np.ndarray]]]:
    """
    Augments image data for each class to balance the dataset.

    Args:
        folder_dict: A dictionary with class names as keys and lists of images (as numpy arrays) as values.
        do_cnn: Flag to determine the output format; True for CNNs requires RGB images, False for grayscale images.

    Returns:
        A dictionary with class names as keys and lists of augmented images as values. The format of the images
        in the list depends on the do_cnn flag.
    """
    augmented_folder_dict = {}
    max_images_per_class = max(len(images) for images in folder_dict.values())

    for class_mri, images_real in folder_dict.items():
        augmented_images = []
        images = copy.deepcopy(images_real)
        num_of_images = len(images)
        num_of_augmentations = max_images_per_class - num_of_images

        for _ in range(num_of_augmentations):
            image = random.choice(images)
            augmented_image = Image.fromarray(image)

            # Apply data augmentation methods
            augmented_image = random_erasing(augmented_image)
            augmented_image = sharpening(augmented_image)

            if do_cnn:
                out_image = augmented_image.convert("RGB")
            else:
                out_image = augmented_image.convert("L")
                out_image = np.array(out_image)

            augmented_images.append(out_image)
            data_show["Original"].append(image)
            data_show["Augmented"].append(np.array(augmented_image))

        images.extend(augmented_images)
        augmented_folder_dict[class_mri] = images

    return augmented_folder_dict


def random_erasing(image: Image.Image) -> Image.Image:
    """
    Applies random erasing to an image.

    Args:
        image: The input PIL Image object.

    Returns:
        The PIL Image object after applying random erasing.
    """
    width, height = image.size
    x = random.randint(0, width)
    y = random.randint(0, height)
    erase_width = random.randint(0, width // 4)
    erase_height = random.randint(0, height // 4)

    erased_image = image.copy()
    erased_image.paste((0), (x, y, x + erase_width, y + erase_height))

    return erased_image


def sharpening(image: Image.Image) -> Image.Image:
    """
    Sharpens an image.

    Args:
        image: The input PIL Image object.

    Returns:
        The sharpened PIL Image object.
    """
    enhancer = ImageEnhance.Sharpness(image)
    sharpened_image = enhancer.enhance(2.5)

    return sharpened_image


def blurring(image: Image.Image) -> Image.Image:
    """
    Applies a blurring effect to an image.

    Args:
        image: The input PIL Image object.

    Returns:
        The blurred PIL Image object.
    """
    blurred_image = image.filter(ImageFilter.BLUR)

    return blurred_image


def plot_augmentated_images(num_images: int) -> None:
    """
    Plots a specified number of original and augmented image pairs.

    Args:
        num_images: The number of image pairs to plot.

    Returns:
        None. The function saves the plot to a file.
    """
    fig, axes = plt.subplots(num_images, 2, figsize=(8, 10))
    plt.tight_layout()
    sampled_indices = random.sample(range(len(data_show["Original"])), num_images)
    for i, idx in enumerate(sampled_indices):
        original, augmented = data_show["Original"][idx], data_show["Augmented"][idx]

        axes[i, 0].imshow(original, cmap="gray")
        axes[i, 0].axis("off")
        axes[i, 0].set_title("Original")

        axes[i, 1].imshow(augmented, cmap="gray")
        axes[i, 1].axis("off")
        axes[i, 1].set_title("Augmented")

    plt.savefig("results/augmented_images.png")
