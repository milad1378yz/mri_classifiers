import os
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random
import copy

data_show = {"Original":[],"Augmented":[]}
def augment_data(folder_dict):
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
            augmented_image = augmented_image.copy()
            # Apply data augmentation
            augmented_image = random_erasing(augmented_image)
            
            # augmented_image = blurring(augmented_image)
            augmented_image = sharpening(augmented_image)
            
            augmented_images.append(np.array(augmented_image))
            
            data_show["Original"].append(image)
            data_show["Augmented"].append(np.array(augmented_image))

        images.extend(augmented_images)
        
        augmented_folder_dict[class_mri] = images
    
    return augmented_folder_dict

def random_erasing(image):
    # Apply random erasing
    width, height = image.size
    x = random.randint(0, width)
    y = random.randint(0, height)
    erase_width = random.randint(0, width // 4)
    erase_height = random.randint(0, height // 4)
    
    erased_image = image.copy()
    erased_image.paste((0), (x, y, x + erase_width, y + erase_height))
    
    return erased_image

def sharpening(image):
    # Apply sharpening
    enhancer = ImageEnhance.Sharpness(image)
    sharpened_image = enhancer.enhance(2.5)
    
    return sharpened_image

def blurring(image):
    # Apply blurring
    blurred_image = image.filter(ImageFilter.BLUR)
    
    return blurred_image




def plot_augmentated_images(num_images):
    fig, axes = plt.subplots(num_images, 2, figsize=(8, 10))
    plt.tight_layout()
    i = 0
    for j in random.sample(range(len(data_show["Original"])), num_images):
        axes[i, 0].imshow(data_show["Original"][j], cmap='gray')
        axes[i, 0].axis('off')
        axes[i, 0].set_title("Original")
        
        axes[i, 1].imshow(data_show["Augmented"][j], cmap='gray')
        axes[i, 1].axis('off')
        axes[i, 1].set_title("Augmented")
        i = i + 1
    
    plt.savefig("results/augmented_images.png")