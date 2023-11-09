import os
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random

data_show = {"Original":[],"Augmented":[]}

def gather_data(classes, data_path):
    folder_dict = {}
    
    for i, class_mri in enumerate(classes):
        # Get the list of files in the folder
        file_list = os.listdir(os.path.join(data_path, class_mri))
        folder_dict[class_mri] = []
        # Filter the list to include only JPG files
        jpg_files = [file for file in file_list if file.endswith(".jpg")]
        
        # Shuffle the list of files
        random.shuffle(jpg_files)
        
        for file in jpg_files:
            image_path = os.path.join(data_path, class_mri, file)
            image = Image.open(image_path)
            
            folder_dict[class_mri].append(np.array(image))  # Add original image
            
    return folder_dict

def augment_data(folder_dict):
    augmented_folder_dict = {}
    max_images_per_class = max(len(images) for images in folder_dict.values())
    
    for class_mri, images in folder_dict.items():
        augmented_images = []
        
        while len(images) < max_images_per_class:
            for image in images:
                augmented_image = Image.fromarray(image)
                augmented_image = augmented_image.copy()
                # Apply data augmentation
                augmented_image = random_erasing(augmented_image)
                
                augmented_image = blurring(augmented_image)
                augmented_image = sharpening(augmented_image)
                
                augmented_images.append(np.array(augmented_image))
                
                data_show["Original"].append(image)
                data_show["Augmented"].append(np.array(augmented_image))

                if len(images) + len(augmented_images) >= max_images_per_class:
                    break
            
            images += augmented_images
            augmented_images = []
        
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
    sharpened_image = enhancer.enhance(2.0)
    
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
    
    plt.show()
if __name__ == '__main__':
    classes = [
        "MildDemented",
        "ModerateDemented",
        "NonDemented",
        "VeryMildDemented"
    ]
    data_path = os.path.join("data", "Alzheimer_MRI_4_classes_dataset")
    folder_dict = gather_data(classes, data_path)
    
    balanced_folder_dict = augment_data(folder_dict)
  
    # Split each class into train, validation, and test
    train_dict = {}
    val_dict = {}
    for class_mri in classes:
        images = balanced_folder_dict[class_mri]
        num_images = len(images)
        num_val = int(num_images * 0.1)  # 10% of data for validation
        
        train_dict[class_mri] = images[:num_images - num_val]
        val_dict[class_mri] = images[num_images - num_val:]

    plot_augmentated_images(5)

    


    
