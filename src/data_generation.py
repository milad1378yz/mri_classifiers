import torch
from diffusers import DiffusionPipeline
import argparse
import os


def generate_data(classes,numbers_of_augmentation):
    # find the maximum number of images per class
    max_images_per_class = numbers_of_augmentation
    # chekc for cuda avalibility
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    shapes = (176,208)
    
    
    for class_mri in classes:
        num_of_augmentations = max_images_per_class
        print("load model for",class_mri)
        pipeline = DiffusionPipeline.from_pretrained("diffiusion_models/"+class_mri).to(device)

        os.makedirs(f"data/generated/{class_mri}",exist_ok=True)
        # use batch size of 64
        batch_size = 64
        for i in range(num_of_augmentations//batch_size+1):

            print(f"Augmenting {class_mri} {i+1}/{num_of_augmentations//batch_size}")
            
            augmented_image = pipeline(batch_size=batch_size).images
            for j in range(batch_size):
                augmented_image[j].resize(shapes).save(f"data/generated/{class_mri}/{i*batch_size+j}.jpg")
        

def parse_args():
    parser = argparse.ArgumentParser(description='Data Generation')
    parser.add_argument('--number_of_images', type=int, help='Number to get')
    return parser.parse_args()

# add main function
if __name__ == '__main__':
    args = parse_args()
    classes = [
        "VeryMildDemented",
        "NonDemented",
        "MildDemented",
        "ModerateDemented",
    ]
    number_of_images = args.number_of_images
    generate_data(classes,number_of_images)
    ## save the augmented iamges as jpg
    # for class_name,images in augmented_folder_dict.items():
    #     # make the directory if not exist
    #     os.makedirs(f"data/generated/{class_name}",exist_ok=True)
    #     for i,image in enumerate(images):
    #         image.save(f"data/generated/{class_name}/{i}.jpg")



    
    
