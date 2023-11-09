import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def gather_data(classes, data_path):
    folder_dict = {}
    for i, class_mri in enumerate(classes):
        # Get the list of files in the folder
        file_list = os.listdir(os.path.join(data_path,class_mri))
        folder_dict[class_mri] = []
        # Filter the list to include only JPG files
        jpg_files = [file for file in file_list if file.endswith(".jpg")]
        for file in jpg_files:
            image_path = os.path.join(data_path,class_mri,file)
            image = np.array(Image.open(image_path))
            folder_dict[class_mri].append(image)
    return folder_dict

if __name__ == '__main__':
    classes = [
        "MildDemented",
        "ModerateDemented",
        "NonDemented",
        "VeryMildDemented"
    ]
    data_path = os.path.join("data","Alzheimer_MRI_4_classes_dataset")
    folder_dict = gather_data(classes, data_path)
    # split each class into train, validation and test
    train_dict = {}
    val_dict = {}
    test_dict = {}
    for class_mri in classes:
        train_dict[class_mri] = folder_dict[class_mri][:1000]
        val_dict[class_mri] = folder_dict[class_mri][1000:1200]
        test_dict[class_mri] = folder_dict[class_mri][1200:]

    

    

            
            
    