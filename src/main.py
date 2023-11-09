import os
import matplotlib.pyplot as plt
from PIL import Image



if __name__ == '__main__':
    folder_paths = [
        "MildDemented",
        "ModerateDemented",
        "NonDemented",
        "VeryMildDemented"
    ]
    size_list = []
    for i, folder_path in enumerate(folder_paths):
        # Get the list of files in the folder
        file_list = os.listdir(os.path.join("Alzheimer_MRI_4_classes_dataset",folder_path))
        
        # Filter the list to include only JPG files
        jpg_files = [file for file in file_list if file.endswith(".jpg")]
        size_list.append(len(jpg_files))
        print(size_list)
            
    