import os
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random
import copy

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

def convert_to_array(folder_dict,classes):
    # Split each class into train, validation, and test
    train_dict = {}
    val_dict = {}
    for class_mri in classes:
        images = folder_dict[class_mri]
        num_images = len(images)
        num_val = int(num_images * 0.1)  # 10% of data for validation
        
        train_dict[class_mri] = images[:num_images - num_val]
        val_dict[class_mri] = images[num_images - num_val:]
    # flatten images
    for class_mri in classes:
        for i in range(len(train_dict[class_mri])):
            train_dict[class_mri][i] = train_dict[class_mri][i].flatten()
        for i in range(len(val_dict[class_mri])):
            val_dict[class_mri][i] = val_dict[class_mri][i].flatten()
    # convert train data to array
    train_data = []
    train_label = []
    for i,class_mri in enumerate(classes):
        for image in train_dict[class_mri]:
            train_data.append(image)
            train_label.append(i)
    train_data = np.array(train_data)
    train_label = np.array(train_label)

    # convert validation data to array
    val_data = []
    val_label = []
    for i,class_mri in enumerate(classes):
        for image in val_dict[class_mri]:
            val_data.append(image)
            val_label.append(i)
    val_data = np.array(val_data)
    val_label = np.array(val_label)

    # shuffle train data and validation data
    combined_data_train = np.column_stack((train_data, train_label))
    combined_data_val = np.column_stack((val_data, val_label))

    # Shuffle the combined data along the first axis
    np.random.shuffle(combined_data_train)
    np.random.shuffle(combined_data_val)

    # Split the shuffled data back into separate arrays
    train_data = combined_data_train[:, :-1]
    train_label = combined_data_train[:, -1]
    val_data = combined_data_val[:, :-1]
    val_label = combined_data_val[:, -1]

    return train_data, train_label, val_data, val_label


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
if __name__ == '__main__':
    print("Start")
    classes = [
        "NonDemented",
        "MildDemented",
        "ModerateDemented",
        "VeryMildDemented"
    ]
    data_path = os.path.join("data", "Alzheimer_MRI_4_classes_dataset")
    folder_dict = gather_data(classes, data_path)
    
    balanced_folder_dict = augment_data(folder_dict)

    print("augemented data is ready")
    plot_augmentated_images(5)
    print("plot done")
    # convert to array
    train_data, train_label, val_data, val_label = convert_to_array(balanced_folder_dict,classes)
    print("train data and validation data is ready")

    # trian SVM model for train data
    from sklearn import svm
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.feature_selection import SelectKBest, f_classif

    # feature selection
    print("feature selection")
    k_best = SelectKBest(score_func=f_classif, k=16)
    X_selected = k_best.fit(train_data, train_label)
    X_selected = X_selected.transform(train_data)
    print("feature selection done")
    # train SVM model
    clf = svm.SVC(kernel='linear')
    clf.fit(X_selected, train_label)
    # predict
    predict_train = clf.predict(X_selected)
    # accuracy
    print("train accuracy: ", accuracy_score(train_label, predict_train))
    # classification report
    print("train classification report: \n", classification_report(train_label, predict_train))
    # confusion matrix
    cm = confusion_matrix(train_label, predict_train)
    print("train confusion matrix: \n", cm)
    # plot confusion matrix
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("results/train_confusion_matrix.png")

    # validation
    # feature selection
    print("feature selection")
    X_selected = k_best.transform(val_data)
    print("feature selection done")
    # predict
    predict_val = clf.predict(X_selected)
    # accuracy
    print("validation accuracy: ", accuracy_score(val_label, predict_val))
    # classification report
    print("validation classification report: \n", classification_report(val_label, predict_val))
    # confusion matrix
    cm = confusion_matrix(val_label, predict_val)
    print("validation confusion matrix: \n", cm)
    # plot confusion matrix
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("results/validation_confusion_matrix.png")


