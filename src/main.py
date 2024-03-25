import os
from PIL import Image
import numpy as np
import random
import argparse
from typing import List, Dict, Tuple

from handlers.parser import parse_config
from handlers.augmentation import augment_data, plot_augmentated_images, generate_data


def gather_dataset_images(
    classes: List[str], dataset_path: str, for_cnn: bool
) -> Dict[str, List[Image.Image]]:
    """
    Collects and processes images from the dataset for each specified class.

    Args:
        classes: List of class names.
        dataset_path: Root path to the dataset.
        for_cnn: Whether to convert images for CNN input (RGB).

    Returns:
        A dictionary with class names as keys and a list of image data as values.
    """
    folder_dict = {}

    for i, class_mri in enumerate(classes):
        # Get the list of files in the folder
        file_list = os.listdir(os.path.join(dataset_path, class_mri))
        folder_dict[class_mri] = []
        # Filter the list to include only JPG files
        jpg_files = [file for file in file_list if file.endswith(".jpg")]

        # Shuffle the list of files
        random.shuffle(jpg_files)

        for file in jpg_files:
            image_path = os.path.join(dataset_path, class_mri, file)
            if for_cnn:
                image = Image.open(image_path).convert("RGB")
                folder_dict[class_mri].append(image)  # Add original image
            else:
                image = Image.open(image_path)
                folder_dict[class_mri].append(np.array(image))  # Add original image

    return folder_dict


def split_data_for_training(
    images: List[np.ndarray], validation_ratio: float = 0.1
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Splits the images into training and validation sets based on the specified ratio.

    Args:
        images (List[np.ndarray]): List of image arrays.
        validation_ratio (float): The proportion of images to be used for validation.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Two lists containing training and validation images, respectively.
    """
    num_val = int(len(images) * validation_ratio)
    return images[:-num_val], images[-num_val:]


def prepare_data_arrays(
    image_data: Dict[str, List[np.ndarray]], classes: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepares flattened arrays for training and validation data and their corresponding labels.

    Args:
        image_data (Dict[str, List[np.ndarray]]): Dictionary with class names as keys and list of images as values.
        classes (List[str]): List of class names.

    Returns:
        Tuple containing arrays for training data, training labels, validation data, and validation labels.
    """
    train_images, train_labels, val_images, val_labels = [], [], [], []

    for idx, class_name in enumerate(classes):
        train, val = split_data_for_training(image_data[class_name])
        train_images.extend([image.flatten() for image in train])
        val_images.extend([image.flatten() for image in val])
        train_labels.extend([idx] * len(train))
        val_labels.extend([idx] * len(val))

    # Convert to numpy arrays
    train_data, train_label = np.array(train_images), np.array(train_labels)
    val_data, val_label = np.array(val_images), np.array(val_labels)

    # Shuffle the training and validation datasets
    train_data, train_label = shuffle_data(train_data, train_label)
    val_data, val_label = shuffle_data(val_data, val_label)

    print_data_shapes(train_data, val_data, train_label, val_label)

    return train_data, train_label, val_data, val_label


def shuffle_data(data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shuffles the data and labels in unison.

    Args:
        data (np.ndarray): Data to be shuffled.
        labels (np.ndarray): Corresponding labels to be shuffled.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Shuffled data and labels.
    """
    assert len(data) == len(labels)
    p = np.random.permutation(len(data))
    return data[p], labels[p]


def print_data_shapes(
    train_data: np.ndarray,
    val_data: np.ndarray,
    train_label: np.ndarray,
    val_label: np.ndarray,
):
    """
    Prints the shapes of training and validation data and labels.

    Args:
        train_data (np.ndarray): Training data array.
        val_data (np.ndarray): Validation data array.
        train_label (np.ndarray): Training labels array.
        val_label (np.ndarray): Validation labels array.
    """
    print(f"data_train shape is: {train_data.shape}")
    print(f"data_val shape is: {val_data.shape}")
    print(f"label_train shape is: {train_label.shape}")
    print(f"label_val shape is: {val_label.shape}")


def prepare_data_for_cnn(
    image_collections: Dict[str, List[np.ndarray]], class_labels: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepares data for CNN training by organizing images into training and validation sets, along with their labels.

    Args:
        image_collections (Dict[str, List[np.ndarray]]): A dictionary where keys are class names and values are lists of image arrays.
        class_labels (List[str]): A list of class names corresponding to the keys in `image_collections`.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Four numpy arrays representing training data, training labels, validation data, and validation labels, respectively.
    """
    training_images, training_labels, validation_images, validation_labels = (
        [],
        [],
        [],
        [],
    )

    for class_index, class_name in enumerate(class_labels):
        images = image_collections[class_name]
        random.shuffle(images)
        split_index = int(
            len(images) * 0.9
        )  # Split index for 90% training and 10% validation

        for i, image in enumerate(images):
            if i < split_index:
                training_images.append(image)
                training_labels.append(class_index)
            else:
                validation_images.append(image)
                validation_labels.append(class_index)

    # Convert lists to numpy arrays
    training_data = np.array(training_images, dtype=np.float32)
    training_labels = np.array(training_labels, dtype=np.int32)
    validation_data = np.array(validation_images, dtype=np.float32)
    validation_labels = np.array(validation_labels, dtype=np.int32)

    return training_data, training_labels, validation_data, validation_labels


def main(args):
    classes = [
        "NonDemented",
        "VeryMildDemented",
        "MildDemented",
        "ModerateDemented",
    ]
    data_path = args.data_path
    print("Gathering data...")
    folder_dict = gather_dataset_images(classes, data_path, args.do_cnn)
    print("Data gathered.")

    if args.do_augmentation:
        print("Starting augmentation...")
        if args.do_generation:
            print("Starting data generation...")
            balanced_folder_dict = generate_data(folder_dict, 5000, args.do_cnn)
            print("Data generation completed.")
        else:
            balanced_folder_dict = augment_data(folder_dict, args.do_cnn)
            print("Augmented data is ready.")
            plot_augmentated_images(5)
            print("Augmentation images plotted.")

        if args.do_cnn:
            train_data, train_label, val_data, val_label = prepare_data_for_cnn(
                balanced_folder_dict, classes
            )
        else:
            train_data, train_label, val_data, val_label = prepare_data_arrays(
                balanced_folder_dict, classes
            )
    else:
        if args.do_cnn:
            train_data, train_label, val_data, val_label = prepare_data_for_cnn(
                folder_dict, classes
            )
        else:
            train_data, train_label, val_data, val_label = prepare_data_arrays(
                folder_dict, classes
            )

    print("Train and validation data are ready.")

    # convert labels to binary
    if args.convert_binary:
        train_label = np.where(train_label == 0, 0, 1)
        val_label = np.where(val_label == 0, 0, 1)
        print("convert binary done")
        classes = ["NonDemented", "Demented"]

    # feature selection
    if args.feature_selection and not args.do_cnn:
        from handlers.feature_selector import FeatureSelector

        print("start feature selection")
        feature_selector = FeatureSelector(k=args.num_feature)
        feature_selector.train(train_data, train_label)
        train_data = feature_selector.transform(train_data)
        val_data = feature_selector.transform(val_data)
        print("feature selection done")

    # apply SVM
    if args.do_svm:
        from classifiers.svm import SVMClassifier

        print("start svm")
        svm_classifier = SVMClassifier(max_iter=args.max_iter)
        svm_classifier.train(train_data, train_label, classes)
        svm_classifier.val(val_data, val_label, classes)
        print("svm done")

    # apply KNN
    if args.do_knn:
        from classifiers.knn import KNNClassifier

        print("start knn")
        knn_classifier = KNNClassifier(n_neighbors=args.n_neighbors)
        knn_classifier.train(train_data, train_label, classes)
        knn_classifier.val(val_data, val_label, classes)
        print("knn done")

    # apply random forest
    if args.do_random_forest:
        from classifiers.random_forest import RandomForest

        print("start random forest")
        random_forest_classifier = RandomForest(args.n_estimators, args.max_depth)
        random_forest_classifier.train(train_data, train_label, classes)
        random_forest_classifier.val(val_data, val_label, classes)
        print("random forest done")

    # apply MLP
    if args.do_mlp:
        from classifiers.mlp import MLP

        print("start mlp")
        mlp_classifier = MLP(
            hidden_layer_sizes=args.hidden_layer_sizes,
            activation=args.activation,
            learning_rate_init=args.learning_rate_init,
            validation_fraction=args.validation_fraction,
        )
        mlp_classifier.train(train_data, train_label, classes)
        mlp_classifier.val(val_data, val_label, classes)
        print("mlp done")

    # apply logistic regression
    if args.do_logistic_regression:
        from classifiers.logistic_regression import LogisticRegressionClassifier

        print("start logistic regression")
        logistic_regression_classifier = LogisticRegressionClassifier()
        logistic_regression_classifier.train(train_data, train_label, classes)
        logistic_regression_classifier.val(val_data, val_label, classes)
        print("logistic regression done")

    # apply CNN
    if args.do_cnn:
        from classifiers.cnn import CNNClassifier

        print("start cnn")
        cnn_classifier = CNNClassifier(
            num_classes=len(classes), learning_rate=args.learning_rate_cnn
        )
        cnn_classifier.trainer(
            train_data,
            train_label,
            val_data,
            val_label,
            classes,
            batch_size=args.batch_size_cnn,
            epochs=args.num_epochs_cnn,
        )
        cnn_classifier.vali(val_data, val_label, classes)
        print("cnn done")

    # apply ensemble AdaBoost
    if args.do_ensemble_adaboost:
        from classifiers.adaboost import AdaBoost

        print("start ensemble AdaBoost")
        ensemble_adaboost_classifier = AdaBoost()
        ensemble_adaboost_classifier.train(train_data, train_label, classes)
        ensemble_adaboost_classifier.val(val_data, val_label, classes)
        print("ensemble AdaBoost done")

    # apply decision tree
    if args.do_decision_tree:
        from classifiers.decision_tree import DecisionTree

        print("start decision tree")
        decision_tree_classifier = DecisionTree(args.max_depth_decision_tree)
        decision_tree_classifier.train(train_data, train_label, classes)
        decision_tree_classifier.val(val_data, val_label, classes)
        print("decision tree done")

    # apply naive bayes
    if args.do_naive_bayes:
        from classifiers.naive_bayes import NaiveBayes_Classifier

        print("start naive bayes")
        naive_bayes_classifier = NaiveBayes_Classifier()
        naive_bayes_classifier.train(train_data, train_label, classes)
        naive_bayes_classifier.val(val_data, val_label, classes)
        print("naive bayes done")

    # apply vision transformer
    if args.do_vit:
        from classifiers.vit import VisionTransformerClassifier

        print("start vision transformer")
        vit_classifier = VisionTransformerClassifier(
            num_classes=len(classes), learning_rate=args.learning_rate_cnn
        )
        vit_classifier.trainer(
            train_data,
            train_label,
            val_data,
            val_label,
            classes,
            batch_size=args.batch_size_cnn,
            epochs=args.num_epochs_cnn,
        )
        vit_classifier.vali(val_data, val_label, classes)
        print("vision transformer done")


if __name__ == "__main__":
    print("Start")
    args = parse_config(r"configs/config.yaml")
    main(args)
