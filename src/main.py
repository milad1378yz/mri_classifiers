import os
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random
import copy

from handlers.parser import parse_config
from handlers.augmentation import augment_data, plot_augmentated_images

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


def convert_to_array_cnn(folder_dict,classes):
    train_data = []
    train_label = []
    val_data = []
    val_label = []
    for class_index, class_name in enumerate(classes):
        data = folder_dict[class_name]
        random.shuffle(data)
        num_train = int(len(data)*0.9)
        for i in range(len(data)):
            # each input's shape should be (1,208,176)
            data_shown = np.array(data[i])
            if i < num_train:
                train_data.append(data_shown.reshape(1,data_shown.shape[0],data_shown.shape[1]))
                train_label.append(class_index)
            else:
                val_data.append(data_shown.reshape(1,data_shown.shape[0],data_shown.shape[1]))
                val_label.append(class_index)
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    val_data = np.array(val_data)
    val_label = np.array(val_label)
    return train_data, train_label, val_data, val_label
    



if __name__ == '__main__':
    print("Start")
    args = parse_config(r'configs\config.yaml')
    classes = [
        "NonDemented",
        "VeryMildDemented",
        "MildDemented",
        "ModerateDemented",
    ]
    data_path = args.data_path
    folder_dict = gather_data(classes, data_path)
    
    if args.do_augmentation:
        print("start augmentation")
        balanced_folder_dict = augment_data(folder_dict)
        print("augemented data is ready")
        plot_augmentated_images(5)
        print("plot augmentated image is done")
        
        if args.do_cnn:
            train_data, train_label, val_data, val_label = convert_to_array_cnn(balanced_folder_dict,classes)
        else:
            train_data, train_label, val_data, val_label = convert_to_array(balanced_folder_dict,classes)
    else:
        if args.do_cnn:
            train_data, train_label, val_data, val_label = convert_to_array_cnn(folder_dict,classes)
        else:
            train_data, train_label, val_data, val_label = convert_to_array(folder_dict,classes)
    
    print("train data and validation data is ready")

    # convert labels to binary
    if args.convert_binary:
        train_label = np.where(train_label == 0, 0, 1)
        val_label = np.where(val_label == 0, 0, 1)
        print("convert binary done")
        classes = ["NonDemented","Demented"]
    
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
        knn_classifier = KNNClassifier(n_neighbors = args.n_neighbors)
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
        mlp_classifier = MLP(hidden_layer_sizes=args.hidden_layer_sizes, activation=args.activation, learning_rate_init=args.learning_rate_init, validation_fraction=args.validation_fraction)
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
        cnn_classifier = CNNClassifier(input_shape=train_data.shape, num_classes=len(classes), learning_rate=args.learning_rate_cnn)
        cnn_classifier.trainer(train_data, train_label, classes, batch_size=args.batch_size_cnn, epochs=args.num_epochs_cnn)
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

