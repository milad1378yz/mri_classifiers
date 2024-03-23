from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


class KNNClassifier:
    def __init__(self, n_neighbors=5):
        self.clf = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, train_data, train_label, classes):
        # train KNN model
        self.clf.fit(train_data, train_label)
        # predict
        predict_train = self.clf.predict(train_data)
        # accuracy
        print("train accuracy: ", accuracy_score(train_label, predict_train))
        # classification report
        print(
            "train classification report: \n",
            classification_report(train_label, predict_train),
        )
        # confusion matrix
        cm = confusion_matrix(train_label, predict_train)
        print("train confusion matrix: \n", cm)
        # plot confusion matrix
        df_cm = pd.DataFrame(cm, index=classes, columns=classes)
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, cmap="Blues", fmt="g")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("results/train_confusion_matrix_KNN.png")
        # save the report
        report = classification_report(train_label, predict_train, output_dict=True)
        df = pd.DataFrame(report).transpose()
        df.to_csv("results/train_classification_report_KNN.csv")

    def val(self, val_data, val_label, classes):
        # predict
        predict_val = self.clf.predict(val_data)
        # accuracy
        print("validation accuracy: ", accuracy_score(val_label, predict_val))
        # classification report
        print(
            "validation classification report: \n",
            classification_report(val_label, predict_val),
        )
        # confusion matrix
        cm = confusion_matrix(val_label, predict_val)
        print("validation confusion matrix: \n", cm)
        # plot confusion matrix
        df_cm = pd.DataFrame(cm, index=classes, columns=classes)
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, cmap="Blues", fmt="g")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("results/validation_confusion_matrix_KNN.png")
        # save the report
        report = classification_report(val_label, predict_val, output_dict=True)
        df = pd.DataFrame(report).transpose()
        df.to_csv("results/validation_classification_report_KNN.csv")

    def get_model(self):
        return self.clf
