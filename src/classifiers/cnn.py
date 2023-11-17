import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import Tuple



class CNNClassifier(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, learning_rate: float = 0.001):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[1], out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=32*52*44, out_features=64)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)
        self.fc3 = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 32*52*44)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def trainer(self, train_data, train_label, classes, batch_size=32, epochs=10):
        # train CNN model
        train_dataset = CustomDataset(train_data, train_label)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.history = []
        for epoch in range(epochs):
            running_loss = 0.0
            print("epoch: ", epoch)
            for data in train_loader:
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            self.history.append(running_loss / len(train_loader))
        # predict
        predict_train = self.my_predict(train_data,train_label)
        # classification report
        print("train classification report: \n", classification_report(train_label, predict_train, target_names=classes))
        # confusion matrix
        cm = confusion_matrix(train_label, predict_train)
        print("train confusion matrix: \n", cm)
        # plot confusion matrix
        df_cm = pd.DataFrame(cm, index=classes, columns=classes)
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("results/train_confusion_matrix_cnn.png")
        # save the report
        report = classification_report(train_label, predict_train, target_names=classes, output_dict=True)
        df = pd.DataFrame(report).transpose()
        df.to_csv("results/train_classification_report_cnn.csv")

    def vali(self, val_data, val_label, classes):
        # predict
        predict_val = self.my_predict(val_data,val_label)
        # classification report
        print("validation classification report: \n", classification_report(val_label, predict_val, target_names=classes))
        # confusion matrix
        cm = confusion_matrix(val_label, predict_val)
        print("validation confusion matrix: \n", cm)
        # plot confusion matrix
        df_cm = pd.DataFrame(cm, index=classes, columns=classes)
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("results/validation_confusion_matrix_cnn.png")
        # save the report
        report = classification_report(val_label, predict_val, target_names=classes, output_dict=True)
        df = pd.DataFrame(report).transpose()
        df.to_csv("results/validation_classification_report_cnn.csv")

    def my_predict(self, data,labels):
        self.eval()
        dataset = CustomDataset(data, labels)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        with torch.no_grad():
            for iter in loader:
                inputs, labels = iter
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
        return predicted

    def get_model(self):
        return self
    

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()
        self.labels = nn.functional.one_hot(torch.from_numpy(labels).long()).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
                