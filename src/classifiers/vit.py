import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
from torchvision import models
import os

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torchvision.models import VisionTransformer


class VisionTransformerClassifier(nn.Module):
    def __init__(self, num_classes: int, learning_rate: float = 0.001):
        super(VisionTransformerClassifier, self).__init__()
        # Load a pre-trained Vision Transformer model
        print("Loading pre-trained Vision Transformer model...")
        self.num_classes = num_classes
        self.vit = VisionTransformer.from_pretrained(
            "vit_base_patch16_224", num_classes=num_classes
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.vit.parameters(), lr=learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform_ops = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def forward(self, x):
        return self.vit(x)

    def trainer(
        self,
        train_data,
        train_label,
        val_data,
        val_label,
        classes,
        batch_size=32,
        epochs=40,
    ):
        # train Vision Transformer model
        # check for cuda availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        train_dataset = CustomDataset(train_data, train_label, self.transform_ops)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = CustomDataset(val_data, val_label, self.transform_ops)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        self.history = []
        for epoch in range(epochs):
            running_loss = 0.0
            print("epoch: ", epoch)
            for data in train_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            self.history.append(running_loss / len(train_loader))
            print("loss: ", running_loss / len(train_loader))
            train_predictions = self.my_predict(train_data, train_label)
            train_acc = self.compute_accuracy(
                torch.tensor(train_predictions), torch.tensor(train_label)
            )
            print(f"Training Accuracy: {train_acc:.2f}")

            # Calculate validation accuracy
            val_predictions = self.my_predict(val_data, val_label)
            val_acc = self.compute_accuracy(
                torch.tensor(val_predictions), torch.tensor(val_label)
            )
            print(f"Validation Accuracy: {val_acc:.2f}")

        # print("train accuracy: ", self.my_predict(train_data,train_label).mean())
        # predict
        predict_train = self.my_predict(train_data, train_label)
        # classification report
        print(
            "train classification report: \n",
            classification_report(train_label, predict_train, target_names=classes),
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
        plt.savefig("results/train_confusion_matrix_vit.png")
        # save the report
        report = classification_report(
            train_label, predict_train, target_names=classes, output_dict=True
        )
        df = pd.DataFrame(report).transpose()
        df.to_csv("results/train_classification_report_vit.csv")

    def vali(self, val_data, val_label, classes):
        # predict
        predict_val = self.my_predict(val_data, val_label)
        # classification report
        print(
            "validation classification report: \n",
            classification_report(val_label, predict_val, target_names=classes),
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
        plt.savefig("results/validation_confusion_matrix_vit.png")
        # save the report
        report = classification_report(
            val_label, predict_val, target_names=classes, output_dict=True
        )
        df = pd.DataFrame(report).transpose()
        df.to_csv("results/validation_classification_report_vit.csv")

    def my_predict(self, data, labels):
        self.eval()
        dataset = CustomDataset(data, labels, self.transform_ops)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        out_predicted = []
        with torch.no_grad():
            for iter in loader:
                inputs, labels = iter
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                # send back to cpu
                predicted = predicted.cpu()
                out_predicted += predicted.tolist()
        print(len(out_predicted))
        return np.array(out_predicted)

    def compute_accuracy(self, predictions, labels):
        """Compute accuracy given predictions and labels."""
        predictions, labels = predictions.to(self.device), labels.to(self.device)
        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        return accuracy


class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        super().__init__()
        self.transform = transform
        self.data = data
        self.labels = labels
        print(len(data), "data")
        print(len(labels), "labels")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
