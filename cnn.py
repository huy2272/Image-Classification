import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import os.path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
from torchvision import models, transforms
from sklearn.metrics import precision_score, recall_score,f1_score, confusion_matrix
from utils import select_n_img, extract_feature_vectors

if __name__ == "__main__":
    data_dir = "./"
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    image_datasets = {
        "train": torchvision.datasets.CIFAR10(
            data_dir, train=True, download=True, transform=data_transforms["train"]
        ),
        "test": torchvision.datasets.CIFAR10(
            data_dir, train=False, download=True, transform=data_transforms["test"]
        ),
    }

    selected_train_dataset, selected_test_dataset = select_n_img(image_datasets)
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            selected_train_dataset, batch_size=500, shuffle=True, num_workers=4
        ),
        "test": torch.utils.data.DataLoader(
            selected_test_dataset, batch_size=100, shuffle=True, num_workers=4
        ),
    }

    pretrained_model = models.resnet18(pretrained=True)
    # remove the last layer of ResNet-18
    modified_model = nn.Sequential(*list(pretrained_model.children())[:-1])

    # We need to set requires_grad = False to freeze the parameters
    for param in pretrained_model.parameters():
        param.requires_grad = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pretrained_model = pretrained_model.to(device)

    train_features_pca, test_features_pca, train_labels_np, test_labels_np = (
        extract_feature_vectors(
            modified_model=modified_model, dataloaders=dataloaders, device=device
        )
    )

    # 6: Convolutional Neural Network
    #MAIN

    train = torch.utils.data.DataLoader(
            selected_train_dataset, batch_size=500, shuffle=True, num_workers=4
        )
    test = torch.utils.data.DataLoader(
            selected_test_dataset, batch_size=100, shuffle=False, num_workers=4
        )

    cnn_model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),

        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),

        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),

        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Flatten(),

        nn.Linear(512 * 1 * 1, 4096),  # Assuming input size is 32x32 or similar (adjust if necessary)
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(4096, 10)
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer_cnn = optim.SGD(cnn_model.parameters(), lr=0.01, momentum=0.9)

    if(os.path.isfile(r"C:\Users\mo_usm\Desktop\472\Image-Classification\main_cnn_model.pth")):
        cnn_model.load_state_dict(torch.load("main_cnn_model.pth"))
    else:

        #Training
        epoch = 10

        for e in range(epoch):
            cnn_model.train()

            epoch_loss=0.0

            for inputs, targets in train:
                #Forward Pass
                outputs = cnn_model(inputs) # get output by passing input data through each layer
                loss = loss_fn(outputs, targets) #Calc loss by comparing output to actual values

                # Backward pass
                optimizer_cnn.zero_grad()  # Clear gradients
                loss.backward()  # Compute gradients
                optimizer_cnn.step()  # Update weights
                epoch_loss += loss.item()

            # Print epoch stats
            print(f"Epoch {e + 1}/{epoch}, Loss: {epoch_loss / len(train):.4f}")

        # Save the trained model

        torch.save(cnn_model.state_dict(), "trained_mlp_model.pth")

        print("Model saved after training.")

    cnn_model.eval()  # Set model to evaluation mode

    test_loss = 0.0
    correct = 0
    total = 0
    y_preds = []
    y_true = []

    with torch.no_grad():  # Disable gradient computation
        for inputs, targets in test:
            outputs = cnn_model(inputs)  # Forward pass

            loss = loss_fn(outputs, targets)  # Compute loss

            test_loss += loss.item() # total loss

            # Compute accuracy
            _, predicted = outputs.max(1)  # Get predicted class

            y_preds.extend(predicted.cpu().numpy()) # Store all predictions
            y_true.extend(targets.cpu().numpy()) # Store all Truth labels

            total += targets.size(0)

            correct += (predicted == targets).sum().item()

    # Compute average loss, accuracy, precision, recall and f1 measure

    avg_loss = test_loss / len(test)
    accuracy = 100.0 * correct / total
    precision = precision_score(y_true, y_preds, average='weighted')
    recall = recall_score(y_true, y_preds, average='weighted')
    f1 = f1_score(y_true, y_preds, average='weighted')

    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")