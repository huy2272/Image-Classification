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
from utils import extract_feature_vectors, select_n_img

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

    # 5: MultiLayer Perceptron

    # Model Architecture (MAIN)

    mlp_model = nn.Sequential(

        nn.Linear(50, 512),  # Input layer

        nn.ReLU(),  # Activation function

        nn.Linear(512, 512),  # Hidden layer

        nn.BatchNorm1d(512),  # Batch Normalization

        nn.ReLU(),  # Activation function

        nn.Linear(512, 10)  # Output layer

    )

    # loss function

    criterion = nn.CrossEntropyLoss()

    # Dataloader for train
    train_loader = torch.utils.data.DataLoader(

        dataset=torch.utils.data.TensorDataset(torch.tensor(train_features_pca), torch.tensor(train_labels_np)),

        batch_size=500,

        shuffle=False

    )

    if (os.path.isfile("./MLP/main_mlp_model.pth")):

        # Load the trained weights

        mlp_model.load_state_dict(torch.load("./MLP/main_mlp_model.pth", weights_only=True))

    else:

        # optimizer (SGD with momentum)

        optimizer = optim.SGD(mlp_model.parameters(), lr=0.01, momentum=0.9)

        # training loop

        num_epochs = 30

        # Training loop

        for epoch in range(num_epochs):

            mlp_model.train()  # Set model to training mode

            epoch_loss = 0.0

            for inputs, targets in train_loader:
                # Forward pass

                outputs = mlp_model(inputs)

                loss = criterion(outputs, targets)

                # Backward pass

                optimizer.zero_grad()  # Clear gradients

                loss.backward()  # Compute gradients

                optimizer.step()  # Update weights

                epoch_loss += loss.item()

            # Print epoch stats

            #print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

        # Save the trained model

        torch.save(mlp_model.state_dict(), "./MLP/main_mlp_model.pth")

        print("Model saved after training.")

    # Testing Model

    test_loader = torch.utils.data.DataLoader(

        dataset=torch.utils.data.TensorDataset(torch.tensor(test_features_pca), torch.tensor(test_labels_np)),

        batch_size=500,

        shuffle=False

    )

    mlp_model.eval()  # Set model to evaluation mode

    test_loss = 0.0

    correct = 0

    total = 0

    all_preds = []

    all_targets = []

    with torch.no_grad():  # Disable gradient computation

        for inputs, targets in test_loader:
            outputs = mlp_model(inputs)  # Forward pass

            loss = criterion(outputs, targets)  # Compute loss

            test_loss += loss.item()  # total loss

            # Compute accuracy
            _, predicted = outputs.max(1)  # Get predicted class

            all_preds.extend(predicted.cpu().numpy())  # Store all predictions
            all_targets.extend(targets.cpu().numpy())  # Store all Truth labels

            total += targets.size(0)

            correct += (predicted == targets).sum().item()

    # Compute average loss, accuracy, precision, recall and f1 measure

    avg_loss = test_loss / len(test_loader)

    accuracy = 100.0 * correct / total
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')

    classes = ('airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks')
    # Build confusion Matrix
    cf_matrix = confusion_matrix(all_targets, all_preds)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('./MLP/main_mlp.png')

    print("\n")
    print("MAIN MLP MODEL EVALUATIONS:")
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    #MLP with 3 hidden layers

    mlp_model = nn.Sequential(

        nn.Linear(50, 512),  # Input layer

        nn.ReLU(),  # Activation function

        nn.Linear(512, 512),  # Hidden layer

        nn.BatchNorm1d(512),  # Batch Normalization

        nn.ReLU(),  # Activation function

        nn.Linear(512, 512),  # Hidden layer

        nn.BatchNorm1d(512),  # Batch Normalization

        nn.ReLU(),  # Activation function

        nn.Linear(512, 512),  # Hidden layer

        nn.BatchNorm1d(512),  # Batch Normalization

        nn.ReLU(),  # Activation function

        nn.Linear(512, 10)  # Output layer

    )

    # loss function

    criterion = nn.CrossEntropyLoss()

    # Dataloader for train
    train_loader = torch.utils.data.DataLoader(

        dataset=torch.utils.data.TensorDataset(torch.tensor(train_features_pca), torch.tensor(train_labels_np)),

        batch_size=500,

        shuffle=False

    )

    if (os.path.isfile("./MLP/mlp_model_3HL.pth")):

        # Load the trained weights

        mlp_model.load_state_dict(torch.load("./MLP/mlp_model_3HL.pth", weights_only=True))

    else:

        # optimizer (SGD with momentum)

        optimizer = optim.SGD(mlp_model.parameters(), lr=0.01, momentum=0.9)

        # training loop

        num_epochs = 30

        # Training loop

        for epoch in range(num_epochs):

            mlp_model.train()  # Set model to training mode

            epoch_loss = 0.0

            for inputs, targets in train_loader:
                # Forward pass

                outputs = mlp_model(inputs)

                loss = criterion(outputs, targets)

                # Backward pass

                optimizer.zero_grad()  # Clear gradients

                loss.backward()  # Compute gradients

                optimizer.step()  # Update weights

                epoch_loss += loss.item()

            # Print epoch stats

            #print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

        # Save the trained model

        torch.save(mlp_model.state_dict(), "./MLP/mlp_model_3HL.pth")

        print("Model saved after training.")

    # Testing Model

    test_loader = torch.utils.data.DataLoader(

        dataset=torch.utils.data.TensorDataset(torch.tensor(test_features_pca), torch.tensor(test_labels_np)),

        batch_size=500,

        shuffle=False

    )

    mlp_model.eval()  # Set model to evaluation mode

    test_loss = 0.0

    correct = 0

    total = 0

    all_preds = []

    all_targets = []

    with torch.no_grad():  # Disable gradient computation

        for inputs, targets in test_loader:
            outputs = mlp_model(inputs)  # Forward pass

            loss = criterion(outputs, targets)  # Compute loss

            test_loss += loss.item()  # total loss

            # Compute accuracy
            _, predicted = outputs.max(1)  # Get predicted class

            all_preds.extend(predicted.cpu().numpy())  # Store all predictions
            all_targets.extend(targets.cpu().numpy())  # Store all Truth labels

            total += targets.size(0)

            correct += (predicted == targets).sum().item()

    # Compute average loss, accuracy, precision, recall and f1 measure

    avg_loss = test_loss / len(test_loader)

    accuracy = 100.0 * correct / total
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')

    classes = ('airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks')
    # Build confusion Matrix
    cf_matrix = confusion_matrix(all_targets, all_preds)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('./MLP/mlp_3HL.png')

    print("\n")
    print("MLP MODEL WITH 3 HIDDEN LAYERS:")
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # MLP with 5 hidden layers

    mlp_model = nn.Sequential(

        nn.Linear(50, 512),  # Input layer

        nn.ReLU(),  # Activation function

        nn.Linear(512, 512),  # Hidden layer

        nn.BatchNorm1d(512),  # Batch Normalization

        nn.ReLU(),  # Activation function

        nn.Linear(512, 512),  # Hidden layer

        nn.BatchNorm1d(512),  # Batch Normalization

        nn.ReLU(),  # Activation function

        nn.Linear(512, 512),  # Hidden layer

        nn.BatchNorm1d(512),  # Batch Normalization

        nn.ReLU(),  # Activation function

        nn.Linear(512, 512),  # Hidden layer

        nn.BatchNorm1d(512),  # Batch Normalization

        nn.ReLU(),  # Activation function

        nn.Linear(512, 512),  # Hidden layer

        nn.BatchNorm1d(512),  # Batch Normalization

        nn.ReLU(),  # Activation function

        nn.Linear(512, 10)  # Output layer

    )

    # loss function

    criterion = nn.CrossEntropyLoss()

    # Dataloader for train
    train_loader = torch.utils.data.DataLoader(

        dataset=torch.utils.data.TensorDataset(torch.tensor(train_features_pca), torch.tensor(train_labels_np)),

        batch_size=500,

        shuffle=False

    )

    if (os.path.isfile("./MLP/mlp_model_5HL.pth")):

        # Load the trained weights

        mlp_model.load_state_dict(torch.load("./MLP/mlp_model_5HL.pth", weights_only=True))

    else:

        # optimizer (SGD with momentum)

        optimizer = optim.SGD(mlp_model.parameters(), lr=0.01, momentum=0.9)

        # training loop

        num_epochs = 30

        # Training loop

        for epoch in range(num_epochs):

            mlp_model.train()  # Set model to training mode

            epoch_loss = 0.0

            for inputs, targets in train_loader:
                # Forward pass

                outputs = mlp_model(inputs)

                loss = criterion(outputs, targets)

                # Backward pass

                optimizer.zero_grad()  # Clear gradients

                loss.backward()  # Compute gradients

                optimizer.step()  # Update weights

                epoch_loss += loss.item()

            # Print epoch stats

            #print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

        # Save the trained model

        torch.save(mlp_model.state_dict(), "./MLP/mlp_model_5HL.pth")

        print("Model saved after training.")

    # Testing Model

    test_loader = torch.utils.data.DataLoader(

        dataset=torch.utils.data.TensorDataset(torch.tensor(test_features_pca), torch.tensor(test_labels_np)),

        batch_size=500,

        shuffle=False

    )

    mlp_model.eval()  # Set model to evaluation mode

    test_loss = 0.0

    correct = 0

    total = 0

    all_preds = []

    all_targets = []

    with torch.no_grad():  # Disable gradient computation

        for inputs, targets in test_loader:
            outputs = mlp_model(inputs)  # Forward pass

            loss = criterion(outputs, targets)  # Compute loss

            test_loss += loss.item()  # total loss

            # Compute accuracy
            _, predicted = outputs.max(1)  # Get predicted class

            all_preds.extend(predicted.cpu().numpy())  # Store all predictions
            all_targets.extend(targets.cpu().numpy())  # Store all Truth labels

            total += targets.size(0)

            correct += (predicted == targets).sum().item()

    # Compute average loss, accuracy, precision, recall and f1 measure

    avg_loss = test_loss / len(test_loader)

    accuracy = 100.0 * correct / total
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')

    classes = ('airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks')
    # Build confusion Matrix
    cf_matrix = confusion_matrix(all_targets, all_preds)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('./MLP/mlp_5HL.png')

    print("\n")
    print("MLP MODEL WITH 5 HIDDEN LAYERS:")
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Varying size of hidden layer - 1024

    mlp_model = nn.Sequential(

        nn.Linear(50, 1024),  # Input layer

        nn.ReLU(),  # Activation function

        nn.Linear(1024, 1024),  # Hidden layer

        nn.BatchNorm1d(1024),  # Batch Normalization

        nn.Linear(1024, 10)  # Output layer

    )

    # loss function

    criterion = nn.CrossEntropyLoss()

    # Dataloader for train
    train_loader = torch.utils.data.DataLoader(

        dataset=torch.utils.data.TensorDataset(torch.tensor(train_features_pca), torch.tensor(train_labels_np)),

        batch_size=500,

        shuffle=False

    )

    if (os.path.isfile("./MLP/mlp_model_1024.pth")):

        # Load the trained weights

        mlp_model.load_state_dict(torch.load("./MLP/mlp_model_1024.pth", weights_only=True))

    else:

        # optimizer (SGD with momentum)

        optimizer = optim.SGD(mlp_model.parameters(), lr=0.01, momentum=0.9)

        # training loop

        num_epochs = 30

        # Training loop

        for epoch in range(num_epochs):

            mlp_model.train()  # Set model to training mode

            epoch_loss = 0.0

            for inputs, targets in train_loader:
                # Forward pass

                outputs = mlp_model(inputs)

                loss = criterion(outputs, targets)

                # Backward pass

                optimizer.zero_grad()  # Clear gradients

                loss.backward()  # Compute gradients

                optimizer.step()  # Update weights

                epoch_loss += loss.item()

            # Print epoch stats

            #print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

        # Save the trained model

        torch.save(mlp_model.state_dict(), "./MLP/mlp_model_1024.pth")

        print("Model saved after training.")

    # Testing Model

    test_loader = torch.utils.data.DataLoader(

        dataset=torch.utils.data.TensorDataset(torch.tensor(test_features_pca), torch.tensor(test_labels_np)),

        batch_size=500,

        shuffle=False

    )

    mlp_model.eval()  # Set model to evaluation mode

    test_loss = 0.0

    correct = 0

    total = 0

    all_preds = []

    all_targets = []

    with torch.no_grad():  # Disable gradient computation

        for inputs, targets in test_loader:
            outputs = mlp_model(inputs)  # Forward pass

            loss = criterion(outputs, targets)  # Compute loss

            test_loss += loss.item()  # total loss

            # Compute accuracy
            _, predicted = outputs.max(1)  # Get predicted class

            all_preds.extend(predicted.cpu().numpy())  # Store all predictions
            all_targets.extend(targets.cpu().numpy())  # Store all Truth labels

            total += targets.size(0)

            correct += (predicted == targets).sum().item()

    # Compute average loss, accuracy, precision, recall and f1 measure

    avg_loss = test_loss / len(test_loader)

    accuracy = 100.0 * correct / total
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')

    classes = ('airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks')
    # Build confusion Matrix
    cf_matrix = confusion_matrix(all_targets, all_preds)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('./MLP/mlp_1024.png')

    print("\n")
    print("MLP MODEL WITH 5 HIDDEN LAYERS AND VARYING SIZE OF HIDDEN LAYER:")
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Varying size of hidden layer - 64

    mlp_model = nn.Sequential(

        nn.Linear(50, 64),  # Input layer

        nn.ReLU(),  # Activation function

        nn.Linear(64, 64),  # Hidden layer

        nn.BatchNorm1d(64),  # Batch Normalization

        nn.Linear(64, 10)  # Output layer

    )

    # loss function

    criterion = nn.CrossEntropyLoss()

    # Dataloader for train
    train_loader = torch.utils.data.DataLoader(

        dataset=torch.utils.data.TensorDataset(torch.tensor(train_features_pca), torch.tensor(train_labels_np)),

        batch_size=500,

        shuffle=False

    )

    if (os.path.isfile("./MLP/mlp_model_64.pth")):

        # Load the trained weights

        mlp_model.load_state_dict(torch.load("./MLP/mlp_model_64.pth", weights_only=True))

    else:

        # optimizer (SGD with momentum)

        optimizer = optim.SGD(mlp_model.parameters(), lr=0.01, momentum=0.9)

        # training loop

        num_epochs = 30

        # Training loop

        for epoch in range(num_epochs):

            mlp_model.train()  # Set model to training mode

            epoch_loss = 0.0

            for inputs, targets in train_loader:
                # Forward pass

                outputs = mlp_model(inputs)

                loss = criterion(outputs, targets)

                # Backward pass

                optimizer.zero_grad()  # Clear gradients

                loss.backward()  # Compute gradients

                optimizer.step()  # Update weights

                epoch_loss += loss.item()

            # Print epoch stats

            #print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

        # Save the trained model

        torch.save(mlp_model.state_dict(), "./MLP/mlp_model_64.pth")

        print("Model saved after training.")

    # Testing Model

    test_loader = torch.utils.data.DataLoader(

        dataset=torch.utils.data.TensorDataset(torch.tensor(test_features_pca), torch.tensor(test_labels_np)),

        batch_size=500,

        shuffle=False

    )

    mlp_model.eval()  # Set model to evaluation mode

    test_loss = 0.0

    correct = 0

    total = 0

    all_preds = []

    all_targets = []

    with torch.no_grad():  # Disable gradient computation

        for inputs, targets in test_loader:
            outputs = mlp_model(inputs)  # Forward pass

            loss = criterion(outputs, targets)  # Compute loss

            test_loss += loss.item()  # total loss

            # Compute accuracy
            _, predicted = outputs.max(1)  # Get predicted class

            all_preds.extend(predicted.cpu().numpy())  # Store all predictions
            all_targets.extend(targets.cpu().numpy())  # Store all Truth labels

            total += targets.size(0)

            correct += (predicted == targets).sum().item()

    # Compute average loss, accuracy, precision, recall and f1 measure

    avg_loss = test_loss / len(test_loader)

    accuracy = 100.0 * correct / total
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')

    classes = ('airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks')
    # Build confusion Matrix
    cf_matrix = confusion_matrix(all_targets, all_preds)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('./MLP/mlp_64.png')

    print("\n")
    print("MLP MODEL WITH 5 HIDDEN LAYERS AND VARYING SIZE OF HIDDEN LAYER:")
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")





