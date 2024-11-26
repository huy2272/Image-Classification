import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, models, transforms
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score
from naive_bayes import GaussianNaiveBayes
from utils import extract_feature_vectors, select_n_img
import numpy as np
from sklearn import tree
from decision_tree import DecisionTree
import matplotlib.pyplot as plt

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
            selected_train_dataset, batch_size=500, shuffle=False, num_workers=4
        ),
        "test": torch.utils.data.DataLoader(
            selected_test_dataset, batch_size=100, shuffle=False, num_workers=4
        ),
    }

    pretrained_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
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

    # 3: Naive Bayes
    # Training the Gaussian Naive Bayes model implementation
    # gnb = GaussianNaiveBayes()
    # gnb.fit(train_features_pca, train_labels_np)
    # torch.save(gnb, "gaussian_naive_bayes.pth")
    loaded_gnb = torch.load("gaussian_naive_bayes.pth")

    # Train accuracy
    train_predictions = loaded_gnb.predict(train_features_pca)
    train_accuracy = np.mean(train_predictions == train_labels_np)

    # Calculate Test Accuracy
    predictions = loaded_gnb.predict(test_features_pca)
    test_accuracy = np.mean(predictions == test_labels_np)
    precision = precision_score(
        test_labels_np, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(
        test_labels_np, predictions, average="weighted", zero_division=0
    )
    f1 = f1_score(test_labels_np, predictions, average="weighted", zero_division=0)

    print(f"Training Accuracy = {train_accuracy* 100:.2f}%")
    print(f"Test Accuracy = {test_accuracy * 100:.2f}%")
    print(f"Precision = {precision}")
    print(f"Recall = {recall}%")
    print(f"f1 = {f1}%")

    # Scikit's Gaussian Naive Bayes
    # nb_model = GaussianNB()
    # nb_model.fit(train_features_pca, train_labels_np)
    # torch.save(nb_model, "scikit_gaussian_naive_bayes.pth")
    loaded_nb = torch.load("scikit_gaussian_naive_bayes.pth")

    # Train accuracy
    nb_train_predictions = loaded_nb.predict(train_features_pca)
    nb_train_accuracy = np.mean(nb_train_predictions == train_labels_np)

    # Calculate Test Accuracy
    nb_predictions = loaded_nb.predict(test_features_pca)
    nb_test_accuracy = np.mean(nb_predictions == test_labels_np)
    nb_precision = precision_score(
        test_labels_np, nb_predictions, average="weighted", zero_division=0
    )
    nb_recall = recall_score(
        test_labels_np, nb_predictions, average="weighted", zero_division=0
    )
    nb_f1 = f1_score(
        test_labels_np, nb_predictions, average="weighted", zero_division=0
    )

    print(f"Training Accuracy = {nb_train_accuracy * 100:.2f}%")
    print(f"Test Accuracy = {nb_test_accuracy * 100:.2f}%")
    print(f"Precision = {nb_precision}")
    print(f"Recall = {nb_recall}")
    print(f"f1 = {nb_f1}")

    # 3: Decision Tree
    tree_depths = range(10, 55, 5)
    train_accuracies, test_accuracies, precisions, recalls, f1_scores = (
        [],
        [],
        [],
        [],
        [],
    )
    for depth in tree_depths:
        # Train the decision tree with the current depth
        dt = DecisionTree(max_depth=depth)
        dt.fit(train_features_pca, train_labels_np)
        torch.save(dt, f"decision_tree_model_{depth}.pth")

        # Predictions on the test set
        test_predictions = dt.predict(test_features_pca)

        # Calculate Training Accuracy
        train_predictions = dt.predict(train_features_pca)
        train_accuracy = np.mean(train_predictions == train_labels_np)
        train_accuracies.append(train_accuracy)
        print(f"Depth {depth}: Training Accuracy = {train_accuracy:.4f}")

        # Calculate Test Accuracy
        test_accuracy = np.mean(test_predictions == test_labels_np)
        test_accuracies.append(test_accuracy)
        print(f"Depth {depth}: Test Accuracy = {test_accuracy:.4f}")
        precision = precision_score(
            test_labels_np, test_predictions, average="weighted", zero_division=0
        )
        recall = recall_score(
            test_labels_np, test_predictions, average="weighted", zero_division=0
        )
        f1 = f1_score(
            test_labels_np, test_predictions, average="weighted", zero_division=0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    plt.figure(figsize=(12, 8))

    # Accuracy Plot
    plt.plot(
        tree_depths,
        train_accuracies,
        label="Training Accuracy",
        marker="o",
        markersize=6,
    )
    plt.plot(
        tree_depths, test_accuracies, label="Test Accuracy", marker="o", markersize=6
    )

    # Precision, Recall, F1-Measure Plots
    plt.plot(tree_depths, precisions, label="Precision", marker="x", markersize=6)
    plt.plot(tree_depths, recalls, label="Recall", marker="x", markersize=6)
    plt.plot(tree_depths, f1_scores, label="F1-Measure", marker="x", markersize=6)

    plt.xlabel("Tree Depth")
    plt.ylabel("Metrics")
    plt.title("Tree Depth vs Metrics (Multiples of 5)")
    plt.legend()
    plt.grid()
    plt.show()
