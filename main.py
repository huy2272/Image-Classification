from sklearn import tree
import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, models, transforms
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score
from naive_bayes import GaussianNaiveBayes, get_naive_bayes_metrics
from utils import extract_feature_vectors, select_n_img
import numpy as np
from decision_tree import DecisionTree, plot_decision_tree_metrics


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
    # torch.save(gnb, "./models/gaussian_naive_bayes.pth")
    get_naive_bayes_metrics(
        train_features_pca,
        train_labels_np,
        test_features_pca,
        test_labels_np,
        useScikit=False,
    )
    # Scikit's Gaussian Naive Bayes
    # Code to train the model
    # nb_model = GaussianNB()
    # nb_model.fit(train_features_pca, train_labels_np)
    # torch.save(nb_model, "./models/scikit_gaussian_naive_bayes.pth")
    get_naive_bayes_metrics(
        train_features_pca,
        train_labels_np,
        test_features_pca,
        test_labels_np,
        useScikit=True,
    )

    # 4: Decision Tree
    # Training code
    # tree_depths = range(10, 55, 5)
    # for depth in tree_depths:
    #     # Train the decision tree with the current depth
    #     dt = DecisionTree(max_depth=depth)
    #     dt.fit(train_features_pca, train_labels_np)
    #     torch.save(dt, f"./models/decision_tree_model_{depth}.pth")

    # Generate metrics for local decision tree implementation and plot the results on a graph
    plot_decision_tree_metrics(
        train_features_pca,
        train_labels_np,
        test_features_pca,
        test_labels_np,
        useScikit=False,
    )

    # Training code
    # tree_depths = range(10, 55, 5)
    # for depth in tree_depths:
    #     dtc = tree.DecisionTreeClassifier(criterion="gini", max_depth=depth)
    #     # Train the decision tree with the current depth
    #     dtc.fit(train_features_pca, train_labels_np)
    #     torch.save(dtc, f"./models/scikit_decision_tree_model_{depth}.pth")

    # Generate metrics for Scikit's decision tree implementation and plot the results on a graph
    plot_decision_tree_metrics(
        train_features_pca,
        train_labels_np,
        test_features_pca,
        test_labels_np,
        useScikit=True,
    )
