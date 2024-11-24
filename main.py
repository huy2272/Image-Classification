import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

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
        # Use the first 500 training images
        "train": torchvision.datasets.CIFAR10(
            data_dir, train=True, download=True, transform=data_transforms["train"]
        ),
        # Use the first 100 testing images
        "test": torchvision.datasets.CIFAR10(
            data_dir, train=False, download=True, transform=data_transforms["test"]
        ),
    }

    dataloaders = {
        "train": torch.utils.data.DataLoader(
            image_datasets["train"], batch_size=500, shuffle=True, num_workers=4
        ),
        "test": torch.utils.data.DataLoader(
            image_datasets["test"], batch_size=100, shuffle=True, num_workers=4
        ),
    }

    # pre-trained ResNet-18 CNN
    pretrained_model = models.resnet18(pretrained=True)
    # remove the last layer of ResNet-18
    modified_model = nn.Sequential(*list(pretrained_model.children())[:-1])

    # We need to set requires_grad = False to freeze the parameters
    for param in pretrained_model.parameters():
        param.requires_grad = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pretrained_model = pretrained_model.to(device)

    criterion = nn.CrossEntropyLoss()

    # We are optimizing all the layers, final layer should have been removed
    optimizer_conv = optim.SGD(pretrained_model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    pca = PCA(n_components=50)

    inputs, labels = next(iter(dataloaders["train"]))
    # [500 x 3 x 224 x 224] tensor inputs
    # [batch_size x channels x height x width]
    inputs = inputs.to(device)
    # [500 x 512 x 1 x 1] tensor outputs
    train_features = modified_model(inputs)
    # Reshaping here because PCA expected array dimension <= 2.
    train_features = train_features.to("cpu").reshape(500, 1 * 512)
    # pca_train_dataset.shape = (500, 50)
    pca_train_dataset = pca.fit_transform(train_features)
