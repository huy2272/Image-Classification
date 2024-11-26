import torch
from sklearn.decomposition import PCA


def select_n_img(dataset, nb_train_images=500, nb_test_images=100):
    train_class_counts = {i: 0 for i in range(10)}
    test_class_counts = {i: 0 for i in range(10)}
    selected_train_dataset = []
    selected_test_dataset = []
    for img, label in dataset["train"]:
        if train_class_counts[label] < nb_train_images:
            selected_train_dataset.append((img, label))
            train_class_counts[label] += 1

    for img, label in dataset["test"]:
        if test_class_counts[label] < nb_test_images:
            selected_test_dataset.append((img, label))
            test_class_counts[label] += 1

    return selected_train_dataset, selected_test_dataset


def extract_feature_vectors(modified_model, dataloaders, device):
    train_features_list = []
    train_labels_list = []
    test_features_list = []
    test_labels_list = []

    pca = PCA(n_components=50)

    for images, labels in dataloaders["train"]:
        with torch.no_grad():
            # features.shape = [500, 512, 1, 1]
            features = modified_model(images.to(device))
        # Reshape to (batch_size, num_features) = (500, 512)
        features = features.to("cpu").reshape(features.size(0), -1)
        train_features_list.append(features)
        train_labels_list.append(labels)

    train_features = torch.cat(train_features_list, dim=0)
    train_labels = torch.cat(train_labels_list, dim=0)

    for images, labels in dataloaders["test"]:
        with torch.no_grad():
            # features.shape = [500, 512, 1, 1]
            features = modified_model(images.to(device))
            # Reshape to (batch_size, num_features) = (100, 512)
            features = features.to("cpu").reshape(features.size(0), -1)
            test_features_list.append(features)
            test_labels_list.append(labels)

    test_features = torch.cat(test_features_list, dim=0)
    test_labels = torch.cat(test_labels_list, dim=0)

    train_features_np = train_features.numpy()
    test_features_np = test_features.numpy()
    train_labels_np = train_labels.numpy()
    test_labels_np = test_labels.numpy()

    train_features_pca = pca.fit_transform(train_features_np)
    test_features_pca = pca.transform(test_features_np)

    return train_features_pca, test_features_pca, train_labels_np, test_labels_np
