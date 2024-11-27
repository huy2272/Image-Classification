import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt


class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.means = {}
        self.variances = {}
        self.priors = {}

    def fit(self, train_features, labels):
        """
        Train the Gaussian Naive Bayes model.
        """
        self.classes = np.unique(labels)  # Get unique class labels
        num_samples = train_features.shape[0]

        for c in self.classes:
            # Filter data by class
            features_labels = train_features[labels == c]

            # Compute mean, variance, and prior for each class
            self.means[c] = np.mean(features_labels, axis=0)
            self.variances[c] = np.var(features_labels, axis=0)
            self.priors[c] = features_labels.shape[0] / num_samples

    def _gaussian_density(self, x, mean, var):
        """
        Compute the Gaussian density function for given mean and variance.
        """
        eps = 1e-9  # Small value to prevent division by zero
        var = var + eps
        coeff = 1 / np.sqrt(2 * np.pi * var)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return coeff * exponent

    def predict(self, test_features):
        """
        Predict the class labels for the input data.
        """
        posteriors = []
        for test_feature in test_features:
            class_posteriors = []
            for c in self.classes:
                prior = np.log(self.priors[c])  # Log of prior probability
                likelihood = np.sum(
                    np.log(
                        self._gaussian_density(
                            test_feature, self.means[c], self.variances[c]
                        )
                    )
                )
                posterior = prior + likelihood
                class_posteriors.append(posterior)
            posteriors.append(class_posteriors)
        return np.array([self.classes[np.argmax(p)] for p in posteriors])


def get_naive_bayes_metrics(
    train_features_pca,
    train_labels_np,
    test_features_pca,
    test_labels_np,
    useScikit=False,
):
    if useScikit:
        loaded_nb = torch.load("./models/scikit_gaussian_naive_bayes.pth")
    else:
        loaded_nb = torch.load("./models/gaussian_naive_bayes.pth")

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

    print(f"Scikit: {useScikit}. Training Accuracy = {nb_train_accuracy * 100:.2f}%")
    print(f"Scikit: {useScikit}. Test Accuracy = {nb_test_accuracy * 100:.2f}%")
    print(f"Scikit: {useScikit}. Precision = {nb_precision}")
    print(f"Scikit: {useScikit}. Recall = {nb_recall}")
    print(f"Scikit: {useScikit}. f1 = {nb_f1}")
    gnb_conf_matrix = confusion_matrix(test_labels_np, nb_predictions)
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    disp = ConfusionMatrixDisplay(
        confusion_matrix=gnb_conf_matrix, display_labels=classes
    )
    disp.plot(cmap=plt.cm.Blues, colorbar=False)
    plt.title(f"Confusion Matrix - GaussianNaiveBayes - Scikit: {useScikit}")
    plt.show()
