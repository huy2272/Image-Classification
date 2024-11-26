import numpy as np


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
