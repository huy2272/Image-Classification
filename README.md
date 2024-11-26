# Image-Classification [WIP]

COMP 472 Project

Mohammad Usman 40170784

Khanh Huy Nguyen 40125396

## Description

`main.py`: Where we will be running our program. This file contains code on transforming and loading our data from CIFAR-10.

`utils.py`: Contains utility functions: select_n_img() and extract_feature_vectors(). select_n_img() will select 500 training images per class and 100 testing images per class.

`naive_bayes.py`: Contains the Naive Bayes algorithm implementation.

`decision_tree.py`: Contains the decision tree algorithm implementation and a help function to plot the calculated metrics (training accuracy, testing accuracy, precision, recall and f1-score).

`Decision_Tree.png`: A graph plotting the metrics (training accuracy, testing accuracy, precision, recall and f1-score) against the max_depths of the local implemented decision tree.

`Scikit_Decision_Tree.png`: A graph plotting the metrics (training accuracy, testing accuracy, precision, recall and f1-score) against the max_depths of the Scikit's decision tree.

`models/`: A folder containing all saved trained models.

## Getting Started

To run the program, simply run the script `main.py` in Interactive Window. This will ensure that the graphs are generated.

![image](https://github.com/user-attachments/assets/64bbc406-6c47-4f64-b54c-9c52c7e0ff2f)

### Image preprocessing
Running the `main.py` script will automatically preprocess the images. Below is a detailed explanation:

![image](https://github.com/user-attachments/assets/f9594dd2-cc1d-4be5-a3bc-12d082bd96b0)

1. `image-datasets` is how we are getting our datasets, we are saving the image batches in our root directory since `data_dir = "./`. We make sure to check whether we have already downloaded the dataset before.
2. `data_transforms` contains instructions on how we are transforming our images. In this case for both `train` and `test` we are resizing these images to 224x224 and then normalizing them.

![image](https://github.com/user-attachments/assets/49b1c304-ee5e-49df-9fee-cc7388946d8e)

3. We then use the utility function `select_n_img` to select the first 500 images of each class for the training datasets, and 100 images of each class for the testing datasets.
4. We then use `dataloaders` to load data into our model, with training batch_size = 500 and testing batch_size=100.

