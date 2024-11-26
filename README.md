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

