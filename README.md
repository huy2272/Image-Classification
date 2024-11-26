# Image-Classification 

Mohammad Usman 40170784

Khanh Huy Nguyen 40125396

## Description

`main.py`: Where we will be running our program. This file contains code on transforming and loading our data from CIFAR-10.

`utils.py`: Contains utility functions: select_n_img() and extract_feature_vectors(). `select_n_img()` will select 500 training images per class and 100 testing images per class. `extract_feature_vectors()` takes in the data along with a model and will generate the PCA-reduced feature vectors.

`naive_bayes.py`: Contains the Naive Bayes algorithm implementation.

`decision_tree.py`: Contains the decision tree algorithm implementation and a help function to plot the calculated metrics (training accuracy, testing accuracy, precision, recall and f1-score).

`Decision_Tree.png`: A graph plotting the metrics (training accuracy, testing accuracy, precision, recall and f1-score) against the max_depths of the local implemented decision tree.

`Scikit_Decision_Tree.png`: A graph plotting the metrics (training accuracy, testing accuracy, precision, recall and f1-score) against the max_depths of the Scikit's decision tree.

`models/`: A folder containing all saved trained models.

`.gitignore`: Ignores folders `cifar-10-batches-py/` and `__pycache__/` since we don't want to commit these.

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

![image](https://github.com/user-attachments/assets/82182a84-b0e0-48d6-bd3a-35ec4e1a540e)

5. Using the pretrained RestNet-18 model, we removed the last layer (restnet18.fc) to use it as a fixed feature extractor.
6. We then pass in the modified model, the dataloaders and our device (cuda:0 or cpu) to the helper function `extract_feature_vectors` to get our PCA-reduced training feature vectors, testing feature vectors, training labels, testing labels.

### Training, evaluating and applying the models

#### Naive Bayes
![image](https://github.com/user-attachments/assets/0b404885-f324-44eb-900a-6461bd4e0033)

1. Train

   To train our locally implemented model. Uncomment line `70-72`. This will train the model and save it in `./models/gaussian_naive_bayes.pth`
   To train Scikit's model. Uncomment line `82-84`. This will train the model and save it in `./models/scikit_gaussian_naive_bayes.pth`
3. Evaluate

   To evaluate the models and get our metrics, we are using the helper function `get_naive_bayes_metrics()`. This function simply calculates the metrics and print them to the console. We can choose which model we want to use via the param `get_naive_bayes_metrics(..., useScikit = True/False)`, if `True` we will use Scikit's model, `False` we will use our locally implemented model.

   ![image](https://github.com/user-attachments/assets/6129f8e7-2d76-42db-9a27-8905a3bb4ecc)

5. Apply
   
   To apply the model, we simply load a trained model and pass in our test feature vectors `test_features_pca`. We can then calculate our metrics using the predictions the model returns `nb_predictions`. To get the model prediction we are using `.predict()`

#### Decision Tree

![image](https://github.com/user-attachments/assets/19dfa6f3-b952-4664-a6d4-373ff22cce14)


1. Train

   To train our locally implemented model. Uncomment line `95-100`. Since we are training this model at different `max_depth` we will save it in `./models/decision_tree_model_{depth}.pth`
   To train Scikit's model. Uncomment line `112-117`. Since we are training this model at different `max_depth` we will save it in `./models/scikit_decision_tree_model_{depth}.pth`
   
3. Evaluate

   To evaluate the models and get our metrics, we are using the helper function `plot_decision_tree_metrics()`. This function calculates the metrics and plot them (metrics vs tree depth). We can choose which model we want to use via the param `plot_decision_tree_metrics(..., useScikit = True/False)`, if `True` we will use Scikit's model, `False` we will use our locally implemented model.

   ![image](https://github.com/user-attachments/assets/c519e3f9-1822-46ce-9b79-8e8a934f1345)


5. Apply
   
   To apply the model, we simply load our trained models into a list. We can then iterate over this list to calculate our metrics for each model. Each model metrics are then store inside the following lists: `train_accuracies, test_accuracies, precisions, recalls, f1_scores`. For example, `train_accuracies[0]` will store the training accuracy for decision tree model with `max_depth = 10`. `train_accuracies[len(loaded_dtcs)]` will store the training accuracy for decision tree model with `max_depth = 50`. To get the model prediction we are using `.predict()`
