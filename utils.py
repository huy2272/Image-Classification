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
