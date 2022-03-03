import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from data.data_processing import ModelDataContainer
from csv import writer
from models.point_ML_model import PointModel
from models.tcn_ML_model import TCNModel

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat


def train_model(model, training_data, testing_data, lr, epochs=10, test_interval=0, tcn=False, print_updates=False):
    """
    Contains the training loop for an ML model and returns final accuracy of model
    :param model: ML model made using PyTorch NN module
    :param training_data: DataLoader made from PyTorch DataLoader containing training dataset
    :param testing_data: DataLoader made from PyTorch DataLoader containing testing dataset
    :param lr: float value for learning rate of model
    :param epochs: int epochs training loop runs for
    :param test_interval: int between 0 and batch_size * epochs, interval of iterations after which ML model must be tested
    :param tcn: boolean of whether model is a TCN or not
    :param print_updates: boolean of whether training loop should print updates to terminal for user
    :return: float acc
    """
    if print_updates:
        print(f'Beginning training with {epochs} epochs at learning rate of {lr}')
    criterion = nn.CrossEntropyLoss()  # Loss function for model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Model optimization function
    acc = 0
    for epoch in range(epochs):
        model.train()
        for batch, (train_data, train_labels) in enumerate(training_data):
            output_labels = model(train_data)
            loss = criterion(output_labels, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if test_interval > 0 and (epoch * len(training_data) + batch + 1) % test_interval == 0:
                # Test model when number of iterations have been reached
                acc, f1 = test_model(model, testing_data, tcn=tcn, print_updates=False)
                if print_updates:
                    print(f"Batch {batch + 1}/{len(training_data)} of the {epoch + 1} out of {epochs} epochs: "
                          f"accuracy of {acc}, f1 of {f1}")
    if print_updates:
        update = "Training completed"
        if test_interval:
            update += f" with accuracy of {acc}"
        print(update)
    return acc


def test_model(model, testing_set, tcn=False, print_updates=True):
    """
    Testing function that tests model and returns accuracy and f1 score of model and its predictions
    :param model: ML model made using PyTorch NN module
    :param testing_set: DataLoader made from PyTorch DataLoader containing testing dataset
    :param tcn: boolean of whether model is a TCN or not
    :param print_updates: boolean of whether function should print updates to terminal for user
    :return: tuple containing float accuracy, float f1 score
    """
    model.eval()
    out_pred = []
    out_true = []
    with torch.no_grad():
        for inputs, labels in testing_set:
            output = model(inputs)  # Get model prediction
            output = (output.argmax(dim=1, keepdim=True)[0]).numpy()
            out_pred.extend(output)  # Save prediction
            labels = labels.numpy()
            out_true.extend(labels)  # Save actual labels
    acc = accuracy_score(out_true, out_pred)
    f1 = f1_score(out_true, out_pred, average='weighted')
    if print_updates:
        print(f"Accuracy of {acc}%. F1 Score of {f1}\n")
    return acc, f1


def validate_model(model, validation_set, classes, tcn=False, print_updates=True):
    """
    Validates model against a validation set and builds necessary data for confusion matrix (only difference between
    this and testing is the confusion matrix
    :param model: ML model made using PyTorch NN module
    :param validation_set: DataLoader made from PyTorch DataLoader containing validation dataset
    :param classes: Tuple where each value is a string that matches with the target indexes
    :param tcn: boolean of whether model is a TCN or not
    :param print_updates: boolean of whether function should print updates to terminal for user
    :return: tuple containing pandas dataframe with confusion matrix data, float accuracy, float f1 score
    """
    model.eval()
    out_pred = []
    out_true = []
    with torch.no_grad():
        for inputs, labels in validation_set:
            output = model(inputs)  # Get model prediction
            output = (output.argmax(dim=1, keepdim=True)[0]).numpy()
            out_pred.extend(output)  # Save prediction
            labels = labels.numpy()
            out_true.extend(labels)  # Save actual labels
    # Build pandas data structure for confusion matrix
    conf_mat = confusion_matrix(out_true, out_pred, normalize='all')
    df_cm = pd.DataFrame(conf_mat / np.sum(conf_mat) * 10, index=[i for i in classes],
                         columns=[i for i in classes])
    acc = accuracy_score(out_true, out_true)
    f1 = f1_score(out_true, out_pred, average='weighted')
    return df_cm, acc, f1


def measure_model(model, model_data, tcn=False):
    """
    Measures average accuracy of model at each time step
    :param model: ML model made using PyTorch NN module
    :param model_data: ModelDataContainer constructed with data being measured
    :param tcn: boolean of whether model is a TCN or not
    :return: An array containing the average accuracy of the model at each timestep
    """
    model.eval()
    # Get correct data needed for model depending on model type (ANN vs. TCN)
    if not tcn:
        data = model_data.x_point
        labels = model_data.y_point
        tests, samples = model_data.y.shape

    else:
        data = model_data.x_time
        labels = model_data.y_time
        tests, samples = model_data.y.shape
        samples = samples - model_data.time_steps

    output_predictions = np.empty((tests, samples))
    output_true = np.empty(output_predictions.shape)

    x = torch.from_numpy(data.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.int64))

    out_acc = []
    data_set = TensorDataset(x, y)
    data_loader = DataLoader(dataset=data_set)
    with torch.no_grad():
        for ind, (inputs, labels) in enumerate(data_loader):
            output = model(inputs)
            output = (output.argmax(dim=1, keepdim=True)[0]).numpy()  # Get model prediction
            output_predictions[ind % tests, ind % samples] = output  # Save prediction
            output_true[ind % tests, ind % samples] = labels  # Save actual labels
    for i in range(samples):
        true = output_true[:, i]
        pred = output_predictions[:, i]
        out_acc.append((np.sum(true == pred)) / len(true))  # Find average accuracy across tests at given time step
    return out_acc


def acc_buckets(data_array, bucket_size):
    """
    Used in conjuction with measure_model to average accuracy across several timesteps
    :param data_array: array of containing accuracy of model across each timestep (can be constructed with measure_model)
    :param bucket_size: int of how many time steps should be average together (ex. 10 => accuracy across every ten time steps averaged together)
    :return: array of average accuracy of model for each section of timesteps
    """
    buckets_array = []
    least_len = float("inf")  # needed to make sure resulting array is not ragged
    for model in data_array:
        bucket = []
        for i in range(0, len(model), bucket_size):
            section = model[i: i + bucket_size]
            avg_acc = sum(section) / len(section)
            bucket.append(avg_acc)
        least_len = min(least_len, len(bucket))
        buckets_array.append(bucket)
    # Make each measurement same length (end of tests are generally not needed since steady state has been reached)
    for i in range(len(buckets_array)):
        buckets_array[i] = buckets_array[i][:least_len]
    return buckets_array


def reset_params(model):
    """
    Void function that resets parameters of model
    :param model: ML model made using PyTorch NN module
    :return: None
    """
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    return


def k_fold_training(model, model_data, k, batch_size, lr, epochs=10, tcn=False, print_updates=False):
    """
    Function that executes a k-fold cross validation of ML model
    :param model: ML model made using PyTorch NN module
    :param model_data: ModelDataContainer constructed with data being used for model training
    :param k: int number of folds for validation
    :param batch_size: int number of samples in each batch trained on before an optimization step
    :param lr: float value for learning rate of model
    :param epochs: int epochs training loop runs for
    :param tcn: boolean of whether model is a TCN or not
    :param print_updates:  boolean of whether function should print updates to terminal for user
    :return: float average accuracy across the folds, average f1 score across the folds
    """
    k_acc = []
    k_f1 = []
    # get list of k tuples containing training and testing set
    cv_set = model_data.create_k_fold_val(k, batch_size, tcn=tcn)
    for fold, (training_set, testing_set) in enumerate(cv_set):
        if print_updates:
            print(f"Beginning {fold + 1} fold out of {len(cv_set)}")
        train_model(model, training_set, testing_set, lr, epochs=epochs, tcn=tcn, print_updates=print_updates)
        acc, f1 = test_model(model, testing_set, tcn=tcn, print_updates=print_updates)
        k_acc.append(acc)
        k_f1.append(f1)
        reset_params(model)  # reset model weights to make sure previous testing does not influence results of next fold
    avg_acc = sum(k_acc) / len(k_acc)
    avg_f1 = sum(k_f1) / len(k_f1)
    if print_updates:
        print(f"Across of {k} folds, average accuracy of {avg_acc} and average F1 of {avg_f1}")
    return avg_acc, avg_f1


def point_model_grid_search(model_data, range_nodes, batch_size, learning_rate, epochs=50, print_updates=False):
    """
    Void function that conducts a grid search for an ANN model and constructs csv's of parameter results, confusion
    matrices, and model measurements
    :param model_data: ModelDataContainer constructed with data being used for model training
    :param range_nodes: tuple containing range of nodes to be swept in same structure as python range function
    (min, max, step)
    :param batch_size: int number of samples in each batch trained on before an optimization step
    :param learning_rate: float value for learning rate of model
    :param epochs: int epochs training loop runs for
    :param print_updates: boolean of whether function should print updates to terminal for user
    :return: NONE
    """
    nodes_min, nodes_max, nodes_step = range_nodes
    measure_array = []
    for num_nodes_in_hl in range(nodes_min, nodes_max, nodes_step):
        model = PointModel(num_nodes_in_hl)
        if print_updates:
            print(f"\n--------------------Trying model with {num_nodes_in_hl} in hidden layer-----------------------")
        # conduct kfold validation of model
        final_acc, final_f1 = k_fold_training(model, model_data, 5, batch_size, learning_rate, epochs, False,
                                              print_updates)
        if print_updates:
            print(
                f"---------Average accuracy of {final_acc} and f1 of {final_f1} for model with {num_nodes_in_hl} hidden nodes-----------\n")
        _, testing_set = model_data.create_train_test(test_size=0, batch_size=batch_size, tcn=False)
        # get confusion matrix
        df_cm, _, _ = validate_model(model, testing_set, model_data.classes, tcn=False, print_updates=print_updates)
        # measure model
        measurement = measure_model(model, model_data)
        measure_array.append(measurement)
        model_features = (num_nodes_in_hl, final_acc * 100, final_f1)
        # construct confusion matrix for model performance
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig(f'data/Graphs/ann_{num_nodes_in_hl}.png')
        plt.close()
        if print_updates:
            print(f"finished model of {num_nodes_in_hl}")
        # save PyTorch model for later use
        torch.save(model.state_dict(),
                   f"models/saved_models/"
                   f"point_model_{num_nodes_in_hl}.pt")
        # add hyper parameter performance to csv
        with open('data/point_models_params.csv', 'a') as f:
            csv_writer = writer(f)
            csv_writer.writerow(model_features)
            f.close()
    # save time step accuracy measurements as matlab array
    measure_matrix = np.asarray(acc_buckets(measure_array, 10))
    mdic = {"point_data": measure_matrix}
    savemat("point_matrix.mat", mdic)
    return


def tcn_model_grid_search(model_data, time_step_range, kernel_sizes, out_channels_range, batch_size, learning_rate,
                          epochs=50,
                          print_updates=False):
    """
    Void function that conducts a grid search for a TCN model and constructs csv's of parameter results, confusion
    matrices, and model measurements
    :param model_data: ModelDataContainer constructed with data being used for model training
    :param time_step_range: tuple containing range of time steps to be swept in same structure as python range function
    (min, max, step)
    :param kernel_sizes: tuple containing range of convolving kernel sizes to be swept in same structure as python range
    function (min, max, step)
    :param out_channels_range: tuple containing range of output channels to be swept in same structure as python range
    function (min, max, step)
    :param batch_size: int number of samples in each batch trained on before an optimization step
    :param learning_rate: float value for learning rate of model
    :param epochs: int epochs training loop runs for
    :param print_updates: boolean of whether function should print updates to terminal for user
    :return: None
    """
    time_min, time_max, time_step_step = time_step_range
    kernel_min, kernel_max, kernel_step = kernel_sizes
    out_min, out_max, out_step = out_channels_range
    measure_array = []
    for output_channels in range(out_min, out_max, out_step):
        for time_steps in range(time_min, time_max, time_step_step):
            for kernel_size in range(kernel_min, kernel_max, kernel_step):
                # kernel size cannot be greater than number of time steps for TCN model
                if kernel_size > time_steps:
                    continue
                if print_updates:
                    print(f"\n---------------Trying model with {output_channels} output channels, {time_steps} time "
                          f"steps, and kernel size of {kernel_size}---------------")

                model = TCNModel(kernel_size, time_steps, output_channels)
                model_data.create_time_series_data(time_steps)
                # conduct kfold validation of model
                final_acc, final_f1 = k_fold_training(model, model_data, 5, batch_size, learning_rate, epochs, True,
                                                      print_updates)
                if print_updates:
                    print(
                        f"---------Average accuracy of {final_acc} and f1 of {final_f1} for model with {time_steps} time "
                        f"steps and {kernel_size} kernel size-----------\n")
                # get confusion matrix
                _, testing_set = model_data.create_train_test(test_size=0, batch_size=batch_size, tcn=True)
                df_cm, _, _ = validate_model(model, testing_set, model_data.classes, tcn=True,
                                             print_updates=print_updates)
                # measure model
                measurement = measure_model(model, model_data, tcn=True)
                measure_array.append(measurement)
                model_features = (time_steps, kernel_size, output_channels, final_acc, final_f1)
                # construct confusion matrix for model performance
                plt.figure(figsize=(12, 7))
                sn.heatmap(df_cm, annot=True)
                plt.savefig(f'data/Graphs/tcn_{time_steps}_{kernel_size}.png')
                plt.close()
                # save PyTorch model for later use
                torch.save(model.state_dict(),
                           f"models/saved_models/"
                           f"tcn_model_{kernel_size}_{time_steps}.pt")
                # add hyper parameter performance to csv
                with open('data/tcn_models_params_with_channels.csv', 'a') as f:
                    csv_writer = writer(f)
                    csv_writer.writerow(model_features)
                    f.close()
    # save time step accuracy measurements as matlab array
    measure_matrix = np.asarray(acc_buckets(measure_array, 10))
    mdic = {"tcn_data": measure_matrix}
    savemat("tcn_matrix.mat", mdic)
    return


def main():
    # Create data container for data needed for model
    classes = ('Toluene', 'M-Xylene', 'Ethylbenzene', 'Methanol', 'Ethanol')
    model_data = ModelDataContainer("data/data6.mat", "data6", classes)

    # Needed for both grid searches
    batch_size = 100
    learning_rate = .01
    epochs = 50

    # ANN grid search param
    num_nodes_in_hl = (1, 21, 1)

    # TCN grid search param
    time_step_range = (2, 11, 1)
    kernel_size = (2, 11, 1)
    output_channels = (1, 7, 1)

    point_search = False
    tcn_search = True

    if point_search:
        print("Beginning point by point model grid search\n Please do not close window.")
        point_model_grid_search(model_data, num_nodes_in_hl, batch_size, learning_rate, epochs=epochs,
                                print_updates=True)
        print("Finished with point to point grid search \n")

    if tcn_search:
        print("Beginning TCN model grid search\n")
        tcn_model_grid_search(model_data, time_step_range, kernel_size, output_channels, batch_size, learning_rate,
                              epochs=epochs, print_updates=True)
        print("Finished with TCN model grid search\n")

    print("Window can be closed.")


if __name__ == "__main__":
    main()
