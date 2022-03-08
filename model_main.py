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
import time


def train_model(model, training_data, testing_data, lr, epochs=10, test_interval=0, print_updates=False):
    """
    Contains the training loop for an ML model and returns final accuracy of model
    :param model: ML model made using PyTorch NN module
    :param training_data: DataLoader made from PyTorch DataLoader containing training dataset
    :param testing_data: DataLoader made from PyTorch DataLoader containing testing dataset
    :param lr: float value for learning rate of model
    :param epochs: int epochs training loop runs for
    :param test_interval: int between 0 and batch_size * epochs, interval of iterations after which ML model is tested
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
                acc, f1 = test_model(model, testing_data, print_updates=False)
                if print_updates:
                    print(f"Batch {batch + 1}/{len(training_data)} of the {epoch + 1} out of {epochs} epochs: "
                          f"accuracy of {acc}, f1 of {f1}")
    if print_updates:
        update = "Training completed"
        if test_interval:
            update += f" with accuracy of {acc}"
        print(update)
    return acc


def test_model(model, testing_set, print_updates=True):
    """
    Testing function that tests model and returns accuracy and f1 score of model and its predictions
    :param model: ML model made using PyTorch NN module
    :param testing_set: DataLoader made from PyTorch DataLoader containing testing dataset
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


def validate_model(model, validation_set, classes):
    """
    Validates model against a validation set and builds necessary data for confusion matrix (only difference between
    this and testing is the confusion matrix
    :param model: ML model made using PyTorch NN module
    :param validation_set: DataLoader made from PyTorch DataLoader containing validation dataset
    :param classes: Tuple where each value is a string that matches with the target indexes
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


def measure_model(model, model_data, data_loader, tcn=False):
    """
    Measures average accuracy of model at each time step
    :param model: ML model made using PyTorch NN module
    :param model_data: ModelDataContainer constructed with data being measured
    :param data_loader: data loader from testing set, must be 2d in testing and then chronological order
    :param tcn: boolean of whether model is a TCN or not
    :return: An array containing the average accuracy of the model at each timestep
    """
    model.eval()
    # Get correct data needed for model depending on model type (ANN vs. TCN)
    tests = model_data.y.shape[0]
    samples = len(data_loader) // tests

    output_predictions = np.empty((tests, samples))
    output_true = np.empty(output_predictions.shape)

    out_acc = []
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
    :param data_array: array of containing accuracy of model across each timestep (constructed with measure_model)
    :param bucket_size: int of how many time steps should be average together
    (ex. 10 => accuracy across every ten time steps averaged together)
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
    model.eval()
    with torch.no_grad():
        for sequential in model.layers:
            for layer in sequential:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
        for layer in model.classification:
            if hasattr(layer, "reset_parameters"):
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
    measure_array = []
    k_acc = []
    k_f1 = []
    # get list of k tuples containing training and testing set
    cv_set = model_data.create_k_fold_val(k, batch_size, tcn=tcn)
    for fold, (training_set, testing_set) in enumerate(cv_set):
        if print_updates:
            print(f"Beginning {fold + 1} fold out of {len(cv_set)}")
        train_model(model, training_set, testing_set, lr, epochs=epochs, print_updates=print_updates)
        acc, f1 = test_model(model, testing_set, print_updates=print_updates)
        k_acc.append(acc)
        k_f1.append(f1)
        measure_array.append(measure_model(model, model_data, testing_set, tcn))
        reset_params(model)  # reset model weights to make sure previous testing does not influence results of next fold
    avg_acc = sum(k_acc) / len(k_acc)
    avg_f1 = sum(k_f1) / len(k_f1)
    avg_acc_array = acc_buckets(measure_array, 10//k)
    if print_updates:
        print(f"Across of {k} folds, average accuracy of {avg_acc} and average F1 of {avg_f1}")
    return avg_acc, avg_f1, avg_acc_array


def point_model_grid_search(model_data, input_size, range_nodes, range_layers, batch_size, learning_rate, epochs=50,
                            print_updates=False, file_base_name=None, save_models=False):
    """
    Void function that conducts a grid search for an ANN model and constructs csv's of parameter results, confusion
    matrices, and model measurements
    :param model_data: ModelDataContainer constructed with data being used for model training
    :param input_size: int number of input variables
    :param range_nodes: tuple containing range of nodes to be swept in same structure as python range function
    (min, max, step)
    :param range_layers: tuple containing range of hidden layers to be swept in same structure as python range function
    (min, max, step)
    :param batch_size: int number of samples in each batch trained on before an optimization step
    :param learning_rate: float value for learning rate of model
    :param epochs: int epochs training loop runs for
    :param print_updates: boolean of whether function should print updates to terminal for user
    :param file_base_name: string for base name for saved files made in function
    :param save_models: boolean if modelss made during grid search should be saved
    :return: None
    """
    if file_base_name is None:
        file_base_name = "ann"
    nodes_min, nodes_max, nodes_step = range_nodes
    node_range = range(nodes_min, nodes_max, nodes_step)
    layers_min, layers_max, layers_step = range_layers
    layer_range = range(layers_min, layers_max, layers_step)
    total_iter = len(node_range) * len(layer_range)
    n_iter = 0
    measure_array = []

    start_time = time.time()
    for num_hid_layers in range(layers_min, layers_max, layers_step):
        for num_nodes_in_hl in range(nodes_min, nodes_max, nodes_step):
            model = PointModel(num_nodes_in_hl, num_hidden_layers=num_hid_layers, input_size=input_size)
            if print_updates:
                print(f"\n--------------------Trying model with {num_nodes_in_hl} nodes "
                      f"in {num_hid_layers} hidden layer-----------------------")
                n_iter += 1
            # conduct kfold validation of model
            final_acc, final_f1, measurement = k_fold_training(model, model_data, 5, batch_size, learning_rate, epochs, False,
                                                  print_updates)
            measure_array.append(measurement)
            if print_updates:
                print(
                    f"---------Average accuracy of {final_acc} and f1 of {final_f1} for model with {num_nodes_in_hl} "
                    f"hidden nodes-----------\n")
            _, testing_set = model_data.create_train_test(test_size=0, batch_size=batch_size, tcn=False)
            # get confusion matrix
            df_cm, _, _ = validate_model(model, testing_set, model_data.classes)
            model_features = (num_hid_layers, num_nodes_in_hl, final_acc * 100, final_f1)
            # construct confusion matrix for model performance
            plt.figure(figsize=(12, 7))
            sn.heatmap(df_cm, annot=True)
            plt.savefig(f'data/Graphs/{file_base_name}_{num_nodes_in_hl}_{num_hid_layers}.png')
            plt.close()
            # save PyTorch model for later use
            if save_models:
                torch.save(model.state_dict(),
                           f"models/saved_models/"
                           f"{file_base_name}_model_{num_nodes_in_hl}_{num_hid_layers}.pt")
            # add hyper parameter performance to csv
            with open(f'data/ParameterData/{file_base_name}_models_params.csv', 'a') as f:
                csv_writer = writer(f)
                csv_writer.writerow(model_features)
                f.close()
            # provide time prediction
            if print_updates:
                print(f"finished model of {num_nodes_in_hl} with {num_hid_layers} hidden layers")
                end_time = time.time()
                time_taken = end_time - start_time
                taken_hours = time_taken // 3600
                taken_min = (time_taken % 3600) // 60
                time_predicted = time_taken / n_iter * (total_iter - n_iter)
                pred_hours = time_predicted // 3600
                pred_min = (time_predicted % 3600) // 60
                print(f"{taken_hours} hr {taken_min} min to try {n_iter} models. "
                      f"Predicted {pred_hours} hr {pred_min} min left for grid search\n")
    # save time step accuracy measurements as matlab array
    measure_matrix = np.asarray(measure_array)
    mdic = {f"{file_base_name}_data": measure_matrix}
    savemat(f"TimeMeasurement/{file_base_name}_matrix.mat", mdic)
    return


def tcn_model_grid_search(model_data, input_size, time_step_range, kernel_sizes, out_channels_range, conv_layers_range,
                          batch_size, learning_rate, epochs=50, print_updates=False, file_base_name=None, save_models=False):
    """
    Void function that conducts a grid search for a TCN model and constructs csv's of parameter results, confusion
    matrices, and model measurements
    :param conv_layers_range:
    :param model_data: ModelDataContainer constructed with data being used for model training
    :param input_size: int number of input variables
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
    :param file_base_name: string for base name for saved files made in function
    :param save_models: boolean if modelss made during grid search should be saved
    :return: None
    """
    if file_base_name is None:
        file_base_name = "tcn"
    time_min, time_max, time_step_step = time_step_range
    time_range = range(time_min, time_max, time_step_step)
    kernel_min, kernel_max, kernel_step = kernel_sizes
    kernel_range = range(kernel_min, kernel_max, kernel_step)
    out_min, out_max, out_step = out_channels_range
    out_range = range(out_min, out_max, out_step)
    layers_min, layers_max, layers_step = conv_layers_range
    layer_range = range(layers_min, layers_max, layers_step)
    total_iter = len(time_range) * len(kernel_range) * len(out_range) * len(layer_range)
    n_iter = 0

    measure_array = []
    start_time = time.time()
    for num_conv_layers in layer_range:
        for output_channels in out_range:
            for time_steps in time_range:
                for kernel_size in kernel_range:
                    # kernel size cannot be greater than number of time steps for TCN model
                    if kernel_size > time_steps:
                        total_iter -= 1
                        continue
                    if print_updates:
                        print(f"\n---------------Trying model with {output_channels} output channels and {num_conv_layers} layers, {time_steps} time "
                              f"steps, and kernel size of {kernel_size}---------------")
                        n_iter += 1

                    model = TCNModel(kernel_size, time_steps, output_channels, num_conv_layers=num_conv_layers, input_size=input_size)
                    model_data.create_time_series_data(time_steps)
                    # conduct kfold validation of model
                    final_acc, final_f1, measurement = k_fold_training(model, model_data, 5, batch_size, learning_rate, epochs, True,
                                                          print_updates)
                    measure_array.append(measurement)
                    if print_updates:
                        print(
                            f"---------Average accuracy of {final_acc} and f1 of {final_f1} for "
                            f"model with {output_channels} output channels and {num_conv_layers} layers, {time_steps} "
                            f"time steps, and kernel size of {kernel_size}-----------\n")

                    # get confusion matrix
                    _, testing_set = model_data.create_train_test(test_size=0, batch_size=batch_size, tcn=True)
                    df_cm, _, _ = validate_model(model, testing_set, model_data.classes)
                    model_features = (time_steps, kernel_size, output_channels, num_conv_layers, final_acc, final_f1)
                    # construct confusion matrix for model performance
                    plt.figure(figsize=(12, 7))
                    sn.heatmap(df_cm, annot=True)
                    plt.savefig(f'data/Graphs/{file_base_name}_{time_steps}_{kernel_size}_{output_channels}_{num_conv_layers}.png')
                    plt.close()
                    # save PyTorch model for later use
                    if save_models:
                        torch.save(model.state_dict(),
                                   f"models/saved_models/"
                                   f"{file_base_name}_model_{time_steps}_{kernel_size}_{output_channels}_{num_conv_layers}.pt")
                    # add hyper parameter performance to csv
                    with open(f'data/ParameterData/{file_base_name}_models_params.csv', 'a') as f:
                        csv_writer = writer(f)
                        csv_writer.writerow(model_features)
                        f.close()
                    # provide time prediction
                    if print_updates:
                        print(f"finished model with {output_channels} output channels and {num_conv_layers} layers,"
                              f" {time_steps} time steps, and kernel size of {kernel_size}")
                        end_time = time.time()
                        time_taken = end_time - start_time
                        taken_hours = time_taken // 3600
                        taken_min = (time_taken % 3600) // 60
                        time_predicted = time_taken / n_iter * (total_iter - n_iter)
                        pred_hours = time_predicted // 3600
                        pred_min = (time_predicted % 3600) // 60
                        print(f"{taken_hours} hr {taken_min} min to try {n_iter} models. "
                              f"Predicted {pred_hours} hr {pred_min} min left for grid search\n")
    # save time step accuracy measurements as matlab array
    measure_matrix = np.asarray(measure_array)
    mdic = {f"{file_base_name}_data": measure_matrix}
    savemat(f"TimeMeasurement/{file_base_name}_matrix.mat", mdic)
    return


def main():
    # Create data container for data needed for model
    print("Loading in data")
    classes = ('Toluene', 'M-Xylene', 'Ethylbenzene', 'Methanol', 'Ethanol')
    input_size = 9
    model_data = ModelDataContainer("data/DataContainers/one_conc_matrix.mat", "one_conc_matrix", classes, num_samples=1346, input_vars=input_size)
    print("Data loaded")

    # Needed for both grid searches
    batch_size = 100
    learning_rate = .01
    epochs = 5

    # ANN grid search param
    num_nodes_in_hl = (5, 10, 1)
    num_hidden_layers = (1, 2, 1)

    # TCN grid search param
    time_step_range = (6, 11, 2)
    kernel_size = (2, 11, 1)
    output_channels = (4, 7, 1)
    conv_layers_range = (1, 2, 1)

    point_search = False
    file_point_name = "fixed_ann_1c"
    tcn_search = True
    file_tcn_name = "fixed_tcn_1c"

    if point_search:
        print("Beginning point by point model grid search\n Please do not close window.")
        point_model_grid_search(model_data, input_size, num_nodes_in_hl, num_hidden_layers, batch_size, learning_rate,
                                epochs=epochs, print_updates=True, file_base_name=file_point_name)
        print("Finished with point to point grid search \n")

    if tcn_search:
        print("Beginning TCN model grid search\n")
        tcn_model_grid_search(model_data, input_size, time_step_range, kernel_size, output_channels, conv_layers_range,
                              batch_size, learning_rate, epochs=epochs, print_updates=True, file_base_name=file_tcn_name)
        print("Finished with TCN model grid search\n")

    print("Window can be closed.")


if __name__ == "__main__":
    main()
