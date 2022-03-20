import torch
import torch.nn as nn
from data.data_processing import ModelDataContainer
from csv import writer
from models.ANN_ML_model import PointModel
from models.Conv1D_ML_model import Conv1D_Model
from models.TCN_model import TCN_Model

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
    optimizer = torch.optim.NAdam(model.parameters(), lr=lr)  # Model optimization function
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, epochs // 10)
    acc = 0
    for epoch in range(epochs):
        model.train()
        for batch, (train_data, train_labels) in enumerate(training_data):
            optimizer.zero_grad()
            output_labels = model(train_data)
            loss = criterion(output_labels, train_labels)
            loss.backward()
            optimizer.step()
        if test_interval > 0 and (epoch + 1) % test_interval == 0:
            # Test model when number of iterations have been reached
            acc, f1 = test_model(model, testing_data, print_updates=False)
            if print_updates:
                print(f"{epoch + 1} out of {epochs} epochs: accuracy of {acc}, f1 of {f1}")
        lr_scheduler.step(epoch)

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


def measure_model(model, model_data, data_loader):
    """
    Measures average accuracy of model at each time step
    :param model: ML model made using PyTorch NN module
    :param model_data: ModelDataContainer constructed with data being measured
    :param data_loader: data loader from testing set, must be 2d in testing and then chronological order
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
    least_len = float("inf")  # needed to make sure resulting array is not jagged
    for model in data_array:
        least_len = min(least_len, len(model))

    for i in range(0, least_len, bucket_size):
        sum_section = 0
        len_section = 0
        for model in data_array:
            section = model[i: i + bucket_size]
            sum_section += sum(section)
            len_section += len(section)
        avg_acc = sum_section / len_section
        buckets_array.append(avg_acc)

    return buckets_array


def fix_jagged(data_array):
    fixed_array = []
    least_len = float("inf")
    for model in data_array:
        least_len = min(least_len, len(model))
    for model in data_array:
        fixed_array.append(model[:least_len])
    return fixed_array


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
        model.train()
        if print_updates:
            print(f"Beginning {fold + 1} fold out of {len(cv_set)}")
        train_model(model, training_set, testing_set, lr, epochs=epochs, print_updates=print_updates)
        acc, f1 = test_model(model, testing_set, print_updates=print_updates)
        k_acc.append(acc)
        k_f1.append(f1)
        measure_array.append(measure_model(model, model_data, testing_set))
        # reset model weights to make sure previous testing does not influence results of next fold
        model.reset_params()

    avg_acc = sum(k_acc) / len(k_acc)
    avg_f1 = sum(k_f1) / len(k_f1)
    avg_acc_array = acc_buckets(measure_array, 10 // k)

    if print_updates:
        print(f"Across of {k} folds, average accuracy of {avg_acc} and average F1 of {avg_f1}")
    return avg_acc, avg_f1, avg_acc_array


def ann_model_grid_search(model_data, input_size, range_nodes, range_layers, batch_size, learning_rate, epochs=50,
                          print_updates=False, file_base_name=None, make_conf_mat=False, save_models=False):
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
    :param make_conf_mat: boolean if confusion matrix should be made for models
    :param save_models: boolean if models made during grid search should be saved
    :return: None
    """
    if file_base_name is None:
        file_base_name = "ann"
    num_outputs = len(model_data.classes)
    total_iter = len(range_nodes) * len(range_layers)
    n_iter = 0
    f = open(f'data/ParameterData/{file_base_name}_models_params.csv', 'w', newline='')
    csv_writer = writer(f)
    measure_array = []
    start_time = time.time()
    for num_hid_layers in range_layers:
        for num_nodes_in_hl in range_nodes:

            model = PointModel(num_nodes_in_hl, num_hid_layers, input_size, num_outputs)
            if print_updates:
                print(f"\n--------------------Trying model with {num_nodes_in_hl} nodes "
                      f"in {num_hid_layers} hidden layer-----------------------")
            n_iter += 1

            # conduct kfold validation of model
            final_acc, final_f1, measurement = k_fold_training(model, model_data, 5, batch_size, learning_rate, epochs,
                                                               False,
                                                               print_updates)
            measure_array.append(measurement)
            model_features = (num_hid_layers, num_nodes_in_hl, final_acc, final_f1)
            if print_updates:
                print(
                    f"---------Average accuracy of {final_acc} and f1 of {final_f1} for model with {num_nodes_in_hl} "
                    f"hidden nodes-----------\n")

            # get confusion matrix
            if make_conf_mat:
                _, testing_set = model_data.create_train_test(test_size=0, batch_size=batch_size, tcn=False)
                df_cm, _, _ = validate_model(model, testing_set, model_data.classes)
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
                csv_writer.writerow(model_features)

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
                      f"Predicted {pred_hours} hr {pred_min} min left for {total_iter - n_iter} "
                      f"models in grid search\n")

    f.close()
    # save time step accuracy measurements as matlab array
    measure_array = fix_jagged(measure_array)
    measure_matrix = np.asarray(measure_array)
    mdic = {f"{file_base_name}_data": measure_matrix}
    savemat(f"data/TimeMeasurement/{file_base_name}_matrix.mat", mdic)
    return


def conv1d_model_grid_search(model_data, input_size, time_step_range, kernel_sizes, out_channels_range,
                             conv_layers_range,
                             batch_size, learning_rate, epochs=50, print_updates=False, file_base_name=None,
                             make_conf_mat=False, save_models=False):
    """
    Void function that conducts a grid search for a 1D CNN model and constructs csv's of parameter results, confusion
    matrices, and model measurements
    :param model_data: ModelDataContainer constructed with data being used for model training
    :param input_size: int number of input variables
    :param time_step_range: iterable containing range of time steps to be swept
    :param kernel_sizes: iterable containing range of convolving kernel sizes to be swept
    :param out_channels_range: iterable containing range of output channels to be swept
    :param conv_layers_range: iterable containing range of convolving layers to be swept
    :param batch_size: int number of samples in each batch trained on before an optimization step
    :param learning_rate: float value for learning rate of model
    :param epochs: int epochs training loop runs for
    :param print_updates: boolean of whether function should print updates to terminal for user
    :param file_base_name: string for base name for saved files made in function
    :param make_conf_mat: boolean if confusion matrix should be made for models
    :param save_models: boolean if models made during grid search should be saved
    :return: None
    """
    if file_base_name is None:
        file_base_name = "conv1d"
    num_outputs = len(model_data.classes)
    total_iter = len(time_step_range) * len(kernel_sizes) * len(out_channels_range) * len(conv_layers_range)
    for num_conv_layers in conv_layers_range:
        for output_channels in out_channels_range:
            for time_steps in time_step_range:
                for kernel_size in kernel_sizes:
                    if kernel_size > time_steps:
                        total_iter -= 1
    n_iter = 0
    f = open(f'data/ParameterData/{file_base_name}_models_params.csv', 'w', newline='')
    csv_writer = writer(f)
    measure_array = []
    start_time = time.time()
    for num_conv_layers in conv_layers_range:
        for output_channels in out_channels_range:
            for time_steps in time_step_range:
                for kernel_size in kernel_sizes:
                    # kernel size cannot be greater than number of time steps for 1DConv model
                    if kernel_size > time_steps:
                        continue
                    if print_updates:
                        print(
                            f"\n---------------Trying model with {output_channels} output channels and {num_conv_layers} layers, {time_steps} time "
                            f"steps, and kernel size of {kernel_size}---------------")
                    n_iter += 1

                    model = Conv1D_Model(kernel_size, time_steps, output_channels, input_size, num_conv_layers,
                                         num_outputs)
                    model_data.create_time_series_data(time_steps)
                    # conduct kfold validation of model
                    final_acc, final_f1, measurement = k_fold_training(model, model_data, 5, batch_size, learning_rate,
                                                                       epochs, True,
                                                                       print_updates)
                    measure_array.append(measurement)
                    model_features = (num_conv_layers, output_channels, time_steps, kernel_size, final_acc, final_f1)

                    if print_updates:
                        print(
                            f"---------Average accuracy of {final_acc} and f1 of {final_f1} for "
                            f"model with {output_channels} output channels and {num_conv_layers} layers, {time_steps} "
                            f"time steps, and kernel size of {kernel_size}-----------\n")

                    # add hyper parameter performance to csv
                    csv_writer.writerow(model_features)

                    if make_conf_mat:
                        # get confusion matrix
                        _, testing_set = model_data.create_train_test(test_size=0, batch_size=batch_size, tcn=True)
                        df_cm, _, _ = validate_model(model, testing_set, model_data.classes)
                        # construct confusion matrix for model performance
                        plt.figure(figsize=(12, 7))
                        sn.heatmap(df_cm, annot=True)
                        plt.savefig(
                            f'data/Graphs/{file_base_name}_{time_steps}_{kernel_size}_{output_channels}_{num_conv_layers}.png')
                        plt.close()

                    # save PyTorch model for later use
                    if save_models:
                        torch.save(model.state_dict(),
                                   f"models/saved_models/"
                                   f"{file_base_name}_model_{time_steps}_{kernel_size}_{output_channels}_{num_conv_layers}.pt")

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
                              f"Predicted {pred_hours} hr {pred_min} min left for {total_iter - n_iter} "
                              f"models in grid search\n")

    f.close()
    # save time step accuracy measurements as matlab array
    measure_array = fix_jagged(measure_array)
    measure_matrix = np.asarray(measure_array)
    mdic = {f"{file_base_name}_data": measure_matrix}
    savemat(f"data/TimeMeasurement/{file_base_name}_matrix.mat", mdic)
    return


def tcn_model_grid_search(model_data, input_size, time_step_range, kernel_sizes, filter_channel_range, dil_base_range,
                          batch_size, learning_rate, epochs=50, print_updates=False, file_base_name=None,
                          make_conf_mat=False, save_models=False):
    """
    Void function that conducts a grid search for a TCN model and constructs csv's of parameter results, confusion
    matrices, and model measurements
    :param model_data: ModelDataContainer constructed with data being used for model training
    :param input_size: int number of input variables
    :param time_step_range: iterable containing range of time steps to be swept
    :param kernel_sizes: iterable containing range of convolving kernel sizes to be swept
    :param filter_channel_range: iterable containing range of number of filter channels to be swept
    :param dil_base_range: iterable containing range of dilation bases to be swept
    :param batch_size: int number of samples in each batch trained on before an optimization step
    :param learning_rate: float value for learning rate of model
    :param epochs: int epochs training loop runs for
    :param print_updates: boolean of whether function should print updates to terminal for user
    :param file_base_name: string for base name for saved files made in function
    :param make_conf_mat: boolean if confusion matrix should be made for models
    :param save_models: boolean if models made during grid search should be saved
    :return: None
    """
    if file_base_name is None:
        file_base_name = "tcn"

    num_outputs = len(model_data.classes)
    n_iter = 0
    total_iter = len(time_step_range) * len(kernel_sizes) * len(filter_channel_range) * len(dil_base_range)
    for dilation in dil_base_range:
        for filter_size in filter_channel_range:
            for time_steps in time_step_range:
                for kernel_size in kernel_sizes:
                    if kernel_size > time_steps or kernel_size < dilation:
                        total_iter -= 1
    f = open(f'data/ParameterData/{file_base_name}_models_params.csv', 'w', newline='')
    csv_writer = writer(f)
    measure_array = []
    start_time = time.time()
    for dilation in dil_base_range:
        for filter_size in filter_channel_range:
            for time_steps in time_step_range:
                for kernel_size in kernel_sizes:
                    if kernel_size > time_steps or kernel_size < dilation:
                        continue

                    if print_updates:
                        print(f"\n---------------Trying model with {filter_size} filter channels and {dilation} "
                              f"dilation base, {time_steps} time steps, and kernel size of {kernel_size}--------------")
                    n_iter += 1

                    model = TCN_Model(kernel_size, time_steps, input_size, num_outputs, filter_size,
                                      dilation_base=dilation)
                    model_data.create_time_series_data(time_steps)
                    # conduct kfold validation of model
                    final_acc, final_f1, measurement = k_fold_training(model, model_data, 5, batch_size, learning_rate,
                                                                       epochs, True,
                                                                       print_updates)
                    measure_array.append(measurement)
                    model_features = (dilation, filter_size, time_steps, kernel_size, final_acc, final_f1)

                    if print_updates:
                        print(
                            f"---------Average accuracy of {final_acc} and f1 of {final_f1} for model with"
                            f" {filter_size} filter channels and {dilation} dilation base, {time_steps} time steps, "
                            f"and kernel size of {kernel_size}-----------\n")

                    # add hyper parameter performance to csv
                    csv_writer.writerow(model_features)

                    if make_conf_mat:
                        # get confusion matrix
                        _, testing_set = model_data.create_train_test(test_size=0, batch_size=batch_size, tcn=True)
                        df_cm, _, _ = validate_model(model, testing_set, model_data.classes)
                        # construct confusion matrix for model performance
                        plt.figure(figsize=(12, 7))
                        sn.heatmap(df_cm, annot=True)
                        plt.savefig(
                            f'data/Graphs/{file_base_name}_{time_steps}_{kernel_size}_{filter_size}_{dilation}.png')
                        plt.close()

                    # save PyTorch model for later use
                    if save_models:
                        torch.save(model.state_dict(),
                                   f"models/saved_models/"
                                   f"{file_base_name}_model_{time_steps}_{kernel_size}_{filter_size}_{dilation}.pt")

                    # provide time prediction
                    if print_updates:
                        print(f"finished model with {filter_size} filter channels and {dilation} dilation base,"
                              f" {time_steps} time steps, and kernel size of {kernel_size}")
                        end_time = time.time()
                        time_taken = end_time - start_time
                        taken_hours = time_taken // 3600
                        taken_min = (time_taken % 3600) // 60
                        time_predicted = time_taken / n_iter * (total_iter - n_iter)
                        pred_hours = time_predicted // 3600
                        pred_min = (time_predicted % 3600) // 60
                        print(f"{taken_hours} hr {taken_min} min to try {n_iter} models. "
                              f"Predicted {pred_hours} hr {pred_min} min left for {total_iter - n_iter} "
                              f"models in grid search\n")
    f.close()
    # save time step accuracy measurements as matlab array
    measure_array = fix_jagged(measure_array)
    measure_matrix = np.asarray(measure_array)
    mdic = {f"{file_base_name}_data": measure_matrix}
    savemat(f"data/TimeMeasurement/{file_base_name}_matrix.mat", mdic)
    return


def main():
    # Create data container for data needed for model
    print("Loading in data")
    classes = ('Toluene', 'M-Xylene', 'Ethylbenzene', 'Methanol', 'Ethanol')
    input_size = 9
    data_file = "data/DataContainers/all_conc_matrix.mat"
    matrix_name = "all_conc_matrix"
    model_data = ModelDataContainer(data_file, classes, matrix_name, num_samples=1346, input_vars=input_size)
    print("Data loaded")

    # Needed for both grid searches
    batch_size = 500
    learning_rate = .01
    epochs = 150

    # ANN grid search param
    num_nodes_in_hl = list(range(1, 15, 1)) + list(range(15, 30, 5)) + list(range(30, 51, 10))
    num_hidden_layers = range(1, 3, 1)
    point_search = False
    file_point_name = "ann_allc_big_grid"
    if point_search:
        print("Beginning point by point model grid search\n Please do not close window.")
        ann_model_grid_search(model_data, input_size, num_nodes_in_hl, num_hidden_layers, batch_size, learning_rate,
                              epochs=epochs, print_updates=True, file_base_name=file_point_name)
        print("Finished with point to point grid search \n")

    # Conv1D grid search param
    time_step_range = range(7, 14, 2)
    kernel_size = range(3, 8, 2)
    output_channels = range(5, 12, 2)
    conv_layers_range = range(1, 3, 1)
    conv1d_search = True
    file_conv_name = "conv1d_allc_big_grid"
    if conv1d_search:
        print("Beginning Conv1D model grid search\n")
        conv1d_model_grid_search(model_data, input_size, time_step_range, kernel_size, output_channels,
                                 conv_layers_range, batch_size, learning_rate, epochs=epochs, print_updates=True,
                                 file_base_name=file_conv_name)
        print("Finished with Conv1D model grid search\n")

    # TCN grid search param
    time_step_range = range(3, 14, 2)
    kernel_size = range(2, 7, 1)
    filter_channels = range(5, 12, 2)
    dilation_bases = range(2, 3, 1)
    tcn_search = True
    file_tcn_name = "tcn_allc_big_grid"
    if tcn_search:
        print("Beginning TCN model grid search\n")
        tcn_model_grid_search(model_data, input_size, time_step_range, kernel_size, filter_channels,
                              dilation_bases, batch_size, learning_rate, epochs=epochs, print_updates=True,
                              file_base_name=file_tcn_name)
        print("Finished with TCN model grid search\n")

    print("Window can be closed.")


if __name__ == "__main__":
    main()
