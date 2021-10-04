import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def csv_to_arrays(file_name):
    # Load data from the csv file into a workable array
    total_data = pd.read_csv(file_name, sep=',', header=None)
    total_data = total_data.to_numpy()

    # separate input data and labelled outputs for model
    labels_start = np.where(total_data[0] == 1)[0][0]
    point_data = total_data[:, :labels_start]
    point_labels = total_data[:, labels_start:]

    # creating time series data with 5 time ordered points and their associated labels
    time_data_shape = (point_data.shape[0] - 25 * 5, 5, point_data.shape[1])
    time_data = np.empty(time_data_shape)
    time_labels = np.empty((time_data.shape[0], point_labels.shape[1]))
    n = 0
    for i in range(len(point_data)):
        if (i + 5) >= len(point_data) or not np.array_equal(point_labels[i], point_labels[i + 5]):
            n -= 1
            continue
        time_labels[n] = point_labels[i]
        time_data[n] = point_data[i:i + 5]
        n += 1

    # if we want to save these arrays to use later
    point_x_train, point_x_test, point_y_train, point_y_test = train_test_split(point_data, point_labels, test_size=.33,
                                                                                random_state=0)
    time_x_train, time_x_test, time_y_train, time_y_test = train_test_split(time_data, time_labels, test_size=.33,
                                                                            random_state=0)
    np.savez("data/np_point_arrays", point_x_train, point_x_test, point_y_train, point_y_test)
    np.savez("data/np_time_arrays", time_x_train, time_x_test, time_y_train, time_y_test)

    # returning a tuple of two arrays that have the point data and time data respectively
    return ([point_x_train, point_x_test, point_y_train, point_y_test],
            [time_x_train, time_x_test, time_y_train, time_y_test])


def load_arrays(file_name):
    loaded_arrays = np.load(file_name)
    data_arrays = (loaded_arrays['arr_0'], loaded_arrays['arr_1'], loaded_arrays['arr_2'], loaded_arrays['arr_3'])
    return data_arrays


point_arrays, time_arrays = csv_to_arrays("data/data.csv")
point_x_train, point_x_test, point_y_train, point_y_test = load_arrays("data/np_point_arrays.npz")
