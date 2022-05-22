import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from scipy.io import loadmat


class ModelDataContainer:
    def __init__(self, classes, file_name=None, matrix_name=None, matrix_cont=None, time_steps=10, num_samples=1346, input_vars=6):
        """
        ModelDataContainer initializer that creates container to handle data for model
        :param classes: tuple where each value is a string at its index value in target index
        :param file_name: string path to .mat matrix
        :param matrix_name: string name of matrix in matlab, defaults to None in which case matrix name is file name
        :param matrix_cont: tuple containing two numpy arrays containing x data and corresponding labels
        :param time_steps: number of time steps for TCN (can be changed)
        :param num_samples: number of samples in each test
        :param input_vars: number of input variables in each sample
        """
        # get data from matlab matrix and split into x and y data
        if file_name is not None:
            if matrix_name is None:
                matrix_name = file_name.split("/")[-1][:-4]
            input_data = loadmat(file_name, verify_compressed_data_integrity=False)[matrix_name]
            x = input_data[:, :, 0:input_vars]
            y = input_data[:, :, input_vars]
        elif matrix_cont is not None:
            x = matrix_cont[0]
            y = matrix_cont[1]
        else:
            raise Exception("Must provide initial data to ModelDataContainer")

        # standardize input data (Necessary step for many ML classification applications)
        standardizer = MinMaxScaler()
        x_standardized = np.zeros(np.shape(x))
        for i in range(num_samples):
            x_standardized[:, i, :input_vars] = standardizer.fit_transform(x[:, i, :input_vars])
        self.input_size = input_vars
        self.classes = classes
        self.x = x_standardized.astype(np.float32)
        self.x = x.astype(np.float32)  # for raw data
        self.y = y.astype(np.int64)
        self.x_point = self.x
        self.y_point = self.y
        self.training = None
        self.testing = None
        self.time_steps = time_steps
        self.x_time = None
        self.y_time = None
        self.create_time_series_data(time_steps)  # format data for tcn use

    def create_train_test(self, test_size=.33, batch_size=5, tcn=False):
        """
        Creates testing and training DataLoader for model training and testing
        :param test_size: float where 0 <= testsize < 1, 0 results in testing set and training set being entire dataset
        :param batch_size: int number of samples in each batch trained on before an optimization step
        :param tcn: boolean of whether DataLoaders are for TCN or not
        :return: tuple containing (DataLoader training_set, DataLoader testing_set)
        """
        # select point data for ANN and time series data for TCN and combine test and sample dimensions
        if not tcn:
            data = self.x_point
            labels = self.y_point
            data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
            labels = labels.reshape((labels.shape[0] * labels.shape[1]))
        else:
            data = self.x_time
            labels = self.y_time
            data = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3]))
            labels = labels.reshape((labels.shape[0] * labels.shape[1]))

        # if test size is 0, testing and training set are entire dataset
        if test_size == 0:
            x_r, x_t, y_r, y_t = data, data, labels, labels
        else:
            x_r, x_t, y_r, y_t = train_test_split(data, labels, test_size=test_size)

        x_train = torch.from_numpy(x_r)
        y_train = torch.from_numpy(y_r)
        x_test = torch.from_numpy(x_t)
        y_test = torch.from_numpy(y_t)

        train = TensorDataset(x_train, y_train)
        test = TensorDataset(x_test, y_test)

        training = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, num_workers=0)
        testing = DataLoader(dataset=test, shuffle=True, num_workers=0)
        self.training = training
        self.testing = testing
        return training, testing

    def create_time_series_data(self, time_steps):
        """
        Create time series data for use in TCN
        :param time_steps: int number of time steps needed for TCN
        :return: tuple containing (np x, np y)
        """
        self.time_steps = time_steps
        point_data_shape = self.x.shape
        time_x = np.empty((point_data_shape[0], point_data_shape[1] - time_steps, point_data_shape[2], time_steps))
        time_y = np.empty((point_data_shape[0], point_data_shape[1] - time_steps))

        for test in range(point_data_shape[0]):
            for sample in range(point_data_shape[1] - time_steps):
                time_series = self.x[test, sample:(sample + time_steps), :]
                time_series = np.transpose(time_series)
                time_x[test, sample] = time_series
            time_y[test, :] = self.y[test, 0]

        self.x_time = time_x.astype(np.float32)
        self.y_time = time_y.astype(np.int64)
        return self.x_time, self.y_time

    def create_k_fold_val(self, k, batch_size, tcn=False):
        """
        Create training and testing sets for k-fold cross validation
        :param k: int number of fold for cross validation
        :param batch_size: int number of samples in each batch trained on before an optimization step
        :param tcn: boolean of whether DataLoaders are for TCN or not
        :return: array of k tuples where each tuple contains (DataLoader training_set, DataLoader testing_set)
        """
        # collect data sets to be used
        if not tcn:
            data = self.x_point
            labels = self.y_point
        else:
            data = self.x_time
            labels = self.y_time
        flat_val = 1 + int(tcn)  # Last dimension kept same in ANN, last 2 in TCN kept same

        kfold_sets = []
        for fold in range(k):
            # each fold uses every kth point for testing and all other points for training to prevent leakage when
            # measuring model performance vs time
            mask_train = [x for x in range(data.shape[1]) if (x + fold) % k != 0]
            mask_test = [x for x in range(data.shape[1]) if (x + fold) % k == 0]
            x_train = data[:, mask_train, :]
            train_shape = x_train.shape
            x_test = data[:, mask_test, :]
            test_shape = x_test.shape
            y_train = labels[:, mask_train]
            y_test = labels[:, mask_test]

            # test and sample dimensions combined
            x_train = torch.from_numpy(x_train.reshape(-1, *x_train.shape[-flat_val:]))
            y_train = torch.from_numpy(y_train.reshape((train_shape[0] * train_shape[1])))
            x_test = torch.from_numpy(x_test.reshape(-1, *x_test.shape[-flat_val:]))
            y_test = torch.from_numpy(y_test.reshape((test_shape[0] * test_shape[1])))

            # create data loaders for training and testing with testing set still in chronological order
            train = TensorDataset(x_train, y_train)
            test = TensorDataset(x_test, y_test)
            training = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, num_workers=0)
            testing = DataLoader(dataset=test, shuffle=False, num_workers=0)
            kfold_sets.append((training, testing))

        return kfold_sets
