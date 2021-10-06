import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class ModelData:
    def __init__(self, file_name, time_steps=10, num_samples=1346):
        self.data = pd.read_csv(file_name, sep=',', header=None).to_numpy()
        point_x, point_y = self.data_to_arrays(self.data)
        self.point_x = point_x
        self.point_y = np.argmax(point_y, axis=-1)
        time_x, time_y = self.point_to_time(self.point_x, self.point_y, time_steps, num_samples)
        self.time_x = time_x
        self.time_y = time_y
        self.time_steps = time_steps


    def data_to_arrays(self, total_data):
        # separate input data and labelled outputs for model
        labels_start = np.where(total_data[0] == 1)[0][0]
        point_data = total_data[:, :labels_start]
        point_labels = total_data[:, labels_start:]

        # returning a tuple of two arrays that have the point data and time data respectively
        return point_data, point_labels

    def point_to_time(self, point_data, point_labels, time_steps=10, num_samples=1346):
        time_data = np.empty(
            ((len(point_data) - time_steps * len(point_data) // num_samples), time_steps, point_data.shape[1]))
        time_labels = np.empty((time_data.shape[0]))
        i = 0
        n = 0
        while i < len(point_data):
            if (i + time_steps) >= len(point_data) or i % num_samples > num_samples - time_steps:
                i += time_steps
                continue
            time_data[n] = point_data[i:i + time_steps]
            time_labels[n] = point_labels[i]
            n += 1
            i += 1
        return point_data, point_labels

    def create_test_train(self, point_or_time, test_size=.33, rand_seed=0, save_name=None):
        if point_or_time == 'point':
            data = self.point_x
            labels = self.point_y
        elif point_or_time == 'time':
            data = self.time_x
            labels = self.time_y
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=rand_seed)
        if save_name:
            np.savez(f"data/{save_name}", x_train, x_test, y_train, y_test)
        return x_train, x_test, y_train, y_test

'''
model_data_1 = ModelData("data/data.csv")
model_data_1.create_test_train('point', save_name="np_point_arrays")
model_data_1.create_test_train('time', save_name="np_time_arrays")
'''