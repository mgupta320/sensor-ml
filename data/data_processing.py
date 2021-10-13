import numpy as np
import pandas as pd
import pickle as pickle
from sklearn.model_selection import train_test_split


class ModelData:
    def __init__(self, file_name, time_steps=10, num_samples=1346):
        self.data = pd.read_csv(file_name, sep=',', header=None).to_numpy()
        labels_start = np.where(self.data[0] == 1)[0][0]
        point_x = self.data[:, :labels_start]
        point_y = self.data[:, labels_start:]
        self.point_x = point_x
        self.point_y = np.argmax(point_y, axis=-1)
        self.time_x = None
        self.time_y = None
        self.time_steps = time_steps
        self.point_to_time(time_steps, num_samples)

    def point_to_time(self, time_steps=10, num_samples=1346):
        point_data = self.point_x
        point_labels = self.point_y
        self.time_steps = time_steps
        time_data = np.empty(
            ((len(point_data) - time_steps * len(point_data) // num_samples), point_data.shape[1], time_steps))
        time_labels = np.empty((time_data.shape[0]))
        i = 0
        n = 0
        while i < len(point_data):
            if (i + time_steps) >= len(point_data) or i % num_samples > num_samples - time_steps:
                i += time_steps
                continue
            time_data[n] = point_data[i:i + time_steps].transpose()
            time_labels[n] = point_labels[i]
            n += 1
            i += 1
        self.time_x = time_data
        self.time_y = time_labels

    def create_test_train(self, test_size=.33, rand_seed=0, save_name=None):
        data = self.point_x
        labels = self.point_y
        point_sets = train_test_split(data, labels, test_size=test_size, random_state=rand_seed)
        data = self.time_x
        labels = self.time_y
        time_sets = train_test_split(data, labels, test_size=test_size, random_state=rand_seed)
        test_train_sets = (point_sets, time_sets)
        if save_name:
            pickle.dump(test_train_sets, open(f"{save_name}_{self.time_steps}.p", "wb"))
        return test_train_sets

