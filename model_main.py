import torch
import torch.nn as nn
from data.data_processing import ModelData2
from torch.utils.data import Dataset, DataLoader
from csv import writer
from models.point_ML_model import PointModel
from models.tcn_ML_model import TCNModel


class ModelDataset(Dataset):
    def __init__(self, data, labels):
        self.x = data
        self.y = labels
        self.n_samples = len(data)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


def train_model(model, training_data, testing_data, lr, epochs=1000, acc_interval=100, tcn=False, print_updates=True):
    if print_updates:
        print(f'Beginning training with {epochs} epochs at learning rate of {lr}\n')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    acc = 0
    for epoch in range(epochs):
        model.train()
        for batch, (train_data, train_labels) in enumerate(training_data):
            output_labels = model(train_data)
            if not tcn:
                output_labels = output_labels.permute(0, 2, 1)
            loss = criterion(output_labels, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch * len(training_data) + batch + 1) % acc_interval == 0:
                acc = test_model(model, testing_data, tcn=tcn, print_updates=False)
                if print_updates:
                    print(f"Batch {batch} of the {epoch} out of {epochs} epochs: accuracy of {acc}")

    if print_updates:
        print(f'Training completed with accuracy of {acc}\n')
    return acc


def test_model(model, testing_set, tcn=False, print_updates=True):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for test_data, test_labels in testing_set:
            test_labels = test_labels.type(torch.LongTensor)
            output = model(test_data)
            if not tcn:
                output_labels = output_labels.permute(0, 2, 1)
            predicted_label = output.argmax(dim=1, keepdim=True)[0]
            correct += predicted_label.eq(test_labels.view_as(predicted_label)).sum().item()
            if tcn:
                total += int(test_labels.shape[0])
            else:
                total += int(test_labels.shape[1])
    acc = 100 * correct / total
    if print_updates:
        print(f"Accuracy of {acc}% ({correct}/{total})\n")
    return acc


def point_model_grid_search(model_data, range_nodes, batch_size, learning_rate, print_updates=False):
    nodes_min, nodes_max, nodes_step = range_nodes
    for num_nodes_in_hl in range(nodes_min, nodes_max, nodes_step):
        model = PointModel(num_nodes_in_hl)
        if print_updates:
            print(f"Trying model with {num_nodes_in_hl} in hidden layer")
        training_set, testing_set = model_data.create_test_train(batch_size=batch_size)
        train_model(model, training_set, testing_set, learning_rate, print_updates=print_updates)

        final_acc = test_model(model, testing_set, print_updates=print_updates)

        model_features = (num_nodes_in_hl, batch_size, learning_rate, final_acc)
        torch.save(model.state_dict(), f"models/saved_models/point_model_{num_nodes_in_hl}.pt")
        print(f"finished model of {num_nodes_in_hl}")
        with open('data/point_models_params.csv', 'a') as f:
            csv_writer = writer(f)
            csv_writer.writerow(model_features)
            f.close()
    return


def tcn_model_grid_search(model_data, time_step_range, kernel_sizes, out_channel_range, range_nodes, batch_size, learning_rate):
    time_min, time_max, time_step_step = time_step_range
    nodes_min, nodes_max, nodes_step = range_nodes
    kernel_min, kernel_max, kernel_step = kernel_sizes
    cout_min, cout_max, cout_step = out_channel_range
    for time_steps in range(time_min, time_max, time_step_step):
        for kernel_size in range(kernel_min, kernel_max, kernel_step):
            if kernel_size > time_steps:
                continue
            for out_channels in range(cout_min, cout_max, cout_step):
                for num_nodes_in_hl in range(nodes_min, nodes_max, nodes_step):
                    tcn_model = TCNModel(kernel_size, time_steps, num_nodes_in_hl, out_channels)
                    model_data.create_time_series_data(time_steps)
                    training_set, testing_set = model_data.create_test_train(batch_size=batch_size, tcn=True)
                    train_model(tcn_model, training_set, testing_set, learning_rate, tcn=True, print_updates=True)
                    final_acc, final_loss = test_model(tcn_model, testing_set, print_updates=True)
                    model_features = (time_steps, kernel_size, out_channels, num_nodes_in_hl, batch_size, learning_rate,
                                      final_acc, final_loss)
                    torch.save(tcn_model.state_dict(),
                               f"models/saved_models/"
                               f"tcn_model_{num_nodes_in_hl}_{kernel_size}_{out_channels}_{time_steps}.pt")

                    with open('data/tcn_models_params.csv', 'a') as f:
                        csv_writer = writer(f)
                        csv_writer.writerow(model_features)
                        f.close()
    return


def main():
    time_step_range = (2, 31, 2)
    kernel_size = (2, 11, 2)
    out_channels = (1, 15, 2)
    num_nodes_in_hl = (7, 8, 1)
    batch_size = 1
    learning_rate = .01

    model_data = ModelData2("data/data6.mat")


    # print("Beginning point by point model grid search\n Please do not close window.")
    # point_model_grid_search(model_data, num_nodes_in_hl, batch_size, learning_rate)
    # print("Finished with point to point grid search \n")

    print("Beginning TCN model grid search\n")
    tcn_model_grid_search(model_data, time_step_range, kernel_size, out_channels, num_nodes_in_hl, batch_size, learning_rate)
    print("Finished with TCN model grid search\n")


    print("Window can be closed.")


if __name__ == "__main__":
    main()
