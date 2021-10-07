import torch
import torch.nn as nn
from data.data_processing import ModelData
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


def get_train_test(batch_size, point_or_time, file_name="data/data.csv"):
    model_data = ModelData(file_name)
    x_train, x_test, y_train, y_test = model_data.create_test_train(point_or_time)
    train_dataset = ModelDataset(x_train, y_train)
    test_dataset = ModelDataset(x_test, y_test)
    training = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    testing = DataLoader(dataset=test_dataset, shuffle=True)
    print(
        f'Created training set with {len(training)} batches of {batch_size} and validation set of size {len(testing)}\n')
    return training, testing


def train_model(model, training_data, testing_data, lr, epochs=1000, acc_interval=50, output=True):
    if output:
        print(f'Beginning training with {epochs} epochs at learning rate of {lr}\n')
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    acc = 0
    for epoch in range(epochs):
        model.train()
        for train_data, train_labels in training_data:
            optimizer.zero_grad()
            output = model(train_data)
            loss = criterion(output, train_labels)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % acc_interval == 0:
            if output:
                print(f'epoch: {epoch + 1}/{epochs}')
            new_acc = test_model(model, testing_data, output)
            diff = new_acc - acc
            acc = new_acc
            if acc >= 99:
                if output:
                    print('Training terminated early due to high accuracy\n')
                break
            if diff < .001:
                if output:
                    print('Training terminated early due to diminishing returns\n')
                break
    if output:
        print(f'Training completed with accuracy of {acc}\n')
    return acc


def test_model(model, testing_set, output=True):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.NLLLoss(reduction='sum')
    with torch.no_grad():
        for test_data, test_labels in testing_set:
            output = model(test_data)
            loss = criterion(output, test_labels)
            test_loss += loss.item()
            predicted_label = output.argmax(dim=1, keepdim=True)
            correct += predicted_label.eq(test_labels.view_as(predicted_label)).sum().item()
    test_loss /= len(testing_set.dataset)
    acc = 100 * correct / len(testing_set.dataset)
    if output:
        print(f"Average loss of {test_loss} and accuracy of {acc}% ({correct}/{len(testing_set.dataset)})\n")
    return acc


def point_model_grid_search(range_nodes, batch_sizes, learning_rates):
    nodes_min, nodes_max, nodes_step = range_nodes
    batch_min, batch_max, batch_step = batch_sizes
    lr_min, lr_max, lr_step = learning_rates
    for num_nodes_in_hl in range(nodes_min, nodes_max, nodes_step):
        for batch_size in range(batch_min, batch_max, batch_step):
            for learning_rate in range(lr_min, lr_max, lr_step):
                point_model = PointModel(num_nodes_in_hl)
                training_set, testing_set = get_train_test(batch_size, 'point')
                train_model(point_model, training_set, testing_set, learning_rate, output=False)
                final_acc = test_model(point_model, testing_set, output=False)
                model_features = (num_nodes_in_hl, batch_size, learning_rate, final_acc)
                torch.save(point_model.state_dict(), f"models/saved_models/point_model_{num_nodes_in_hl}.pt")
                with open('data/point_models_params.csv', 'a') as f:
                    csv_writer = writer(f)
                    csv_writer.writerow(model_features)
                    f.close()


def tcn_model_grid_search(kernel_sizes, range_nodes, batch_sizes, learning_rates):
    nodes_min, nodes_max, nodes_step = range_nodes
    batch_min, batch_max, batch_step = batch_sizes
    lr_min, lr_max, lr_step = learning_rates
    kernel_min, kernel_max, kernel_step = kernel_sizes
    for kernel_size in range(kernel_min, kernel_max, kernel_step):
        for num_nodes_in_hl in range(nodes_min, nodes_max, nodes_step):
            for batch_size in range(batch_min, batch_max, batch_step):
                for learning_rate in range(lr_min, lr_max, lr_step):
                    tcn_model = TCNModel(kernel_size, num_nodes_in_hl)
                    training_set, testing_set = get_train_test(batch_size, 'time')
                    train_model(tcn_model, training_set, testing_set, learning_rate, output=False)
                    final_acc = test_model(tcn_model, testing_set, output=False)
                    model_features = (kernel_size, num_nodes_in_hl, batch_size, learning_rate, final_acc)
                    torch.save(tcn_model.state_dict(),
                               f"models/saved_models/tcn_model_{num_nodes_in_hl}_{kernel_size}.pt")
                    with open('data/tcn_models_params.csv', 'a') as f:
                        csv_writer = writer(f)
                        csv_writer.writerow(model_features)
                        f.close()


def main():
    kernel_size = (2, 10, 1)
    num_nodes_in_hl = (0, 31, 2)
    batch_size = (1, 102, 10)
    learning_rate = (1e-5, 2e-5, 1e-5)

    print("Beginning point by point model grid search\n")
    point_model_grid_search(num_nodes_in_hl, batch_size, learning_rate)
    print("Finished with point to point grid search \n")

    print("Beginning TCN model grid search\n")
    tcn_model_grid_search(kernel_size, num_nodes_in_hl, batch_size, learning_rate)
    print("Finished with TCN model grid search\n")


if __name__ == "__main__":
    main()
