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


def get_train_test(model_data, batch_size):
    point_and_time = model_data.create_test_train()
    pair = []
    for x_train, x_test, y_train, y_test in point_and_time:
        train = ModelDataset(x_train, y_train)
        test = ModelDataset(x_test, y_test)
        training = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
        testing = DataLoader(dataset=test, shuffle=True)
        pair.append((testing, training))
    return pair


def train_model(model, training_data, testing_data, lr, epochs=1000, acc_interval=50, output=True):
    if output:
        print(f'Beginning training with {epochs} epochs at learning rate of {lr}\n')
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    acc = 0
    for epoch in range(epochs):
        model.train()
        for train_data, train_labels in training_data:
            train_labels = train_labels.type(torch.LongTensor)
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
            train_labels = train_labels.type(torch.LongTensor)
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


def point_model_grid_search(model_data, range_nodes, batch_sizes, learning_rate):
    nodes_min, nodes_max, nodes_step = range_nodes
    batch_min, batch_max, batch_step = batch_sizes
    for num_nodes_in_hl in range(nodes_min, nodes_max, nodes_step):
        for batch_size in range(batch_min, batch_max, batch_step):
            point_model = PointModel(num_nodes_in_hl)
            training_set, testing_set = get_train_test(model_data, batch_size)[0]
            train_model(point_model, training_set, testing_set, learning_rate, output=False)
            final_acc = test_model(point_model, testing_set, output=False)
            model_features = (num_nodes_in_hl, batch_size, learning_rate, final_acc)
            torch.save(point_model.state_dict(), f"models/saved_models/point_model_{num_nodes_in_hl}.pt")

            with open('data/point_models_params.csv', 'a') as f:
                csv_writer = writer(f)
                csv_writer.writerow(model_features)
                f.close()
    return


def tcn_model_grid_search(model_data, time_step_range, kernel_sizes, range_nodes, batch_sizes, learning_rate):
    time_min, time_max, time_step_step = time_step_range
    nodes_min, nodes_max, nodes_step = range_nodes
    batch_min, batch_max, batch_step = batch_sizes
    kernel_min, kernel_max, kernel_step = kernel_sizes
    for time_steps in range(time_min, time_max, time_step_step):
        for kernel_size in range(kernel_min, kernel_max, kernel_step):
            if kernel_size < time_steps:
                continue
            for num_nodes_in_hl in range(nodes_min, nodes_max, nodes_step):
                for batch_size in range(batch_min, batch_max, batch_step):
                    tcn_model = TCNModel(kernel_size, time_steps, num_nodes_in_hl)
                    model_data.point_to_time(time_steps)
                    training_set, testing_set = get_train_test(model_data, batch_size)[1]
                    train_model(tcn_model, training_set, testing_set, learning_rate, output=False)
                    final_acc = test_model(tcn_model, testing_set, output=False)
                    model_features = (time_steps, kernel_size, num_nodes_in_hl, batch_size, learning_rate, final_acc)
                    torch.save(tcn_model.state_dict(),
                               f"models/saved_models/tcn_model_{num_nodes_in_hl}_{kernel_size}_{time_steps}.pt")
                    '''
                    #with open('data/tcn_models_params.csv', 'a') as f:
                        csv_writer = writer(f)
                        csv_writer.writerow(model_features)
                        f.close()
                    '''


def main():

    time_step_range = (2, 31, 2)
    kernel_size = (2, 10, 1)
    num_nodes_in_hl = (0, 31, 2)
    batch_size = (1, 102, 10)
    learning_rate = 1e-5

    model_data = ModelData("data/data.csv")

    print("Beginning point by point model grid search\n")
    point_model_grid_search(model_data, num_nodes_in_hl, batch_size, learning_rate)
    print("Finished with point to point grid search \n")

    print("Beginning TCN model grid search\n")
    tcn_model_grid_search(model_data, time_step_range, kernel_size, num_nodes_in_hl, batch_size, learning_rate)
    print("Finished with TCN model grid search\n")


if __name__ == "__main__":
    main()