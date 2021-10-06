import torch
import torch.nn as nn
from data_processing import ModelData
from torch.utils.data import Dataset, DataLoader


class PointDataset(Dataset):
    def __init__(self, data, labels):
        self.x = data
        self.y = labels
        self.n_samples = len(data)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


class PointModel(nn.Module):
    def __init__(self, n_hidden):
        super(PointModel, self).__init__()
        self.n_hidden = n_hidden
        if n_hidden == 0:
            self.linear = nn.Linear(6, 25)
        else:
            self.linear1 = nn.Linear(6, n_hidden)
            self.linear2 = nn.Linear(n_hidden, 25)

    def forward(self, x):
        x = x.float()
        if self.n_hidden == 0:
            x = self.linear(x)
        else:
            x = self.linear1(x)
            x = self.linear2(x)
        x = torch.sigmoid(x)
        y_predicted = nn.functional.log_softmax(x, dim=1)
        return y_predicted


def get_train_test(batch_size, file_name="data/data.csv"):
    model_data = ModelData(file_name)
    x_train, x_test, y_train, y_test = model_data.create_test_train("point")
    train_dataset = PointDataset(x_train, y_train)
    test_dataset = PointDataset(x_test, y_test)
    training = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    testing = DataLoader(dataset=test_dataset, shuffle=True)
    print(
        f'Created training set with {len(training)} batches of {batch_size} and validation set of size {len(testing)}\n')
    return training, testing


def train_point_model(point_model, training_data, testing_data, lr, epochs, acc_interval=50):
    print(f'Beginning training with {epochs} epochs at learning rate of {lr}\n')
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(point_model.parameters(), lr=lr)
    acc = 0
    for epoch in range(epochs):
        point_model.train()
        for train_data, train_labels in training_data:
            optimizer.zero_grad()
            output = point_model(train_data)
            loss = criterion(output, train_labels)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % (epochs // acc_interval) == 0:
            print(f'epoch: {epoch + 1}/{epochs}')
            new_acc = test_point_model(point_model, testing_data)
            diff = new_acc - acc
            acc = new_acc
            if acc >= 99:
                print('Training terminated early due to high accuracy\n')
                break
            if diff < .001:
                print('Training terminated early due to diminishing returns\n')
                break
    print(f'Training completed with accuracy of {acc}\n')
    return acc


def test_point_model(model, testing_set):
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
    print(f"Average loss of {test_loss} and accuracy of {acc}% ({correct}/{len(testing_set.dataset)})\n")
    return acc


def main():
    num_nodes_in_hl = 0
    batch_size = 100
    learning_rate = 1
    epochs = 1000
    point_model = PointModel(num_nodes_in_hl)
    training_set, testing_set = get_train_test(batch_size)

    train_point_model(point_model, training_set, testing_set, learning_rate, epochs)
    final_acc = test_point_model(point_model, testing_set)
    model_features = (num_nodes_in_hl, batch_size, learning_rate, final_acc)
    torch.save(point_model.state_dict(), f"point_model{num_nodes_in_hl}.pt")


if __name__ == "__main__":
    main()


