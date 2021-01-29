import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch


# an-8Flower
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.train = True

        self.classifier = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3),
            nn.AvgPool2d(12),
            nn.Flatten(),
            nn.Linear(256, 6)
        )

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        # self.gap = nn.AvgPool2d(12)
        self.ReLU = nn.ReLU(True)
        self.flat = nn.Flatten()
        self.fc2 = nn.Linear(9216, 256)
        self.fc1 = nn.Linear(256, 6)
        self.sm = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.pool(self.ReLU(self.conv1(x)))
        x = self.pool(self.ReLU(self.conv2(x)))
        x = self.pool(self.ReLU(self.conv3(x)))
        x = self.conv4(x)
        x = self.pool(self.ReLU(x))  # If we don't use GAP
        # x = self.gap(x)
        x = self.flat(x)
        x = self.ReLU(self.fc2(x))  # If we don't use GAP
        x = self.fc1(x)
        if not self.train:  # Give soft-max probability
            x = self.sm(x)
        return x

    def eval(self):
        # self.train = False
        return self


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
#         self.fc1 = nn.Linear(128, 64)
#         self.fc2 = nn.Linear(64, 10)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.pool(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = self.pool(x)
#         x = self.conv3(x)
#         x = F.relu(x)
#         x = self.pool(x)
#         x = x.view((1,128))  # flatten
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         return x



def train(train_dataset) -> nn.Module:
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, nesterov=True)

    running_loss = 0.0
    for j in range(0):
        trainloader = iter(DataLoader(train_dataset, 1, shuffle=True))

        for i, train_data in enumerate(trainloader, 0):
            inputs, labels, _ = train_data
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 600 == 599:
                print('[%d - %5d] loss: %.3f' % (j, i + 1, running_loss / 600))
                running_loss = 0.0

    net.eval()
    return net


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 4)
#         self.pool = nn.MaxPool2d(6, 3)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(1936, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 6)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 1936)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

def save_net(net, path):
    torch.save(net.state_dict(), path)


def load_net(path):
    net = Net()
    net.load_state_dict(torch.load(path))
    net.eval()
    return net
