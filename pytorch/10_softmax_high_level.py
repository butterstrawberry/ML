import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

# W = torch.zeros((4, 3), requires_grad=True)
# b = torch.zeros(1, requires_grad=True)

class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)

model = SoftmaxClassifierModel()

# optimizer = optim.SGD([W, b], lr=0.1)

optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # z = x_train.matmul(W) + b
    optimizer.zero_grad()
    prediction = model(x_train)
    cost = F.cross_entropy(prediction, y_train)
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

'''
class TestDataset(Dataset):
    def __init__(self):
        self.x_data = torch.FloatTensor([[1, 2, 1, 1],
                       [2, 1, 3, 2],
                       [3, 1, 3, 4],
                       [4, 1, 5, 5],
                       [1, 7, 5, 5],
                       [1, 2, 5, 6],
                       [1, 6, 6, 6],
                       [1, 7, 7, 7]])
        self.y_data = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0])

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4, 3)
        )

    def forward(self, x):
        return self.layer(x)

dataset = TestDataset()
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = TestModel()

optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples

        optimizer.zero_grad()
        prediction = model(x_train)
        cost = F.cross_entropy(prediction, y_train)
        cost.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
                epoch, nb_epochs, batch_idx + 1, len(dataloader), cost.item()
            ))
'''