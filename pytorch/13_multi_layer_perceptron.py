import torch
import torch.nn as nn
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed(777)

X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(2, 10),
            nn.Sigmoid(),
            nn.Linear(10, 10),

            nn.Sigmoid(),
            nn.Linear(10, 10),
            
            nn.Sigmoid(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer(x)
        return out

'''
model = nn.Sequential(
            nn.Linear(2, 10),
            nn.Sigmoid(),
            nn.Linear(10, 10),

            nn.Sigmoid(),
            nn.Linear(10, 10),

            nn.Sigmoid(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        ).to(device)
'''

model = MLP()

criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=1)

for epoch in range(7801):
    optimizer.zero_grad()

    hypothesis = model(X)

    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('{:5d} {}'.format(epoch, cost.item()))

with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print('Hypothesis :', hypothesis.detach().cpu().numpy())
    print('Predicted :', predicted.detach().cpu().numpy())
    print('Y :', Y.cpu().numpy())
    print('Accuracy :', accuracy.item())

'''
from torch.utils.data import Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1)
if device == 'cuda':
    torch.cuda.manual_seed(1)

class TestDataset(Dataset):
    def __init__(self):
        self.x_data = torch.FloatTensor([[0, 0], 
                                         [0, 1], 
                                         [1, 0], 
                                         [1, 1]]).to(device)
        self.y_data = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

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
            nn.Linear(2, 10),
            nn.Sigmoid(),
            nn.Linear(10, 10),

            nn.Sigmoid(),
            nn.Linear(10, 10),

            nn.Sigmoid(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)

dataset = TestDataset()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = TestModel().to(device)

criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 10000
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples

        optimizer.zero_grad()
        hypothesis = model(x_train)
        cost = criterion(hypothesis, y_train)
        cost.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print('Epoch {:5d}/{} Batch {}/{} Cost: {:.4f}'.format(
                epoch, nb_epochs, batch_idx + 1, len(dataloader), cost.item()))

with torch.no_grad():
    hypothesis = model(x_train)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == y_train).float().mean()
    print('Hypothesis :', hypothesis.detach().cpu().numpy())
    print('Predicted :', predicted.detach().cpu().numpy())
    print('Y :', y_train.cpu().numpy())
    print('Accuracy :', accuracy.item())
'''