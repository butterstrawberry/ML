import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

torch.manual_seed(1)

digits = load_digits()

X = digits.data
Y = digits.target

model = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10)
)

X = torch.tensor(X, dtype = torch.float32)
Y = torch.tensor(Y, dtype = torch.int64)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

losses = []

for epoch in range(101):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, Y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, 100, loss.item()
        ))

    losses.append(loss.item())

plt.plot(losses)
plt.show()

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt

# from torch.utils.data import Dataset, DataLoader
# from sklearn.datasets import load_digits

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# torch.manual_seed(1)
# if device == 'cuda':
#     torch.cuda.manual_seed(1)

# digits = load_digits()

# class DigitDataset(Dataset):
#     def __init__(self):
#         self.x_data = torch.FloatTensor(digits.data)
#         self.y_data = torch.LongTensor(digits.target)

#     def __len__(self):
#         return len(self.x_data)

#     def __getitem__(self, idx):
#         x = self.x_data[idx]
#         y = self.y_data[idx]
#         return x, y

# class MLPModel(nn.Module):
#     def __init__(self):
#         super(MLPModel, self).__init__()
#         self.layer = nn.Sequential(
#             nn.Linear(64, 32),
#             nn.ReLU(),

#             nn.Linear(32, 16),
#             nn.ReLU(),

#             nn.Linear(16, 10),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         return self.layer(x)

# dataset = DigitDataset()
# dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

# model = MLPModel().to(device)
# criterion = nn.CrossEntropyLoss().to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)

# losses = []

# nb_epochs = 100
# for epoch in range(nb_epochs + 1):
#     for batch_idx, samples in enumerate(dataloader):
#         X, Y = samples

#         optimizer.zero_grad()
#         prediction = model(X)
#         cost = criterion(prediction, Y)
#         cost.backward()
#         optimizer.step()

#         if epoch % 10 == 0:
#             print('Epoch {:4d}/{} Cost: {:.6f}'.format(
#                 epoch, nb_epochs, cost.item()
#             ))

#     losses.append(cost.item())

# plt.plot(losses)
# plt.show()