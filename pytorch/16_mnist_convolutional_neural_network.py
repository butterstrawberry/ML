import torch

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)

if device == 'cuda':
    torch.cuda.manual_seed_all(777)

learning_rate = 1e-3
training_epochs = 15
batch_size = 100

mnist_train = datasets.MNIST(root='D:/Dataset/MNIST_data/',
                             train=True,
                             transform=transforms.ToTensor(),
                             download=True)

mnist_test = datasets.MNIST(root='D:/Dataset/MNIST_data/',
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)               

data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # first layer
        # Image shape = (?, 28, 28, 1)
        #   Conv   -> (?, 28, 28, 32)
        #   Pool   -> (?, 14, 14, 32)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # second layer
        # Image shape = (?, 14, 14, 32)
        #   Conv   -> (?, 14, 14, 64)
        #   Pool   -> (?, 7, 7, 64)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Linear(7 * 7 * 64, 10, bias=True) # Full Connected Layers

        nn.init.xavier_uniform_(self.fc.weight) # Only Full Connected Layers Reset Weight

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1) # Flatten
        out = self.fc(out)
        return out

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)
print('Total number of batches : {}'.format(total_batch))

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

with torch.no_grad():
    X_test = mnist_test.data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.targets.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy :', accuracy.item())
