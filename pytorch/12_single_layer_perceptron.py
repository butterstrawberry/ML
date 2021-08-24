import torch
import torch.nn as nn
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed(777)

X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

linear = nn.Linear(2, 1, bias=True)
sigmoid = nn.Sigmoid()
model = nn.Sequential(linear, sigmoid).to(device)

criterion = nn.BCELoss().to(device) # binary classification cross entropy function
optimizer = optim.SGD(model.parameters(), lr=1)

for step in range(1001):
    
    optimizer.zero_grad()
    hypothesis = model(X)
    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    if step % 100 == 0:
        print('{:4d} {}'.format(step, cost.item()))

with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print('Hypothesis :', hypothesis.detach().cpu().numpy())
    print('Predicted :', predicted.detach().cpu().numpy())
    print('Y :', Y.cpu().numpy())
    print('Accuracy :', accuracy.item())