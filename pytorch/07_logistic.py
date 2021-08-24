import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# W = torch.zeros((2, 1), requires_grad=True)
# b = torch.zeros(1, requires_grad=True)

# model = nn.Sequential(
#     nn.Linear(2, 1),
#     nn.Sigmoid()
# )

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
        # self.linear = nn.Linear(2, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.layer(x)
        # return self.sigmoid(self.linear(x))

model = BinaryClassifier()

# optimizer = optim.SGD([W, b], lr=1)

optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # hypothesis = torch.sigmoid(x_train.matmul(W) + b)
    
    # cost = -(y_train * torch.log(hypothesis) + 
    #          (1 - y_train) * torch.log(1 - hypothesis)).mean()
    
    optimizer.zero_grad()
    hypothesis = model(x_train)
    cost = F.binary_cross_entropy(hypothesis, y_train)
    cost.backward()
    optimizer.step()

    if epoch % 10 == 0:
        # 예측값이 0.5를 넘으면 True로 간주
        prediction = hypothesis >= torch.FloatTensor([0.5]) 
        # 실제값과 일치하는 경우만 True로 간주
        correct_prediction = prediction.float() == y_train 
        # 정확도를 계산
        accuracy = correct_prediction.sum().item() / len(correct_prediction) 
        # 각 에포크마다 정확도를 출력

        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy: {:6.2f}%'.format( 
            epoch, nb_epochs, cost.item(), accuracy * 100,
        ))



