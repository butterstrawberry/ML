import torch
from torch import optim 
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
# x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
# x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
# y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# w1 = torch.zeros(1, requires_grad=True)
# w2 = torch.zeros(1, requires_grad=True)
# w3 = torch.zeros(1, requires_grad=True)
# b = torch.zeros(1, requires_grad=True)

model = nn.Linear(3, 1)

# optimizer = optim.SGD([w1, w2, w3, b], lr=le-5)

optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 90000

for epoch in range(nb_epochs + 1):

    # hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b

    # cost = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()
    prediction = model(x_train) # model.forward(x_train)과 동일
    cost = F.mse_loss(prediction, y_train)
    cost.backward()
    optimizer.step()

    if epoch % 10000 == 0:
        print('Epoch {:5d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

new_var = torch.FloatTensor([[73, 80, 75]])

pred_y = model(new_var)
print("\n훈련 후 입력이 73, 80, 75일 때의 예측값 {:.3f}:".format(pred_y.item())) 
print()
print(list(model.parameters()))