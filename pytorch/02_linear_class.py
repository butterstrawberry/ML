import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

        # super()에 파생 클래스와 self를 넣어서 현재 클래스가 
        # 어떤 클래스인지 명확하게 표시 가능, super()와 동일.
        # 명확하게 super을 사용하기 위해 기반 클래스의 메서드 호출.
        # super(LR_Model, self).__init__()
        
        # 연속된 범위의 Dim을 텐서로 평평하게 만듬
        # self.flatten = nn.Flatten()          

        # self.linear = nn.Sequential(         
        #     nn.Linear(1, 1)                  
        # )

    def forward(self, x):
        return self.linear(x)

        # x = self.flatten(x)
        # linear = self.linear(x)
        # return linear

model = LinearRegressionModel()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

nb_epochs = 2000
for epoch in range(nb_epochs + 1):
    
    optimizer.zero_grad()
    prediction = model(x_train)
    cost = F.mse_loss(prediction, y_train)
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))