import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# 모델 초기화
# W = torch.zeros(1, requires_grad=True)
# b = torch.zeros(1, requires_grad=True)

# optimizer 설정
# optimizer = torch.optim.SGD([W, b], lr=0.01)

model = nn.Linear(1, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

nb_epochs = 10000

for epoch in range(nb_epochs + 1):

    # hypothesis = x_train * W + b

    # cost = torch.mean((hyothesis - y_train) ** 2)

    optimizer.zero_grad() # gradient를 0으로 초기화
    prediction = model(x_train)
    cost = F.mse_loss(prediction, y_train) # 평균 제곱 오차 함수
    cost.backward() # 비용 함수를 미분하여 gradient 계산
    optimizer.step() # W와 b를 업데이트 <- 학습률을 곱해서 업데이트

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

new_var = torch.FloatTensor([[4.0]])

pred_y = model(new_var)

print("\n훈련 후 입력이 4일 때의 예측값 :", pred_y)
print()
print(list(model.parameters()))