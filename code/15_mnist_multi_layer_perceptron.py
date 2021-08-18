import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

mnist = fetch_openml('mnist_784', version=1, cache=True)

mnist.target = mnist.target.astype(np.int8)

x = mnist.data / 255 # 정규화
y = mnist.target

x = np.array(x)
y = np.array(y)

# 훈련 데이터와 테스트 데이터의 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/7, random_state=0)

x_train = torch.Tensor(x_train)
x_test = torch.Tensor(x_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

ds_train = TensorDataset(x_train, y_train)
ds_test = TensorDataset(x_test, y_test)

loader_train = DataLoader(ds_train, batch_size=64, shuffle=True)
loader_test = DataLoader(ds_test, batch_size=64, shuffle=False)

# multi layer perceptron
model = nn.Sequential()
model.add_module('fc1', nn.Linear(28*28*1, 100))
model.add_module('relu1', nn.ReLU())
model.add_module('fc2', nn.Linear(100, 100))
model.add_module('relu2', nn.ReLU())
model.add_module('fc3', nn.Linear(100, 10))


# 오차함수 선택
loss_fn = nn.CrossEntropyLoss()

# 가중치를 학습하기 위한 최적화 기법 선택
optimizer = optim.Adam(model.parameters(), lr=0.001)

nb_epoch = 25

def train(epoch):
    model.train() # 신경망을 학습 모드로 전환

    # 데이터로더에서 미니배치를 하나씩 꺼내 학습을 수행
    for data, targets in loader_train:
        optimizer.zero_grad() # 경사를 0으로 초기화
        ouputs = model(data) # 데이터를 입력하고 출력을 계산
        loss = loss_fn(ouputs, targets) # 출력과 훈련 데이터 정답 간의 오차를 계산
        loss.backward() # 오차를 역전파 계산
        optimizer.step() # 역전파 계산한 값으로 가중치를 수정

    print("epoch {}/{} Complete\n".format(
        epoch + 1, nb_epoch
        ))

def test():
    model.eval() # 신경망을 추론 모드로 전환
    correct = 0

    # 데이터로더에서 미니배치를 하나씩 꺼내 추론을 수행
    with torch.no_grad(): # 추론 과정에는 미분이 불필요
        for data, targets in loader_test:
            outputs = model(data) # 데이터를 입력하고 출력을 계산

            # 추론 계산
            _, predicted = torch.max(outputs.data, 1) # 확률이 가장 높은 레이블이 무엇인지 계산
            correct += predicted.eq(targets.data.view_as(predicted)).sum() # 정답과 일치한 경우 정답 카운트를 증가

    # 정확도 출력
    data_num = len(loader_test.dataset) # 데이터 총 건수
    print('\nPredictive Accuracy : {}/{} ({:.0f}%)\n'.format(
        correct, data_num, 100 * correct / data_num
        ))

for epoch in range(nb_epoch):
    train(epoch)

test()

index = 111

model.eval() # 신경망을 추론 모드로 전환
data = x_test[index]
output = model(data) # 데이터를 입력하고 출력을 계산
_, predicted = torch.max(output.data, 0) # 확률이 가장 높은 레이블이 무엇인지 계산

print("Predicted Result : {}".format(predicted))

x_test_show = (x_test[index]).numpy()
plt.imshow(x_test_show.reshape(28, 28), cmap='gray')
print("The correct label is {:.0f}.".format(y_test[index]))
plt.show()
