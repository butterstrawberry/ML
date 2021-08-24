import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

torch.manual_seed(1)

class CustomDataset(Dataset):
    # 데이터셋의 전처리를 해주는 부분
    def __init__(self):
        self.x_data = [[73, 80, 75],
                   [93, 88, 93],
                   [89, 91, 90],
                   [96, 98, 100],
                   [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]] 

    # 총 샘플의 수를 적어주는 부분, len(dataset)을 했을 때 데이터셋의 크기 리턴
    def __len__(self):
        return len(self.x_data)

    # 데이터셋에서 특정 1개의 샘플을 가져오는 함수, dataset[i]을 했을 때 i번째 샘플을 가져오도록 하는 인덱싱
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y

class MutiLinearReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = MutiLinearReg()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples

        optimizer.zero_grad()
        prediction = model(x_train)
        cost = F.mse_loss(prediction, y_train)
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, batch_idx + 1, len(dataloader), cost.item()
        ))

new_var = torch.FloatTensor([[73, 80, 75]])

pred_y = model(new_var)
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y)

