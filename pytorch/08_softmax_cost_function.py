import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

z = torch.rand(3, 5, requires_grad=True)

hypothesis = F.softmax(z, dim=1)

print(hypothesis)

y = torch.randint(5, size=(3, )).long()

y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(dim=1, index=y.unsqueeze(1), value=1)

# Low level
cost1 = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost1)

# F.softmax() + torch.log() = F.log_softmax()
cost2 = (y_one_hot * -F.log_softmax(z, dim=1)).sum(dim=1).mean()
print(cost2)

# (y_one_hot * -X).sum(dim=1).mean() = F.nll_loss(X)
cost3 = F.nll_loss(F.log_softmax(z, dim=1), y)
print(cost3)

# F.log_softmax() + F.nll_loss() = F.cross_entropy()
cost4 = F.cross_entropy(z, y)
print(cost4)