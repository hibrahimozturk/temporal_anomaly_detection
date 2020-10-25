import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, featureSize):
        super().__init__()
        self.linear1 = nn.Linear(featureSize, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 32)
        self.linear4 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = torch.sigmoid(x)
        return x


if __name__ == "__main__":
    x = torch.rand(20, 1024)
    mlp = MLP()
    x = mlp(x)

    x = torch.range(0, 19).view(5, 4)
    x = x.view(-1, 1)
    x = x.view(5, 4)

    print("finish")
