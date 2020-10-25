import torch
import torch.nn as nn
from models.MLP import MLP


class EDTCN(nn.Module):
    def __init__(self, kernelSize=11, featureSize=1024):
        super().__init__()
        assert kernelSize % 2 == 1, "kernel should be odd"
        self.conv1 = nn.Conv1d(featureSize, 64, kernelSize, padding=kernelSize//2)
        self.maxpool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 96, kernelSize, padding=kernelSize//2)
        self.maxpool2 = nn.MaxPool1d(2)
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv3 = nn.Conv1d(96, 64, kernelSize, padding=kernelSize//2)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv4 = nn.Conv1d(64, featureSize, kernelSize, padding=kernelSize//2)
        self.classifier = MLP(featureSize)
        self.featureSize = featureSize

    def forward(self, x):
        # Batchsize: N, Steps: S, Channels=C
        # N, S, C -> # N, C, S
        assert x.shape[1] % 4 == 0, "data will be pooled twice, it should be divisible to 4"
        x = x.transpose(2, 1)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.upsample1(x)
        x = self.conv3(x)
        x = self.upsample2(x)
        x = self.conv4(x)
        # N, C, S -> # N, S, C
        x = x.transpose(2, 1)
        batchSize = x.shape[0]
        x = x.reshape(-1, self.featureSize)
        x = self.classifier(x)
        x = x.view(batchSize, -1)

        return x


if __name__ == "__main__":
    x = torch.rand(1, 20, 1024)
    edtcn = EDTCN(kernelSize=11)
    edtcn(x)

    print("finish")

