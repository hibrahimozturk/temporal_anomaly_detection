import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class MultiClassBinaryTCN(nn.Module):
    def __init__(self, numClassStages, numBinaryStages, num_layers, num_f_maps, dim, numClasses, ssRepeat=1):
        super(MultiClassBinaryTCN, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, numClasses, repeat=ssRepeat)
        self.multiClassStages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, numClasses, numClasses))
                                               for s in range(numClassStages - 1)])
        self.stage2 = SingleStageModel(num_layers, num_f_maps, numClasses, 1)
        self.binaryStages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, 1, 1))
                                           for s in range(numBinaryStages - 1)])

    def forward(self, x, mask):
        x = x.transpose(2, 1)
        out, lastLayer = self.stage1(x, mask)
        # out = torch.sigmoid(out) * mask[:, 0:1, :]
        out = torch.softmax(out, dim=1) * mask[:, 0:1, :]
        classOutputs = out.unsqueeze(0)
        for s in self.multiClassStages:
            out, lastLayer = s(out, mask)
            # out = torch.sigmoid(out) * mask[:, 0:1, :]
            out = torch.softmax(out, dim=1) * mask[:, 0:1, :]
            classOutputs = torch.cat((classOutputs, out.unsqueeze(0)), dim=0)

        out, lastLayer = self.stage2(out, mask)
        out = torch.sigmoid(out) * mask[:, 0:1, :]
        binaryOutputs = out.unsqueeze(0)
        for s in self.binaryStages:
            out, lastLayer = s(out, mask)
            out = torch.sigmoid(out) * mask[:, 0:1, :]
            binaryOutputs = torch.cat((binaryOutputs, out.unsqueeze(0)), dim=0)
        return classOutputs, binaryOutputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, numClasses, repeat=1):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([])
        for r in range(repeat):
            for i in range(num_layers):
                self.layers.append(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps))
        self.conv_out = nn.Conv1d(num_f_maps, numClasses, 1)

    def forward(self, x, mask, previousLastLayer=None):
        out = self.conv_1x1(x)
        if previousLastLayer is not None:
            out = out + previousLastLayer
        for layer in self.layers:
            out = layer(out, mask)
        lastLayer = out * mask[:, 0:1, :]
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out, lastLayer


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        # self.batchnorm1 = nn.BatchNorm1d(out_channels)
        # self.batchnorm2 = nn.BatchNorm1d(out_channels)
        self.dropout2d = nn.Dropout2d(p=0.3)

    def forward(self, x, mask):
        out = self.conv_dilated(x)
        # out = self.batchnorm1(out)
        out = F.relu(out)
        out = self.conv_1x1(out)
        # out = self.batchnorm2(out)
        out = self.dropout2d(out.unsqueeze(3))
        out = x + out.mean(3)
        # out = F.relu(out)
        out = out * mask[:, 0:1, :]
        return out


if __name__ == "__main__":
    x = torch.rand(1, 20, 1024).float().cuda()
    mask = torch.ones(1, 1, 20).cuda().float()
    mstcn = MultiStageModel(num_stages=2, num_layers=10, num_f_maps=64, dim=1024, ssRepeat=2)
    mstcn = mstcn.cuda()
    mstcn(x, mask)

    print("finish")
