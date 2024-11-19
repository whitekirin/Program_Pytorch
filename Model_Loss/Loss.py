from torch import nn
from torch.nn import functional
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np


class Entropy_Loss(nn.Module):
    def __init__(self):
        super(Entropy_Loss, self).__init__()

    def forward(self, outputs, labels):
        # 範例: 使用均方誤差作為損失計算
        # outputs = torch.argmax(outputs, 1)
        # outputs = outputs.float()
        labels = labels.float()
        loss = functional.binary_cross_entropy(outputs, labels)
        return loss