import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(SimpleNet, self).__init__()
        self.Conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),

            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.Classes = torch.nn.Sequential(
            torch.nn.Linear(16 * 16 * 256, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(512, output_channel),
            torch.nn.Softmax(dim=1))

    def forward(self, inputs):
        x = self.Conv(inputs)
        x = x.view(-1, 16 * 16 * 256)
        x = self.Classes(x)

        return x
