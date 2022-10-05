import torch
import numpy as np


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.lstm = torch.nn.LSTM(3, 128, num_layers=1, batch_first=True)
        self.pitch_linear = torch.nn.Linear(128, 128)
        self.pitch_sigmoid = torch.nn.Sigmoid()
        self.step_linear = torch.nn.Linear(128, 1)
        self.duration_linear = torch.nn.Linear(128, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        pitch = self.pitch_sigmoid(self.pitch_linear(x[:, -1]))
        step = self.step_linear(x[:, -1])
        duration = self.duration_linear(x[:, -1])
        return {'pitch': pitch, 'step': step, 'duration': duration}


def mse_with_positive_pressure(pred, y):
    mse = (y-pred) ** 2
    positive_pressure = 10*torch.maximum(-pred, torch.tensor(0))
    return torch.mean(mse+positive_pressure)


class MyLoss(torch.nn.Module):
    def __init__(self, weight):
        super(MyLoss, self).__init__()
        self.weight = torch.Tensor(weight)
        self.pitch_loss=torch.nn.CrossEntropyLoss()
        self.step_loss=mse_with_positive_pressure
        self.duration_loss=mse_with_positive_pressure

    def forward(self, pred, y):
        a = self.pitch_loss(pred['pitch'], y['pitch'])
        b = self.step_loss(pred['step'], y['step'])
        c = self.duration_loss(pred['duration'], y['duration'])
        return a*self.weight[0]+b*self.weight[1]+c*self.weight[2]
