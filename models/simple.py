import torch


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.norm = torch.nn.BatchNorm1d(3)
        self.linear = torch.nn.Linear(3,15)
        self.activation = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.softmax(x)
        return x

simple_model = SimpleModel()


