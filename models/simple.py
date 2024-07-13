import torch

class SimpleModel(torch.nn.Module):
    def __init__(self, name):
        super(SimpleModel, self).__init__()

        # model architecture
        self.norm = torch.nn.BatchNorm1d(3)
        self.linear = torch.nn.Linear(3,27)

        # optimizer and loss function
        self.lossfunc = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
    
    def get_data

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x








