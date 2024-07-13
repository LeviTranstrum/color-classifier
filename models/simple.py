import torch
import os
from data import Colorset
from classifier import Classifier


class SimpleModel(torch.nn.Module):
    def __init__(self, dataset_name):
        super(SimpleModel, self).__init__()

        # model architecture
        self.norm = torch.nn.BatchNorm1d(3)
        self.linear = torch.nn.Linear(3,27)

        # optimizer and loss function
        self.lossfunc = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
    
        # classifier
        self.classifier = Classifier(dataset_name)
        
        # datasets
        train_path = f'./data/{dataset_name}/train.json'
        test_path = f'./data/{dataset_name}/test.json'
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            self.classifier.generate_data()
        self.training_set = Colorset(train_path)
        self.validation_set = Colorset(test_path)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x








