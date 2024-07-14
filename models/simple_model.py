import torch
import os
from models.colorset import Colorset
from classifier.classifier import Classifier


class SimpleModel(torch.nn.Module):
    def __init__(self, dataset_name):
        super(SimpleModel, self).__init__()

        # classifier
        self.classifier = Classifier(dataset_name)

        # model architecture
        self.norm = torch.nn.BatchNorm1d(3)
        self.linear = torch.nn.Linear(3, len(self.classifier.palette))

        # optimizer and loss function
        self.lossfunc = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        # datasets
        train_path = f'./data/{dataset_name}/train.json'
        validate_path = f'./data/{dataset_name}/validate.json'
        if not os.path.exists(train_path) or not os.path.exists(validate_path):
            self.classifier.generate_data()
        self.training_set = Colorset(train_path)
        self.validation_set = Colorset(validate_path)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x

    def run_training(self, num_epochs = 1, visualize = False):
        train_dataloader = torch.utils.data.DataLoader(self.training_set, batch_size=64, shuffle=True)

        running_loss = 0.0

        if visualize:
            self.run_validation('epoch 0')

        for n in range(num_epochs):
            self.train()
            running_loss = 0
            for i, data in enumerate(train_dataloader):
                batch = data
                inputs = batch[0]
                labels = batch[1]

                # zero gradients for every batch
                self.optimizer.zero_grad()

                # make predictions for this batch
                outputs = self(inputs)

                # calculate loss and gradients
                loss = self.lossfunc(outputs, labels)
                loss.backward()

                # adjust learning weights
                self.optimizer.step()

                # track loss
                running_loss += loss.item()

                # log
                if i % 100 == 99:
                    print(f'epoch {n+1} batch {i+1} average loss: {running_loss / 100}')
                    running_loss = 0.0

            if visualize:
                self.run_validation(f'epoch {n+1}')

    def run_validation(self, title = None):
        self.eval()
        validation_dataloader = torch.utils.data.DataLoader(self.validation_set,
                                                            batch_size=len(self.classifier.palette),
                                                            shuffle=True)

        for i, data in enumerate(validation_dataloader):
            batch = data
            inputs = batch[0]

            outputs = self(inputs)

            colors = []
            results = []
            for i in range(len(self.classifier.palette)):
                colors.append(inputs[i])
                results.append(self.classifier.index_to_label(torch.argmax(outputs[i]).item()))

            self.classifier.visualize(colors, results, title)

    @classmethod
    def test():
        print('testing SimpleModel')
        model = SimpleModel('geometric27-narrow')
        print(f'model params: {model.parameters()}')







