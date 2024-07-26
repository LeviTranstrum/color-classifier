import torch
import os
from model.colorset import Colorset
from generator.generator import Generator

class LargeModel(torch.nn.Module):
    def __init__(self, dataset_name):
        super(LargeModel, self).__init__()

        self.name = dataset_name
        # generator
        self.generator = Generator(dataset_name)

        # model architecture
        # Batch norm has learnable parameters that trivialize the model architecture for this problem
        self.norm1 = torch.nn.BatchNorm1d(3)
        self.linear1 = torch.nn.Linear(3, 6)
        self.norm2 = torch.nn.BatchNorm1d(6)
        self.linear2 = torch.nn.Linear(6, 12)
        self.norm3 = torch.nn.BatchNorm1d(12)
        self.linear3 = torch.nn.Linear(12, len(self.generator.palette))
        self.activation = torch.nn.ReLU()

        # optimizer and loss function
        self.lossfunc = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        # datasets
        train_path = f'./data/{dataset_name}/train.json'
        validate_path = f'./data/{dataset_name}/validate.json'
        if not os.path.exists(train_path) or not os.path.exists(validate_path):
            self.generator.generate_data()
        self.training_set = Colorset(train_path)
        self.validation_set = Colorset(validate_path)

    def forward(self, x):
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.norm2(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.norm3(x)
        x = self.linear3(x)
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

            self.save()
            if visualize:
                self.run_validation(f'epoch {n+1}')

    def run_validation(self, title=None):
        self.eval()
        validation_dataloader = torch.utils.data.DataLoader(self.validation_set,
                                                            batch_size=len(self.generator.palette),
                                                            shuffle=False)

        for i, data in enumerate(validation_dataloader):
            batch = data
            inputs = batch[0]

            outputs = self(inputs)

            colors = []
            results = []
            for i in range(len(self.generator.palette)):
                colors.append(inputs[i])
                results.append(self.generator.index_to_label(torch.argmax(outputs[i]).item()))

            self.generator.visualize(colors, results, title)

    def test(self, num_samples = 100):
        self.eval()
        colors = torch.tensor([self.generator.random_color() for _ in range(num_samples)], dtype=torch.float32)
        model_outputs = self(colors)
        results = [self.generator.index_to_label(torch.argmax(model_outputs[i]).item()) for i in range(num_samples)]
        self.generator.visualize(colors, results, f'{self.name} Model Test')

    def save(self):
        path = f'./trained_models/large/{self.name}'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), f'{path}')

    @classmethod
    def load(cls, dataset_name):
        path = f'./trained_models/large/{dataset_name}'
        model = LargeModel(dataset_name)
        try:
            model.load_state_dict(torch.load(path))
        except:
            print(f'No trained model found with the name {dataset_name}. Creating a new model')
        return model







