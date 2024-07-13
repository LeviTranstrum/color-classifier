import torch
import json
from models import SimpleModel
from data import Colorset
from classifier import Classifier



def main():
    # Load data
    train_data = Colorset('./data/geometric27-narrow/train.json')
    test_data = Colorset('./data/geometric27-narrow/test.json')
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=27, shuffle=False)

    # Set up color classifier
    with open('./classifier/palettes/geometric27-narrow.json') as palette:
        classifier = Classifier(json.load(palette), 'geometric27-narrow')

    # Report split sizes
    print(f'Training set has {len(train_data)} instances')
    print(f'Validation set has {len(test_data)} instances')

    model = SimpleModel()

    def train_one_epoch():
        running_loss = 0.0
        last_loss = 0.0

        for i, data in enumerate(train_dataloader):
            batch = data
            inputs = batch['color']
            labels = batch['label']

            # zero gradients for every batch
            model.optimizer.zero_grad()

            # make predictions for this batch
            outputs = model(inputs)

            # calculate loss and gradients
            loss = model.lossfunc(outputs, labels)
            loss.backward()

            # adjust learning weights
            model.optimizer.step()

            # track loss
            running_loss += loss.item()

            # log
            print(f'batch {i+1} loss: {running_loss / 100}')
            running_loss = 0.0
   
    train_one_epoch()

if __name__=="__main__":
    main()
    
