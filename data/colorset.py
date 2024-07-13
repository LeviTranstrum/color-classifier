import json
import torch

class Colorset(torch.utils.data.Dataset):
    def __init__(self, filepath):
        self.transform = ToTensor()
        with open(filepath) as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'color': self.data[idx]['color'], 'label': int(self.data[idx]['label'])}
        return self.transform(sample)


class ToTensor(object):
    def __call__(self, sample):
        color, label = sample['color'], sample['label']
        return {'color': torch.tensor(color, dtype=torch.float32), 'label': torch.tensor(label, dtype=torch.long)}

def test():
    train_data = Colorset('./data/geometric27-narrow/train.json')
    test_data = Colorset('./data/geometric27-narrow/test.json')

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    train_batch = next(iter(train_dataloader))

    print(train_batch['color'])
    print(train_batch['label'])

# test()