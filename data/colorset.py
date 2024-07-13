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
        sample = [self.data[idx][0], self.data[idx][1]]
        return self.transform(sample)


class ToTensor(object):
    def __call__(self, sample):
        return [torch.tensor(sample[0], dtype=torch.float32), torch.tensor(sample[1], dtype=torch.long)]

def test():
    train_data = Colorset('./data/geometric27-narrow/train.json')
    test_data = Colorset('./data/geometric27-narrow/test.json')

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    print(f'next iter: {next(iter(train_dataloader))}')

test()