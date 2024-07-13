import json
import torch
from torch.utils.data import Dataset

class Colorset(Dataset):
    def __init__(self, filepath, transform):
        with open(filepath) as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'color': self.data[idx].color, 'label': self.data[idx].label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        color, label = sample.color, sample.label
        return {'color': torch.tensor(color), 'label': torch.tensor(label)}

colorset = Colorset('./data/geometric27-narrow.json', transform=ToTensor())
for i, sample in enumerate(colorset):
    print(i, sample)
    if i > 5:
        break