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

    @classmethod
    def test():
        print('testing Colorset')
        validate_data = Colorset('./data/geometric27-narrow/validate.json')

        validate_dataloader = torch.utils.data.DataLoader(validate_data, batch_size=27, shuffle=False)

        print(f'next iter: {next(iter(validate_dataloader))}')


class ToTensor(object):
    def __call__(self, sample):
        return [torch.tensor(sample[0], dtype=torch.float32), torch.tensor(sample[1], dtype=torch.long)]
