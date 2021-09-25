from torch.utils.data import Dataset, DataLoader
import torch

class MyDataset(Dataset):
    def __init__(self, n, path):
        self.n = n
        self.path = path

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x, y  = torch.load(f'{self.path}/data{i}.pt')
        return x, y

def getDataloader(n, path, batch_size, shuffle=True):
    my_dataset = MyDataset(n, path)
    return DataLoader(my_dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == '__main__':
    dataloader = getDataloader(64000, '../data_deal/dataset/train_data', 32)
    for sample, label in dataloader:
        print(sample.shape)
        print(label.shape)