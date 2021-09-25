import torch
import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, TensorDataset
from model import GCNLSTM, Config
from test import test_accu
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from dataloader import getDataloader
writer = SummaryWriter('../log/')

adj = torch.load('../data_deal/adj.pt')

config = Config()


def train():
    net = GCNLSTM(adj, config)
    # data = np.load('../data.npy')
    # label = np.load('../label.npy').reshape(-1, 1)
    # test_data = np.load('../data/testdata.npy')
    # test_label = np.load('../data/testlabel.npy').reshape(-1, 1)
    #
    # test_data = torch.tensor(test_data).type(torch.float32)
    # test_label = torch.tensor(test_label).type(torch.float32)
    #
    # data = torch.tensor(data).type(torch.float32)
    # label = torch.tensor(label).type(torch.float32)
    #
    # data = TensorDataset(data, label)
    # data_getloader = DataLoader(data, config.batch_size, config.shuffle)
    dataloader = getDataloader(64000, '../data_deal/dataset/train_data', 32)


    loss_function = nn.MSELoss()
    # if config.cuda_is_aviable:
    #     net = net.cuda(device=config.cuda_device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)

    i = 0
    for epoch in range(config.epoch):
        total = 0
        for data, label in dataloader:
            # if data.shape[0] != config.batch_size:
            #     continue
            i += 1
            pre = net(data)

            loss = loss_function(pre, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += loss.item()
            print("epoch: %s, loss: %s" % (epoch, loss.item()/data.shape[0]))
            writer.add_scalar('itear_loss', i, loss.item()/data.shape[0])
        # acc = test_accu(net, test_data, test_label)
        # print(acc.item())
        writer.add_scalar('epoch_loss', epoch, total)
    stat_dict = {'net': net.state_dict()}
    torch.save(stat_dict, config.save_path + 'net_parameters')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='param ')
    parser.add_argument('-batch', type=int, help='batch size', default=1000)
    parser.add_argument('-l', type=float, help='learning rate', default=0.0001)
    parser.add_argument('-epoch', type=int, help='epoch', default=100)
    parser.add_argument('-GPU', type=int, help='useing GPU ??', default=1)

    args = parser.parse_args()
    config.batch_size = args.batch
    config.l = args.l
    config.epoch = args.epoch
    config.cuda_is_aviable = args.GPU
    train()
