import torch
import numpy as np
import pandas as pd
from model import GCNLSTM, Config
from dataloader import getDataloader


def test_accu(net, data, label):
    with torch.no_grad():
        pre = net(data)
        loss = torch.sum((pre - label) * (pre - label))
        return loss


def test_the_model(airline):
    # test_data = np.load('../data/{airline}.npy')
    # test_label = np.load('../data/{airline}.npy').reshape(-1, 1)
    # test_data = torch.tensor(test_data).type(torch.float32)
    # test_label = torch.tensor(test_label).type(torch.float32)

    dataloader = getDataloader(16000, '../data_deal/dataset/test_data', 32)

    adj = pd.read_csv('../data_deal/adj.pt', header=None)
    adj = torch.tensor(adj)

    config = Config()
    model = GCNLSTM(adj, config)
    state_dict = torch.load(f'{config.model_save_path}net_parameters')
    model.load_state_dict(state_dict)
    predict = model(test_data)
    return torch.sum((predict-test_label)*(predict-test_label))/test_data[0]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-a', type=str, help='airline name',
                        default='AAT_URC')

    args = parser.parse_args()
    test_the_model(args.a)
