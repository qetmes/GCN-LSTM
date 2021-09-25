import argparse
import torch
import torch.nn as nn


class Config():
    def __init__(self) -> None:
        self.model_name = 'Airline'
        self.data_path = '../data/'                                         # 所有数据的存储路径
        self.model_save_path = '../save/Airline.pkl'                         # 模型的存储路径
        self.log_path = '../log/SummaryWriter'

        self.dropout = 0.5
        # self.batch_size = 1000                                             # 批量的大小
        self.shuffle = True                                                # 加载数据时是否随机加载
        self.cuda_is_aviable = False                                        # 是否可以GPU加速
        self.cuda_device = 2                                               # 指定训练的GPU
        self.learning_rate = 1e-4                                          # 学习率的大小
        self.epoch = 100
        self.node_size = 448                                              # 节点个数
        self.input_dim = 44                                  # 时间序列的长度
        self.num_layers = 1
        self.batch_first = True
        self.input_size =                                  # 特征的个数
        self.hidden = 1024
        self.num_classes = self.node_size
        self.last_hidden = 512
        self.hidden2 = 1024
        self.hidden3 = 512
        self.hidden4 = 256


class GCNLSTM(nn.Module):
    def __init__(self, adj, config) -> None:
        super(GCNLSTM, self).__init__()
        self.GCN = GCN(adj, config)
        self.lstm = nn.LSTM(config.input_size * config.node_size, config.hidden,
                            config.num_layers, config.batch_first)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden * config.input_dim, config.hidden2)
        self.fc2 = nn.Linear(config.hidden2, config.hidden3)
        self.fc3 = nn.Linear(config.hidden3, config.hidden4)
        self.fc4 = nn.Linear(config.hidden4, config.num_classes)


    def forward(self, x):
        out = self.GCN(x)
        out, (hn, cn) = self.lstm(out)
        out = self.fc(out.view(out.shape[0], -1))
        out = self.dropout(out)
        out = nn.ReLU()(self.fc2(out))
        out = nn.ReLU()(self.fc3(out))
        out = self.fc4(out)
        return out


class GCN(nn.Module):
    def __init__(self, adj, config):
        super(GCN, self).__init__()
        self.register_buffer(
            "laplacian", calculate_laplacian_with_self_loop(
                torch.FloatTensor(adj))
        )  # 引入邻接矩阵
        self._num_nodes = adj.shape[0]
        self.input_dim = config.input_dim  # 要预测的句子的长度
        self.feature_dim = config.input_size
        self.out_dim = config.hidden  # 输出的隐层的长度
        self.weights = nn.Parameter(
            torch.FloatTensor(self.input_dim, self.out_dim)
        )
        self.fc = nn.Linear(self._num_nodes, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(
            self.weights, gain=nn.init.calculate_gain("tanh"))

    def forward(self, x):

        batch_size = x.shape[0]
        x = x.transpose(2, 3)

        x = x.transpose(0, 3).transpose(1, 3).transpose(2, 3)
        inputs = x.reshape((self._num_nodes, batch_size *
                            self.feature_dim * self.input_dim))

        ax = self.laplacian @ inputs
        ax = ax.reshape((self._num_nodes, batch_size,
                         self.feature_dim, self.input_dim))
        ax = ax.reshape((self._num_nodes * batch_size *
                         self.feature_dim, self.input_dim))
        outputs = ax.reshape(
            (self._num_nodes, batch_size, self.feature_dim, self.input_dim))
        outputs = outputs.transpose(0, 1).transpose(1, 2).transpose(2, 3).transpose(1, 2)
        # batch_size, time_step, hidden_status
        # outputs = nn.ReLU()(self.fc(outputs).squeeze(-1))
        outputs = outputs.reshape(batch_size, self.input_dim, self.feature_dim * self._num_nodes)
        return outputs


def calculate_laplacian_with_self_loop(matrix):
    matrix = matrix + torch.eye(matrix.size(0))
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian
