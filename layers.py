import torch
import math
import torch.nn as nn

import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    带有attention计算的网络层

    参数：in_features 输入节点的特征数F
    参数：out_features 输出的节点的特征数F'
    参数：dropout
    参数：alpha LeakyRelu激活函数的斜率
    参数：concat
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features  # 输入特征数
        self.out_features = out_features  # 输出特征数
        self.alpha = alpha  # 激活斜率 (LeakyReLU)的激活斜率
        self.concat = concat  # 用来判断是不是最后一个attention # if this layer is not last layer,

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))  # 建立一个w权重，用于对特征数F进行线性变化
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 对权重矩阵进行初始化 服从均匀分布的Glorot初始化器
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))  # 计算函数α，输入是上一层两个输出的拼接，输出的是eij，a的size为(2*F',1)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)  # 激活层

    # 前向传播过程
    def forward(self, h, adj):
        '''
        参数h：表示输入的各个节点的特征矩阵
        参数adj ：表示邻接矩阵
        '''
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        # 线性变化特征的过程,Wh的size为(N,F')，N表示节点的数量，F‘表示输出的节点的特征的数量
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)  # 生成一个矩阵，size为(N,N)
        attention = torch.where(adj > 0, e, zero_vec)
        # 对于邻接矩阵中的元素，>0说明两种之间有边连接，就用e中的权值，否则表示没有边连接，就用一个默认值来表示
        attention = F.softmax(attention, dim=1)
        # 做一个softmax，生成贡献度权重
        attention = F.dropout(attention, self.dropout, training=self.training)
        # 根据权重计算最终的特征输出。
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)  # 做一次激活
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        '''
            #下面是self-attention input ，构建自我的特征矩阵
            #matmul 的size为(N,1)表示eij对应的数值
            #e的size为(N,N)，每一行表示一个节点，其他各个节点对该行的贡献度
            '''

    # Wh.shape (N, out_feature)
    # self.a.shape (2 * out_feature, 1)
    # Wh1&2.shape (N, 1)
    # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # 矩阵乘法
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        a = self.leakyrelu(e)
        return a


# 打印输出类名称，输入特征数量，输出特征数量
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

