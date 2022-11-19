# import scipy.stats as stats
# import torch.nn as nn
#
# import torch
# import torch.nn.functional as F
# from torch.nn import Sequential, Linear, ReLU
# from torch_geometric.nn import GINConv, global_add_pool
#
# import numpy as np


# class simclr(nn.Module):
#     def __init__(self, hidden_dim, num_gc_layers, dataset_num, alpha=0.5, beta=1., gamma=.1 ,device= 'cuda'):
#         super(simclr, self).__init__()
#         self.device = device
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#
#         self.dataset_num = dataset_num
#         self.embedding_dim = mi_units = hidden_dim * num_gc_layers
#         # self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)
#
#         self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
#                                        nn.Linear(self.embedding_dim, self.embedding_dim))
#
#         self.init_emb()
#
#         self.data.x = torch.ones((1, 1)) ###########################(number_enconde,1)从别的方法处摘过来
#
#     def init_emb(self):
#         initrange = -1.5 / self.embedding_dim
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 torch.nn.init.xavier_uniform_(m.weight.data)
#                 if m.bias is not None:
#                     m.bias.data.fill_(0.0)
#
#     @torch.no_grad()
#     def sample_negative_index(self, negative_number, epoch, epochs):
#
#         lamda = 1/2
#         # lamda = 1
#         #lamda = 2
#         lower, upper = 0, self.dataset_num
#         mu_1 = ((epoch-1) / epochs) ** lamda * (upper - lower)
#         mu_2 = ((epoch) / epochs) ** lamda * (upper - lower)
#         # sigma = negative_number / 6
#         # # X表示含有最大最小值约束的正态分布
#         # X = stats.truncnorm(
#         #     (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)  # 有区间限制的随机数 正态分布采样
#         # X = stats.uniform(mu_1,mu_2-mu_1)  # 均匀分布采样
#         X = stats.uniform(1,mu_2)
#         index = X.rvs(negative_number)  # 采样
#         index = index.astype(np.int)
#         return index
#
#     def forward(self, x, edge_index, batch, num_graphs):
#
#         # batch_size = data.num_graphs
#         if x is None:
#             x = torch.ones(batch.shape[0]).to(self.device)
#
#         y, M = self.encoder(x, edge_index, batch)
#
#         y = self.proj_head(y)
#
#         return y
#
#     def rank_negative_queue(self, x1, x2):
#
#         x2 = x2.t()
#         x = x1.mm(x2)
#
#         x1_frobenius = x1.norm(dim=1).unsqueeze(0).t()
#         x2_frobenins = x2.norm(dim=0).unsqueeze(0)
#         x_frobenins = x1_frobenius.mm(x2_frobenins)
#
#         final_value = x.mul(1 / x_frobenins)
#
#         sort_queue, _ = torch.sort(final_value, dim=0, descending=False)
#
#         return sort_queue
#
#     def loss_cal(self, q_batch, q_aug_batch, negative_sim):
#
#         T = 0.2
#
#         # q_batch = q_batch[: q_aug_batch.size()[0]]
#
#         positive_sim = torch.cosine_similarity(q_batch, q_aug_batch, dim=1)  # 维度有时对不齐
#
#         positive_exp = torch.exp(positive_sim / T)
#
#         negative_exp = torch.exp(negative_sim / T)
#
#         negative_sum = torch.sum(negative_exp, dim=0)
#
#         loss = positive_exp / (positive_exp+negative_sum)
#
#         loss = -torch.log(loss).mean()
#
#         return loss
#
# class Encoder(torch.nn.Module):
#     def __init__(self, num_features, dim, num_gc_layers):
#         super(Encoder, self).__init__()
#
#         # num_features = dataset.num_features
#         # dim = 32
#         self.num_gc_layers = num_gc_layers
#
#         # self.nns = []
#         self.convs = torch.nn.ModuleList()
#         self.bns = torch.nn.ModuleList()
#
#         for i in range(num_gc_layers):
#
#             if i:
#                 nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
#             else:
#                 nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
#             conv = GINConv(nn)
#             bn = torch.nn.BatchNorm1d(dim)
#
#             self.convs.append(conv)
#             self.bns.append(bn)
#
#     def forward(self, x, edge_index, batch):
#         if x is None:
#             x = torch.ones((batch.shape[0], 1)).to(self.device)
#
#         xs = []
#         for i in range(self.num_gc_layers):
#
#             x = F.relu(self.convs[i](x, edge_index))
#             x = self.bns[i](x)
#             xs.append(x)
#             # if i == 2:
#                 # feature_map = x2
#
#         xpool = [global_add_pool(x, batch) for x in xs]
#         x = torch.cat(xpool, 1)
#
#         return x, torch.cat(xs, 1)
#
#     def get_embeddings(self, loader):
#
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         ret = []
#         y = []
#         with torch.no_grad():
#             for data in loader:
#
#                 data = data[0]
#                 data.to(device)
#                 x, edge_index, batch = data.x, data.edge_index, data.batch
#                 if x is None:
#                     x = torch.ones((batch.shape[0],1)).to(device)
#                 x, _ = self.forward(x, edge_index, batch)
#
#                 ret.append(x.cpu().numpy())
#                 y.append(data.y.cpu().numpy())
#         ret = np.concatenate(ret, 0)
#         y = np.concatenate(y, 0)
#         return ret, y
#
#     def get_embeddings_v(self, loader):
#
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         ret = []
#         y = []
#         with torch.no_grad():
#             for n, data in enumerate(loader):
#                 data.to(device)
#                 x, edge_index, batch = data.x, data.edge_index, data.batch
#                 if x is None:
#                     x = torch.ones((batch.shape[0],1)).to(device)
#                 x_g, x = self.forward(x, edge_index, batch)
#                 x_g = x_g.cpu().numpy()
#                 ret = x.cpu().numpy()
#                 y = data.edge_index.cpu().numpy()
#                 print(data.y)
#                 if n == 1:
#                    break
#
#         return x_g, ret, y
#
# model = simclr(hidden_dim, num_gc_layers,  dataset_num).to(device)
# q_batch = model(data.x, data.edge_index, data.batch, data.num_graphs)
#
# q_aug_batch = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs)
#
# q_batch = q_batch[: q_aug_batch.size()[0]]
#
# sort_queue = model.rank_negative_queue(dataset_embedding, q_batch)
#
# sample_index = model.sample_negative_index(negative_number, epoch, epochs)
#
# sample_index = torch.tensor(sample_index).to(device)
#
# negative_sim = sort_queue.index_select(0, sample_index)
#
# loss = model.loss_cal(q_batch, q_aug_batch, negative_sim)
import pickle
import pprint

file=open("E:\drug-prediction\COGNet-master\src\saved\COGNet\history_COGNet.pkl","rb")
data=pickle.load(file)
pprint.pprint(data)
file.close()