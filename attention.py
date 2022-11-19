# import torch
# import torch.nn.functional as F
#
# class StructuredSelfAttention(nn.moudal):
#
#     def __init__(self, batch_size, lstm_hid_dim, d_a, n_classes, label_embed, embeddings):
#         super(StructuredSelfAttention, self).__init__()
#         self.n_classes = n_classes
#         self.embeddings = self._load_embeddings(embeddings)
#         self.label_embed = self.load_labelembedd(label_embed)
#         self.lstm = torch.nn.LSTM(300, hidden_size=lstm_hid_dim, num_layers=1,
#                                   batch_first=True, bidirectional=True)
#         self.linear_first = torch.nn.Linear(lstm_hid_dim * 2, d_a)
#         self.linear_second = torch.nn.Linear(d_a, n_classes)
#
#         self.weight1 = torch.nn.Linear(lstm_hid_dim * 2, 1)
#         self.weight2 = torch.nn.Linear(lstm_hid_dim * 2, 1)
#
#         self.output_layer = torch.nn.Linear(lstm_hid_dim * 2, n_classes)
#         self.embedding_dropout = torch.nn.Dropout(p=0.3)
#         self.batch_size = batch_size
#         self.lstm_hid_dim = lstm_hid_dim
#
#     def _load_embeddings(self, embeddings):
#         """Load the embeddings based on flag"""
#         word_embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
#         word_embeddings.weight = torch.nn.Parameter(embeddings)
#         return word_embeddings
#
#     def load_labelembedd(self, label_embed):
#         """Load the embeddings based on flag"""
#         embed = torch.nn.Embedding(label_embed.size(0), label_embed.size(1))
#         embed.weight = torch.nn.Parameter(label_embed)
#         return embed
#
#     def init_hidden(self):
#         return (torch.randn(2, self.batch_size, self.lstm_hid_dim).cuda(),
#                 torch.randn(2, self.batch_size, self.lstm_hid_dim).cuda())
#
#     def forward(self, x):
#         embeddings = self.embeddings(x)
#         embeddings = self.embedding_dropout(embeddings)
#         # step1 get LSTM outputs
#         hidden_state = self.init_hidden()
#         # print(hidden_state,'____________________')
#         outputs, hidden_state = self.lstm(embeddings, hidden_state)
#         # step2 get self-attention
#         selfatt = torch.tanh(self.linear_first(outputs))
#         selfatt = self.linear_second(selfatt)
#         selfatt = F.softmax(selfatt, dim=1)
#         print(selfatt.shape, '+++++++++++++++++++++++++++++++++++++++++++')
#         selfatt = selfatt.transpose(1, 2)
#         # print(selfatt.shape,')))))))))))))')
#         # print(outputs,outputs.shape)
#         self_att = torch.bmm(selfatt, outputs)
#         # step3 get label-attention
#         h1 = outputs[:, :, :self.lstm_hid_dim]
#         h2 = outputs[:, :, self.lstm_hid_dim:]
#
#         label = self.label_embed.weight.data
#         m1 = torch.bmm(label.expand(self.batch_size, self.n_classes, self.lstm_hid_dim), h1.transpose(1, 2))
#         # print(label.expand(self.batch_size, self.n_classes, self.lstm_hid_dim).shape)
#         # print(h2.transpose(1, 2).shape,'+++++++++++++++++++++++++++++++++')
#         # print(m1.shape)
#         m2 = torch.bmm(label.expand(self.batch_size, self.n_classes, self.lstm_hid_dim), h2.transpose(1, 2))
#         label_att = torch.cat((torch.bmm(m1, h1), torch.bmm(m2, h2)), 2)
#
#         # label_att = F.normalize(label_att, p=2, dim=-1)
#         # self_att = F.normalize(self_att, p=2, dim=-1) #all can
#         weight1 = torch.sigmoid(self.weight1(label_att))
#         weight2 = torch.sigmoid(self.weight2(self_att))
#         weight1 = weight1 / (weight1 + weight2)
#         weight2 = 1 - weight1
#
#         doc = weight2 * self_att  ########################## doc = weight1*label_att+weight2*self_att
#         # there two method, for simple, just add
#         # also can use linear to do it
#         avg_sentence_embeddings = torch.sum(doc, 1) / self.n_classes
#
#         pred = torch.sigmoid(self.output_layer(avg_sentence_embeddings))
#         return pred

from graphviz import Digraph

import torch
import models


def make_dot(var, params):
    """ Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    param_map = {id(v): k for k, v in params.items()}
    print(param_map)

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot


inputs = torch.randn(100, 50).cuda()
adj = torch.randn(100, 100).cuda()
model = models.SpGAT(50, 8, 7, 0.5, 0.01, 3)
model = model.cuda()
y = model(inputs, adj)

g = make_dot(y, model.state_dict())
g.view()
