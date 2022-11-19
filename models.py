import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dnc import DNC
from layers import GraphConvolution
from graph_transformer_pytorch import GraphTransformer
from layers import GraphAttentionLayer
import math
from torch.nn.parameter import Parameter

#### 2022-3-27 ********
class SelfAttend(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super(SelfAttend, self).__init__()

        self.h1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh()
        )

        self.gate_layer = nn.Linear(128, 64)

        self.tran = nn.Linear(128,64)

    def forward(self, seqs, seq_masks=None):
        """
        :param seqs: shape [batch_size, seq_length, embedding_size]
        :param seq_lens: shape [batch_size, seq_length]
        :return: shape [batch_size, seq_length, embedding_size]
        """
        gates = self.gate_layer(self.h1(seqs)).squeeze(-1)
        if seq_masks is not None:
            gates = gates + seq_masks
        p_attn = F.softmax(gates, dim=-1)

        # p_attn = p_attn.unsqueeze(-1)
       
        seqs = self.tran(seqs)


        h = seqs * p_attn
        output = torch.sum(h, dim=1)
        return output


class MaskLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__()
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

    def forward(self, input, mask):

        q = self.weight

        weight = torch.mul(self.weight, mask) ## 矩阵对应位相乘，形状必须一致
        output = torch.mm(input, weight)  ## 矩阵相乘，有相同维度即可乘，维度可变

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.cls_vec = nn.Parameter(torch.randn(in_dim))
        self.fc = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):

        weights = torch.matmul(x.view(-1, x.shape[1]), self.cls_vec)
        weights = self.softmax(weights.view(x.shape[0], -1))
        x = torch.bmm(x.view(x.shape[0], x.shape[1], -1), weights.unsqueeze(-1)).squeeze()
        x = x + self.cls_vec
        x = self.fc(x)
        x = x + self.cls_vec
        return x


# class MolecularGraphNeuralNetwork(nn.Module):
#     def __init__(self, N_fingerprint, dim, layer_hidden, device):
#         super(MolecularGraphNeuralNetwork, self).__init__()
#         self.device = device
#         self.embed_fingerprint = nn.Embedding(N_fingerprint, dim).to(self.device)
#         self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim).to(self.device)
#                                             for _ in range(layer_hidden)])
#         self.layer_hidden = layer_hidden
#
#
#
#     def pad(self, matrices, pad_value):
#         """Pad the list of matrices
#         with a pad_value (e.g., 0) for batch proc essing.
#         For example, given a list of matrices [A, B, C],
#         we obtain a new matrix [A00, 0B0, 00C],
#         where 0 is the zero (i.e., pad value) matrix.
#         """
#         shapes = [m.shape for m in matrices]
#         M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
#         zeros = torch.FloatTensor(np.zeros((M, N))).to(self.device)
#         pad_matrices = pad_value + zeros
#         i, j = 0, 0
#         for k, matrix in enumerate(matrices):
#             m, n = shapes[k]
#             pad_matrices[i:i+m, j:j+n] = matrix
#             i += m
#             j += n
#         return pad_matrices
#
#     def update(self, matrix, vectors, layer):
#         hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
#         return hidden_vectors + torch.mm(matrix, hidden_vectors)
#
#     def sum(self, vectors, axis):
#         sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
#         return torch.stack(sum_vectors)
#
#     def mean(self, vectors, axis):
#         mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
#         return torch.stack(mean_vectors)
#
#     def forward(self, inputs):
#
#         """Cat or pad each input data for batch processing."""
#         fingerprints, adjacencies, molecular_sizes = inputs
#         fingerprints = torch.cat(fingerprints)
#         adjacencies = self.pad(adjacencies, 0)
#
#         """MPNN layer (update the fingerprint vectors)."""
#         fingerprint_vectors = self.embed_fingerprint(fingerprints)
#         for l in range(self.layer_hidden):
#             hs = self.update(adjacencies, fingerprint_vectors, l)
#             # fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.
#             fingerprint_vectors = hs
#
#         """Molecular vector by sum or mean of the fingerprint vectors."""
#
#
#         molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
#
#         # molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)
#
#         return molecular_vectors

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2), bias=False),
            nn.BatchNorm1d(outchannel),
            nn.Tanh(),
            nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=int(kernel_size / 2), bias=False),
            nn.BatchNorm1d(outchannel)
        )

        self.use_res = use_res
        if self.use_res:
            self.shortcut = nn.Sequential(
                        nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(outchannel)
                    )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out

## Both GAT and GCN can use the encoder part of graph contrastive learning. Select one
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, adj, device=torch.device('cuda:0')):
        """Dense version of GAT."""
        """
           参数1 ：nfeat   输入层数量
           参数2： nhid    输出特征数量
           参数3： nclass  分类个数
           参数4： dropout dropout概率
           参数5： alpha  激活函数的斜率
           参数6： nheads 多头部分

        """
        super(GAT, self).__init__()
        self.dropout = dropout

        self.x = torch.eye(256).to(device)
        adj = self.normalize(adj + np.eye(256))

        self.adj = torch.FloatTensor(adj).to(device)


        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        # 根据多头部分给定的数量声明attention的数量
        # 将多头的各个attention作为子模块添加到当前模块中
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        # 最后一个attention层，输出的是分类
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self):
        x = F.dropout(self.x, self.dropout, training=self.training)
        x = torch.cat([att(x, self.adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, self.adj))
        return F.log_softmax(x, dim=1)
    def normalize(self, mx):
        """Row-normalize sparse matrix"""


        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        self.w = adj.shape[0]
        adj = self.normalize(adj + np.eye(256))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(256).to(device)

        self.gcn1 = GraphConvolution(256, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, 64) ## graph_transformer 改为16

    def forward(self):

       
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        edge = self.adj

        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""


        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class DGCL(nn.Module):
    def __init__(self, vocab_size, ehr_adj, ddi_adj,   ddi_mask_H,  emb_dim=64,device=torch.device('cuda'),ddi_in_memory=True):
        super(DGCL, self).__init__()
        K = len(vocab_size)
        # self.prescription_net = nn.Sequential(
        #     nn.Linear(112, 112 * 4),
        #     nn.ReLU(),
        #     nn.Linear(112 * 4, 112)
        # )
        self.emb_dim = 64
        self.nhead = 2
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.ddi_in_memory = ddi_in_memory
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(K-1)])
        self.dropout = nn.Dropout(p=0.63)

        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim*2, batch_first=True) for _ in range(K-1)])

        self.e = nn.ModuleList([nn.GRU(64, 64, batch_first=True) for _ in range(K - 1)])
        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
############
        self.MED_PAD_TOKEN = vocab_size[2] + 2
        self.med_embedding = nn.Sequential(
            # 添加padding_idx，表示取0向量
            nn.Embedding(vocab_size[2] + 3, emb_dim, self.MED_PAD_TOKEN),
            nn.Dropout(0.3)
        )
################
        # self.ehr_gat = GAT(256, 64, 256, 0.6, 0.2, 4, adj=ddi_adj )
        # self.ddi_gat = GAT(256, 64, 256, 0.6, 0.2, 4, adj=ddi_adj )
        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))


        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2])
        )

        self.init_weights()

        self.z = nn.Linear(256,112)

        self.tran = nn.Linear(4096,153)

## 2022-3-27##########################################
        # 聚合单个visit内的diag和proc得到visit-level的表达
        # self.diag_self_attend = SelfAttend(32)
        # self.proc_self_attend = SelfAttend(32)

        self.medication_encoder = nn.TransformerEncoderLayer(emb_dim, self.nhead, batch_first=True, dropout=0.2)

        # self.bipartite_transform = nn.Sequential(
        #     nn.Linear(emb_dim, ddi_mask_H.shape[1])
        # )
        # self.bipartite_output = MaskLinear(ddi_mask_H.shape[1], vocab_size[2], False)  ## 是将ddi_mask矩阵中的药物屏蔽掉


        # self.attavg = AttentionPooling(112)
        # graphs, bipartite matrix
        # self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        # self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)
        #
        # self.gat_trans = nn.Linear(256,64)

        self.abc = nn.Linear(153,112)

        # self.ppp = nn.Linear(64,1)

        # self.residual = ResidualBlock(1, 64, 5, 1, True, 0.1)

        self.sample = nn.Linear(64,112)
        self.sample2 = nn.Linear(256,1)

### 2022-3-27 Calculate the distance between different medical records
    def calc_cross_visit_scores(self, visit_diag_embedding, visit_proc_embedding):
        """
        visit_diag_embedding: (batch * visit_num * emb)
        visit_proc_embedding: (batch * visit_num * emb)
        """
        max_visit_num = visit_diag_embedding.size(1)
        batch_size = visit_diag_embedding.size(0)

        new = visit_diag_embedding.size(2)

        # mask表示每个visit只能看到自己之前的visit
        mask = (torch.triu(torch.ones((max_visit_num, max_visit_num), device=self.device)) == 1).transpose(0,
                                                                                                           1)  # 返回一个下三角矩阵
        mask = mask.float().masked_fill(mask == 0, -1e9).masked_fill(mask == 1, float(0.0))
        mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)  # batch * max_visit_num * max_visit_num

        # 每个visit后移一位
        padding = torch.zeros((batch_size, 1, new), device=self.device).float()

        # print(visit_diag_embedding.shape,'qqqqqqqq')
        # print(padding.shape,'wwwwww')

        diag_keys = torch.cat([padding, visit_diag_embedding[:, :-1, :]], dim=1)  # batch * max_visit_num * emb
        proc_keys = torch.cat([padding, visit_proc_embedding[:, :-1, :]], dim=1)

        # 得到每个visit跟自己前面所有visit的score
        diag_scores = torch.matmul(visit_diag_embedding, diag_keys.transpose(-2, -1)) \
                      / math.sqrt(visit_diag_embedding.size(-1))
        proc_scores = torch.matmul(visit_proc_embedding, proc_keys.transpose(-2, -1)) \
                      / math.sqrt(visit_proc_embedding.size(-1))
        # 1st visit's scores is not zero!
        scores = F.softmax(diag_scores + proc_scores + mask, dim=-1)

        ###### case study
        # 将第0个val置0，然后重新归一化
        # scores_buf = scores
        # scores_buf[:, :, 0] = 0.
        # scores_buf = scores_buf / torch.sum(scores_buf, dim=2, keepdim=True)

        # print(scores_buf)
        return scores


    def forward(self, input):
        # input (adm, 3, codes)
        device = self.device
        # generate medical embeddings and queries
        i1_seq = []
        i2_seq = []
        i3_seq = []

        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)
        for adm in input:

            i1 = mean_embedding(self.dropout(self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)))) # (1,1,dim)
            i2 = mean_embedding(self.dropout(self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))
            i3 = mean_embedding(self.dropout(self.embeddings[0](torch.LongTensor(adm[2]).unsqueeze(dim=0).to(self.device))))


            ##  此处adm[0]代表diagnosis文件，adm[1]代表procedure文件；
            i1_seq.append(i1)
            i2_seq.append(i2)
            i3_seq.append(i3)
        i1_seq = torch.cat(i1_seq, dim=1) #(1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1) #(1,seq,dim)
        i3_seq = torch.cat(i3_seq, dim=1)

        o1, h1 = self.encoders[0](
            i1_seq
        ) # o1:(1, seq, dim*2) hi:(1,1,dim*2)
        o2, h2 = self.encoders[1](
            i2_seq
        )
        o3, h3 = self.e[1](
            i3_seq
        )

## 2022-3-27 we pay attention to the diagnosis and the procedure  separately, and then enter the distance network
        visit_diag_embedding = o1.view(64, 1, -1)

        visit_proc_embedding = o2.view(64, 1, -1)

        cross_visit_scores = self.calc_cross_visit_scores(visit_diag_embedding, visit_proc_embedding)

        last_seq_medication = h3

        last_seq_medication = last_seq_medication.long()


        last_seq_medication_emb = self.med_embedding(last_seq_medication)

        last_seq_medication_emb = torch.squeeze(last_seq_medication_emb)


        encoded_medication = self.medication_encoder(last_seq_medication_emb,src_mask=None)  # (batch*seq, max_med_num, emb_dim)


        prob_g = F.softmax(cross_visit_scores, dim=-1)

        prob_c_to_g = torch.zeros_like(prob_g).to(self.device) #

        patient_representations = torch.cat([o1, o2], dim=-1).squeeze(dim=0) # (seq, dim*4)

        queries = self.query(patient_representations) # (seq, dim)


        query = queries[-1:] # (1,dim)

## 2022-4-18 Graph coding of EHR and ddi knowledge to prepare for graph contrastive
        knowledge_graph = self.ddi_gcn()
        ehr_graph = self.ehr_gcn()

        drug_memory = ehr_graph.t()
        drug_memory = self.z(drug_memory)
        drug_memory = drug_memory.t()




## graph sample
        knowledge_graph = self.sample(knowledge_graph)
        ehr_graph = self.sample(ehr_graph)

        knowledge_graph=knowledge_graph.t()
        ehr_graph=ehr_graph.t()

        knowledge_graph = self.sample2(knowledge_graph)
        ehr_graph = self.sample2(ehr_graph)

        fin_k = knowledge_graph.t()
        fin_e = ehr_graph.t()

##  History query section, learn the GAMENet project code
        if len(input) > 1:
            history_keys = queries[:(queries.size(0)-1)] # (seq-1, dim)

            history_values = np.zeros((len(input)-1, self.vocab_size[2]))
            for idx, adm in enumerate(input):
                if idx == len(input)-1:
                    break
                history_values[idx, adm[2]] = 1
            history_values = torch.FloatTensor(history_values).to(self.device) # (seq-1, size)

        fun2 = torch.mm(query, drug_memory.t())


        key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # (1, size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (1, dim)
        w1 = F.sigmoid(fun2)
        w2 = 1 - w1


        if len(input) > 1:
            visit_weight = F.softmax(torch.mm(query, history_keys.t())) # (1, seq-1)
            weighted_values = visit_weight.mm(history_values) # (1, size)
           
            fact2 = torch.mm(weighted_values, drug_memory) # (1, dim)

        else:
            fact2 = fact1

        output = self.output(torch.cat([query, fact1, fact2], dim=-1)) # (1, dim)

        prob_g = prob_g.squeeze(dim=1)

        b = prob_g * encoded_medication

        b = b.view(1,-1)
        b = self.tran(b)


        b = self.abc(b)

        # a = 0.9*output + 1.85*b + result

        a = w1*( 0.9*output)+ 1.85*b
        # a = output , drug


        if self.training:


            return a,fin_e,fin_k
        else:

            return a

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

        self.inter.data.uniform_(-initrange, initrange)

