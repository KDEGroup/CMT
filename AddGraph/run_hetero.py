# coding=utf-8
import os, argparse
import sys

sys.path.append("/home/zqxu/MHTGNN/code/")

from Utils import set_seed, Logging
from Metrics import *
from Input import loadXinyeDataHeteroDynamic
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import dgl, dgl.nn.pytorch.conv, dgl.nn.pytorch.hetero
import dgl.function as fn
import numpy as np
import pandas as pd
import pickle
from time import time
from tqdm import tqdm
import random
import torch.optim

log_dir = "/home/zqxu/MHTGNN/log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir, 'Addgraph_hetero.log')
log = Logging(log_path)


parser = argparse.ArgumentParser(description='Addgraph')
parser.add_argument("--start", type=int, default=1)
parser.add_argument("--hid_dim", type=int, default=64, help="hidden layer dimension")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for regularization")
parser.add_argument("--epochs", type=int, default=100, help="Epochs for training")
parser.add_argument("--agg_fn", type=str, default='mean', help="Aggregation function for RGCN")
parser.add_argument("--agg_room", type=str, default='mean', help="Aggregation for room feature definition")
parser.add_argument("--w", type=int, default=3, help="Historical windows size")
parser.add_argument("--batch_size", type=int, default=1024*8, help="Batch size")
parser.add_argument("--path", type=str, default="/home/zqxu/MHTGNN/data/", help="Dataset path")

args = parser.parse_args()
log.record(args)

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, rel_names, agg_fn):
        super().__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.embed = nn.Linear(in_feats, hid_feats)
        self.conv1 = dgl.nn.pytorch.hetero.HeteroGraphConv(mods={
            rel: dgl.nn.pytorch.conv.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate=agg_fn)
        self.conv2 = dgl.nn.pytorch.hetero.HeteroGraphConv(mods={
            rel: dgl.nn.pytorch.conv.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate=agg_fn)

        self.predict = nn.Sequential(
            nn.Linear(hid_feats, 1),
            nn.Sigmoid()
        )

    def forward(self, blocks, x):
        # inputs are features of nodes
        x['user'] = self.embed(x['user'])
        h = self.conv1(blocks[0], x)
        h = {k: F.relu(v, inplace=True) for k, v in h.items()}
        h = self.conv2(blocks[1], h)['user']
        return h

# borrowed from https://github.com/dmlc/dgl/blob/306e0a46e182f3bf3bea717688ed82224c121276/examples/pytorch/han/model_hetero.py#L17
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)                       # (N, D * K)

class AddGraph(nn.Module):
    def __init__(self, in_feats, hid_feats, rel_names, agg_fn):
        super(AddGraph, self).__init__()
        self.rgcn = RGCN(in_feats, hid_feats, rel_names, agg_fn)
        self.gru = GRU(hid_feats)
        self.cab = HCA(hid_feats)
        self.embed = nn.Linear(in_feats, hid_feats)
        self.score = Score(hid_feats)

class HCA(nn.Module):
    def __init__(self, hidden):
        super(HCA, self).__init__()
        self.hidden = hidden
        self.dropout = 0.2
        self.Q = Parameter(torch.FloatTensor(hidden, hidden))
        self.r = Parameter(torch.FloatTensor(hidden))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt((self.Q.size(0)))
        self.Q.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.r.size(0))
        self.r.data.uniform_(-stdv, stdv)

    def forward(self, C):
        # [w, num_nodes, hid_dim] -> [num_nodes, w, hid_dim]
        C_ = C.permute(1, 0, 2)
        # [num_nodes, w, hid_dim] -> [num_nodes, hid_dim, w]
        C_t = C_.permute(0, 2, 1)
        e_ = torch.einsum('ih,nhw->niw', self.Q, C_t)
        e_ = F.dropout(e_, self.dropout, training=self.training)
        e = torch.einsum('h,nhw->nw', self.r, torch.tanh(e_))
        e = F.dropout(e, self.dropout, training=self.training)
        a = F.softmax(e, dim=1)
        short = torch.einsum('nw,nwh->nh', a, C_)
        return short

    def loss(self):
        loss = torch.norm(self.Q, 2).pow(2) + torch.norm(self.r, 2).pow(2)
        return loss


class GRU(nn.Module):
    def __init__(self, hid_feats):
        super(GRU, self).__init__()
        self.dropout = nn.Dropout(p=0.2, inplace=True)
        self.u_p = Parameter(torch.FloatTensor(hid_feats, hid_feats))
        self.w_p = Parameter(torch.FloatTensor(hid_feats, hid_feats))
        self.b_p = Parameter(torch.FloatTensor(hid_feats))
        self.u_r = Parameter(torch.FloatTensor(hid_feats, hid_feats))
        self.w_r = Parameter(torch.FloatTensor(hid_feats, hid_feats))
        self.b_r = Parameter(torch.FloatTensor(hid_feats))
        self.u_c = Parameter(torch.FloatTensor(hid_feats, hid_feats))
        self.w_c = Parameter(torch.FloatTensor(hid_feats, hid_feats))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.u_p.size(0))
        self.u_p.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.w_p.size(0))
        self.w_p.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.b_p.size(0))
        self.b_p.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.u_r.size(0))
        self.u_r.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.w_r.size(0))
        self.w_r.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.b_r.size(0))
        self.b_r.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.u_c.size(0))
        self.u_c.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.w_c.size(0))
        self.w_c.data.uniform_(-stdv, stdv)

    def forward(self, current, short):
        P = torch.sigmoid(torch.matmul(current, self.u_p) + torch.matmul(short, self.w_p) + self.b_p)
        # self.dropout(P)
        R = torch.sigmoid(torch.matmul(current, self.u_r) + torch.matmul(short, self.w_r) + self.b_r)
        # self.dropout(R)
        Htn = torch.tanh(torch.matmul(current, self.u_c) + R * torch.matmul(short, self.w_c))
        # self.dropout(Htn)
        R = short * (1 - P)
        P = P * Htn
        Ht = P + R
        # Ht = short * (1 - P) + P * Htn
        return Ht

    def loss(self):
        loss1 = torch.norm(self.u_p, 2).pow(2) + torch.norm(self.w_p, 2).pow(2) + torch.norm(self.b_p, 2).pow(2)
        loss2 = torch.norm(self.u_r, 2).pow(2) + torch.norm(self.w_r, 2).pow(2) + torch.norm(self.b_r, 2).pow(2)
        loss3 = torch.norm(self.u_c, 2).pow(2) + torch.norm(self.w_c, 2).pow(2)
        return (loss1 + loss2 + loss3)

class Score(nn.Module):
    def __init__(self, hid_feats):
        super(Score, self).__init__()
        self.predict = nn.Sequential(
            nn.Linear(hid_feats, 1),
            nn.Sigmoid()
        )

    def forward(self, h):
        return self.predict(h)


def run():
    set_seed(42)
    hetero_graph, node_feature, n_hetero_features, train_user_idx, test_user_idx, edges, label = loadXinyeDataHeteroDynamic()
    model = AddGraph(n_hetero_features, args.hid_dim, hetero_graph.etypes, args.agg_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss(reduction='sum')

    timespan = 7
    num_nodes = hetero_graph.num_nodes('user')

    bestepoch = 0
    bestauc = 0.
    for epoch in range(1, args.epochs+1):
        H_list = torch.zeros(1, num_nodes, args.hid_dim)
        for k in range(args.w - 1):
            H_list = torch.cat([H_list, torch.zeros(num_nodes, args.hid_dim).unsqueeze(0)], dim=0)
        for t in range(args.start, timespan+1):
            edge = edges[edges['ts'] <= t]
            uid_set = set(edge['src'].values.tolist() + edge['dst'].values.tolist())

            train_user_idx_t = pd.Series(train_user_idx)
            train_user_idx_t = train_user_idx_t[train_user_idx_t.isin(uid_set)].tolist()
            train_user_idx_t = np.array(train_user_idx_t)
            test_user_idx_t = pd.Series(test_user_idx)
            test_user_idx_t = test_user_idx_t[test_user_idx_t.isin(uid_set)].tolist()
            test_user_idx_t = np.array(test_user_idx_t)
            random.shuffle(train_user_idx_t)
            random.shuffle(test_user_idx_t)
            edges_one = edge[edge['type'] == 1].drop(columns=['type', 'ts']).to_numpy()
            edges_two = edge[edge['type'] == 2].drop(columns=['type', 'ts']).to_numpy()
            edges_three = edge[edge['type'] == 3].drop(columns=['type', 'ts']).to_numpy()

            one_src, one_dst = torch.tensor(edges_one[:, 0]), torch.tensor(edges_one[:, 1])
            two_src, two_dst = torch.tensor(edges_two[:, 0]), torch.tensor(edges_two[:, 1])
            three_src, three_dst = torch.tensor(edges_three[:, 0]), torch.tensor(edges_three[:, 1])
            
            hg = dgl.heterograph(
                data_dict={
                    ('user', 'one', 'user'): (one_src, one_dst),
                    ('user', 'one_by', 'user'): (one_dst, one_src),
                    ('user', 'two', 'user'): (two_src, two_dst),
                    ('user', 'two_by', 'user'): (two_dst, two_src),
                    ('user', 'three', 'user'): (three_src, three_dst),
                    ('user', 'three_by', 'user'): (three_dst, three_src)
                },
                num_nodes_dict={
                    'user': node_feature.shape[0]
                }
            )
            hg.nodes['user'].data['feature'] = torch.FloatTensor(node_feature)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
            dataloader_train = dgl.dataloading.NodeDataLoader(hg, {'user': train_user_idx_t}, sampler, batch_size=args.batch_size, shuffle=True, drop_last=True)
            dataloader_test = dgl.dataloading.NodeDataLoader(hg, {'user': test_user_idx_t}, sampler, batch_size=args.batch_size, shuffle=False, drop_last=True)

            model.train()
            optimizer.zero_grad()
            train_loss = 0.
            results = []
            Ht = torch.zeros(num_nodes, args.hid_dim)
            for input_nodes, output_nodes, blocks in tqdm(dataloader_train):
                batch_size = output_nodes['user'].shape[0]
                user_feature = blocks[0].nodes['user'].data['feature']
                node_features = {'user': user_feature}
                current = model.rgcn(blocks, node_features)
                short = torch.zeros((args.w, batch_size, args.hid_dim))
                for j in range(args.w):
                    short[j] = H_list[-args.w + j][output_nodes['user'].long()]
                # short = torch.mean(short, dim=0)
                short = model.cab(short)
                h = model.gru(short, current)
                logits = model.score(h)
                prob = logits
                pred = torch.where(prob > 0.5, 1, 0)
                train_label = label[output_nodes['user'].long()].reshape(-1, 1)
                loss = criterion(logits, train_label)
                loss.backward()
                optimizer.step()
                Ht[output_nodes['user'].long()] = h
                train_loss += loss.item()
                auc, ks, acc = calc_auc(train_label, prob.detach()), calc_ks(train_label, prob.detach()), calc_acc(train_label, pred)
                pre, recall, f1 = calc_f1(train_label, pred)
                results.append([auc, ks, pre, recall, f1, acc])
            results = np.array(results)
            results = np.mean(results, axis=0)
            log.record("Epoch: %d, Time: %d, Train Loss: %.5f, AUC: %.4f, KS: %.4f, PRE: %.4f, RECALL: %.4f F1:%.4f, ACC: %.4f" % (
                epoch, t, train_loss, results[0], results[1], results[2], results[3], results[4], results[5])
            )

            
            model.eval()
            test_loss = 0.
            results = []
            with torch.no_grad():
                for input_nodes, output_nodes, blocks in tqdm(dataloader_test):
                    batch_size = output_nodes['user'].shape[0]
                    user_feature = blocks[0].nodes['user'].data['feature']
                    node_features = {'user': user_feature}
                    current = model.rgcn(blocks, node_features)
                    short = torch.zeros((args.w, batch_size, args.hid_dim))
                    for j in range(args.w):
                        short[j] = H_list[-args.w + j][output_nodes['user'].long()]
                    # short = torch.mean(short, dim=0)
                    short = model.cab(short)
                    h = model.gru(short, current)
                    logits = model.score(h)
                    prob = logits
                    pred = torch.where(prob > 0.5, 1, 0)
                    test_label = label[output_nodes['user'].long()].reshape(-1, 1)
                    loss = criterion(logits, test_label)
                    Ht[output_nodes['user'].long()] = h
                    test_loss += loss.item()
                    auc, ks, acc = calc_auc(test_label, prob.detach()), calc_ks(test_label, prob.detach()), calc_acc(test_label, pred)
                    pre, recall, f1 = calc_f1(test_label, pred)
                    results.append([auc, ks, pre, recall, f1, acc])
                results = np.array(results)
                results = np.mean(results, axis=0)
                log.record("Epoch: %d, Time: %d, Test Loss: %.5f, AUC: %.4f, KS: %.4f, PRE: %.4f, RECALL: %.4f F1:%.4f, ACC: %.4f" % (
                    epoch, t, test_loss, results[0], results[1], results[2], results[3], results[4], results[5])
                )

                if t == timespan and results[0] > bestauc:
                    bestepoch = epoch
                    bestauc = results[0]
                    torch.save(model.state_dict(), '/home/zqxu/MHTGNN/model_save/AddGraph_H_Xinye_params.pth')

                H_list = torch.cat([H_list, Ht.unsqueeze(0)], dim=0)

    log.record("Best Epoch[%d] Best AUC Score[%.4f]" % (bestepoch, bestauc))

if __name__ == '__main__':
    run()