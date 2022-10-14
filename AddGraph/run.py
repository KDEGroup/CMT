# coding=utf-8
import os, argparse
import sys

sys.path.append("XXX")

from Utils import set_seed, Logging
from Metrics import *
from Input import loadXinyeDataHomoDynamic
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

import warnings
warnings.filterwarnings("ignore")

log_dir = "XXX"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir, 'AddGraph_homo.log')
log = Logging(log_path)


parser = argparse.ArgumentParser(description='Addgraph')
parser.add_argument("--hid_dim", type=int, default=64, help="hidden layer dimension")
parser.add_argument("--batch_size", type=int, default=1024*8, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for regularization")
parser.add_argument("--epochs", type=int, default=200, help="Epochs for training")
parser.add_argument("--w", type=int, default=3, help="Historical windows size")
parser.add_argument("--path", type=str, default="XXX", help="Dataset path")

args = parser.parse_args()
log.record(args)

class AddGraph(nn.Module):
    def __init__(self, in_feats, hid_feats):
        super(AddGraph, self).__init__()
        self.gcn = GCN(in_feats, hid_feats)
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
        self.dropout = nn.Dropout(p=0.2)
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
        P = self.dropout(P)
        R = torch.sigmoid(torch.matmul(current, self.u_r) + torch.matmul(short, self.w_r) + self.b_r)
        R = self.dropout(R)
        Htn = torch.tanh(torch.matmul(current, self.u_c) + R * torch.matmul(short, self.w_c))
        Htn = self.dropout(Htn)
        Ht = short * (1 - P) + P * Htn
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

class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats):
        super(GCN, self).__init__()
        self.embed = nn.Linear(in_feats, hid_feats)
        self.conv1 = dgl.nn.pytorch.conv.GraphConv(hid_feats, hid_feats)
        self.conv2 = dgl.nn.pytorch.conv.GraphConv(hid_feats, hid_feats)

    def forward(self, blocks, x):
        x = self.embed(x)
        x = F.relu(self.conv1(blocks[0], x))
        x = F.relu(self.conv2(blocks[1], x))
        return x

def preprocess():
    train_user_idx, test_user_idx, edges = loadXinyeDataHomoDynamic()
    
    with open("XXX/u_train_test_Xinye.pickle", "rb") as fp:
        X_train_p, X_train_n, X_test_p, X_test_n = pickle.load(fp)

    train_user_idx = X_train_p + X_train_n
    test_user_idx = X_test_p + X_test_n

    graph_data = np.load("XXX/phase1_gdata.npz")
    features = graph_data['x']
    features[features==-1] = np.nan
    col_mean = np.nanmean(features, axis=0)
    inds = np.where(np.isnan(features))
    features[inds] = np.take(col_mean, inds[1])

    num_nodes = features.shape[0]
    num_features = features.shape[1]
    print("num_nodes: %d" % num_nodes)
    timespan = 7

    cached_subgraph = []
    uid_sets = []
    train_user = []
    test_user = []
    for t in range(1, timespan+1):
        edge = edges[edges['ts'] <= t]
        uid_set = set(edge['src'].values.tolist() + edge['dst'].values.tolist())

        train_labels = torch.empty(num_nodes).fill_(-1)
        train_p = pd.Series(X_train_p)
        train_p = train_p[train_p.isin(uid_set)].tolist()
        train_n = pd.Series(X_train_n)
        train_n = train_n[train_n.isin(uid_set)].tolist()
        train_labels[train_p] = 1
        train_labels[train_n] = 0
        
        test_labels = torch.empty(num_nodes).fill_(-1)
        test_p = pd.Series(X_test_p)
        test_p = test_p[test_p.isin(uid_set)].tolist()
        test_n = pd.Series(X_test_n)
        test_n = test_n[test_n.isin(uid_set)].tolist()
        test_labels[test_p] = 1
        test_labels[test_n] = 0

        edge_index = edge.drop(columns=['ts']).to_numpy()
        g = dgl.graph((edge_index[:, 0], edge_index[:, 1]), num_nodes=features.shape[0])
        g = dgl.to_bidirected(g)
        g = dgl.add_self_loop(g)
        g.ndata['feature'] = torch.FloatTensor(features)
        g.ndata['training'] = train_labels
        g.ndata['testing'] = test_labels
        cached_subgraph.append(g)
        uid_sets.append(uid_set)

    with open('XXX/cached_subgraph.pickle', 'wb') as f:
        pickle.dump(cached_subgraph, f)
    with open('XXX/uid_sets.pickle', 'wb') as f:
        pickle.dump(uid_sets, f)

def run():
    set_seed(42)
    graph_data = np.load("XXX/phase1_gdata.npz")
    features = graph_data['x']
    y = graph_data['y']
    label = torch.FloatTensor(y)
    num_nodes = features.shape[0]
    num_features = features.shape[1]
    timespan = 7
    with open('XXX/cached_subgraph.pickle', 'rb') as f:
        cached_subgraph = pickle.load(f)
    with open('XXX/uid_sets.pickle', 'rb') as f:
        uid_sets = pickle.load(f)

    model = AddGraph(num_features, args.hid_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.BCELoss(reduction='sum')

    for epoch in range(1, args.epochs+1):
        H_list = torch.zeros(1, num_nodes, args.hid_dim)
        for k in range(args.w - 1):
            H_list = torch.cat([H_list, torch.zeros(num_nodes, args.hid_dim).unsqueeze(0)], dim=0)
        for t in range(1, timespan+1):
            snapshot = cached_subgraph[t-1]
            if t == 0:
                H_list[-1] = model.embed(torch.FloatTensor(features.to_numpy()))
            uid_set = uid_sets[t-1]
            History = H_list[-1]
            train_p, train_n = (snapshot.ndata['training'] == 1).nonzero(as_tuple=True)[0].tolist(), (snapshot.ndata['training'] == 0).nonzero(as_tuple=True)[0].tolist()
            test_p, test_n = (snapshot.ndata['testing'] == 1).nonzero(as_tuple=True)[0].tolist(), (snapshot.ndata['testing'] == 0).nonzero(as_tuple=True)[0].tolist()
            X_train = train_p + train_n
            X_test = test_p + test_n
            random.shuffle(X_train)
            random.shuffle(X_test)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
            dataloader_train = dgl.dataloading.NodeDataLoader(snapshot, X_train, sampler, batch_size=args.batch_size, shuffle=False, drop_last=False)
            dataloader_test = dgl.dataloading.NodeDataLoader(snapshot, X_test, sampler, batch_size=args.batch_size, shuffle=False, drop_last=False)
            dataloader_snapshot = dgl.dataloading.NodeDataLoader(snapshot, list(uid_set), sampler, batch_size=args.batch_size, shuffle=False, drop_last=False)

            model.train()
            optimizer.zero_grad()
            train_loss = 0.
            Current = torch.zeros(num_nodes, args.hid_dim)
            Short = torch.zeros((args.w, num_nodes, args.hid_dim))
            for j in range(args.w):
                Short[j] = H_list[-args.w + j]
            for input_nodes, output_nodes, blocks in tqdm(dataloader_train):
                input_features = blocks[0].srcdata['feature']
                # input_features = History[input_nodes.long()]
                h = model.gcn(blocks, input_features)
                Current[output_nodes.long()] = h
            # Short = torch.mean(Short, dim=0)
            Short = model.cab(Short)
            Ht = model.gru(Short, Current)
            h = Ht[X_train]
            logits = model.score(h)
            prob = logits
            pred = torch.where(prob > 0.5, 1, 0)
            train_label = label[X_train].reshape(-1, 1)
            loss = criterion(logits, train_label)
            loss.backward()
            optimizer.step()
            auc, ks, acc = calc_auc(train_label, prob.detach()), calc_ks(train_label, prob.detach()), calc_acc(train_label, pred)
            pre, recall, f1 = calc_f1(train_label, pred)
            train_loss = loss.item()
            log.record("Epoch: %d, Time: %d, Train Loss: %.2f, AUC: %.4f, KS: %.4f, PRE: %.4f, RECALL: %.4f F1:%.4f, ACC: %.4f" % (
                epoch, t, train_loss, auc, ks, pre, recall, f1, acc)
            )

            model.eval()
            test_loss = 0.
            with torch.no_grad():
                Current = torch.zeros(num_nodes, args.hid_dim)
                Short = torch.zeros((args.w, num_nodes, args.hid_dim))
                for j in range(args.w):
                    Short[j] = H_list[-args.w + j]
                for input_nodes, output_nodes, blocks in tqdm(dataloader_test):
                    input_features = blocks[0].srcdata['feature']
                    # input_features = History[input_nodes.long()]
                    h = model.gcn(blocks, input_features)
                    Current[output_nodes.long()] = h
                # Short = torch.mean(Short, dim=0)
                Short = model.cab(Short)
                Ht = model.gru(Short, Current)
                h = Ht[X_test]
                logits = model.score(h)
                prob = logits
                pred = torch.where(prob > 0.5, 1, 0)
                test_label = label[X_test].reshape(-1, 1)
                loss = criterion(logits, test_label)
                auc, ks, acc = calc_auc(test_label, prob.detach()), calc_ks(test_label, prob.detach()), calc_acc(test_label, pred)
                pre, recall, f1 = calc_f1(test_label, pred)
                test_loss = loss.item()
                log.record("Epoch: %d, Time: %d, Test Loss: %.2f, AUC: %.4f, KS: %.4f, PRE: %.4f, RECALL: %.4f F1:%.4f, ACC: %.4f" % (
                    epoch, t, test_loss, auc, ks, pre, recall, f1, acc)
                )

                H_list = torch.cat([H_list, Ht.unsqueeze(0)], dim=0)

if __name__ == '__main__':
    # preprocess()
    run()
    
    