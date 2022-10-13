# coding=utf-8

from Input import loadXinyeDataHeteroDynamic
from Metrics import *
from Utils import set_seed, Logging
import os, argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl, dgl.nn.pytorch.conv, dgl.nn.pytorch.hetero
import dgl.function as fn
import numpy as np
import random
import pandas as pd
from time import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

log_dir = "/home/zqxu/MHTGNN/log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir, 'pretrain.log')
log = Logging(log_path)

parser = argparse.ArgumentParser(description='IG_RGCN')
parser.add_argument("--hid_dim", type=int, default=32, help="Hidden layer dimension")
parser.add_argument("--batch_size", type=int, default=1024*4, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for regularization")
parser.add_argument("--epochs", type=int, default=20, help="Epochs for training")
parser.add_argument("--agg_fn", type=str, default='stack', help="Aggregation function for RGCN")
parser.add_argument("--agg_room", type=str, default='mean', help="Aggregation for room feature definition")
parser.add_argument("--model", type=str, default='RGCN', help="Model name RGCN/IG_RGCN")

args = parser.parse_args()
log.record(args)

class IG_RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, rel_names, agg_fn):
        super().__init__()
        
        self.embed = nn.Linear(in_feats, hid_feats)
        self.conv1 = dgl.nn.pytorch.hetero.HeteroGraphConv({
            rel: IGConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate=agg_fn)
        self.conv2 = dgl.nn.pytorch.hetero.HeteroGraphConv({
            rel: IGConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate=agg_fn)
        self.attn = SemanticAttention(hid_feats, hid_feats)

        self.predict = nn.Sequential(
            nn.Linear(hid_feats, 1),
            nn.Sigmoid()
        )

    def forward(self, blocks, x, encode=False):
        # inputs are features of nodes
        x['user'] = self.embed(x['user'])
        h = self.conv1(blocks[0], x)
        h = {k: F.relu(self.attn(v), inplace=True) for k, v in h.items()}
        h = self.attn(self.conv2(blocks[1], h)['user'])
        if encode:
            return h
        logits = self.predict(h)
        return logits

class IGConv(nn.Module):
    def __init__(self, in_feats, hid_feats):
        super(IGConv, self).__init__()
        self.fc1 = nn.Linear(in_feats, hid_feats)
        self.fc2 = nn.Linear(in_feats*3, hid_feats)
        self.fc3 = nn.Sequential(nn.Linear(in_feats*2, hid_feats), nn.ReLU())

    def forward(self, graph, feat):
        aggregate_fn = fn.copy_u('h', 'm')
        graph.srcdata['h'] = feat[0]
        graph.dstdata['h'] = feat[1]
        graph.update_all(aggregate_fn, fn.max('m', 'max'))
        graph.update_all(aggregate_fn, fn.mean('m', 'mean'))
        graph.update_all(aggregate_fn, fn.sum('m', 'sum'))
        graph.dstdata['h'] = self.fc3(F.relu(torch.cat(
            [self.fc2(torch.cat([graph.dstdata['max'], graph.dstdata['mean'], graph.dstdata['sum']], 1)), self.fc1(graph.dstdata['h'])],
            1)))
        return graph.dstdata['h']

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


if __name__ == '__main__':
    set_seed(42)
    hetero_graph, node_feature, n_hetero_features, train_user_idx, test_user_idx, edges, label = loadXinyeDataHeteroDynamic()
    model = IG_RGCN(n_hetero_features, args.hid_dim, hetero_graph.etypes, args.agg_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss(reduction='sum')

    timespan = 7
    num_user_nodes = hetero_graph.num_nodes('user')
    embed_table = torch.zeros(timespan-1, num_user_nodes, args.hid_dim)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

    train_user = []
    test_user = []
    for t in range(1, timespan):
        edge = edges[edges['ts'] <= t]
        uid_set = set(edge['src'].values.tolist() + edge['dst'].values.tolist())
        train_user_idx_t = pd.Series(train_user_idx)
        train_user_idx_t = train_user_idx_t[train_user_idx_t.isin(uid_set)].tolist()
        if train_user:
            prev_train_user = set(train_user[-1])
            train_user.append(train_user_idx_t)
            cur_train_user = set(train_user_idx_t)
            train_user_idx_t = list(cur_train_user - prev_train_user)
        else:
            train_user.append(train_user_idx_t)
        train_user_idx_t = np.array(train_user_idx_t)
        test_user_idx_t = pd.Series(test_user_idx)
        test_user_idx_t = test_user_idx_t[test_user_idx_t.isin(uid_set)].tolist()
        if test_user:
            prev_test_user = set(test_user[-1])
            test_user.append(test_user_idx_t)
            cur_test_user = set(test_user_idx_t)
            test_user_idx_t = list(cur_test_user - prev_test_user)
        else:
            test_user.append(test_user_idx_t)
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

        test_uid = set(test_user_idx_t)
        edge_train = edge[~edge['src'].isin(test_uid) & ~edge['dst'].isin(test_uid)]
        edges_one = edge_train[edge_train['type'] == 1].drop(columns=['type']).to_numpy()
        edges_two = edge_train[edge_train['type'] == 2].drop(columns=['type']).to_numpy()
        edges_three = edge_train[edge_train['type'] == 3].drop(columns=['type']).to_numpy()
        one_src, one_dst = torch.tensor(edges_one[:, 0]), torch.tensor(edges_one[:, 1])
        two_src, two_dst = torch.tensor(edges_two[:, 0]), torch.tensor(edges_two[:, 1])
        three_src, three_dst = torch.tensor(edges_three[:, 0]), torch.tensor(edges_three[:, 1])

        hg_train = dgl.heterograph(
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
        hg_train.nodes['user'].data['feature'] = torch.FloatTensor(node_feature)
        hg = [hg_train, hg]

        dataloader_train = dgl.dataloading.NodeDataLoader(hg[0], {'user': train_user_idx_t}, sampler, batch_size=args.batch_size, shuffle=True, drop_last=True)
        dataloader_test = dgl.dataloading.NodeDataLoader(hg[1], {'user': test_user_idx_t}, sampler, batch_size=args.batch_size, shuffle=False, drop_last=True)
        dataloader_snapshot = dgl.dataloading.NodeDataLoader(hg[1], {'user': list(uid_set)}, sampler, batch_size=args.batch_size, shuffle=False, drop_last=False)

        if t <= 4:
            max_epochs = args.epochs
        else:
            max_epochs = 10

        for epoch in range(1, max_epochs+1):
            model.train()
            train_loss = 0.
            results = []
            t0 = time()
            for input_nodes, output_nodes, blocks in tqdm(dataloader_train):
                optimizer.zero_grad()
                user_feature = blocks[0].nodes['user'].data['feature']
                node_features = {'user': user_feature}
                logits = model(blocks, node_features)
                prob = logits
                pred = torch.where(prob > 0.5, 1, 0)
                train_label = label[output_nodes['user'].long()].reshape(-1, 1)
                loss = criterion(prob, train_label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                auc, ks, acc = calc_auc(train_label, prob.detach()), calc_ks(train_label, prob.detach()), calc_acc(train_label, pred)
                pre, recall, f1 = calc_f1(train_label, pred)
                results.append([auc, ks, pre, recall, f1, acc])
            results = np.array(results)
            results = np.mean(results, axis=0)
            log.record("Snapshot: %d, Epoch: %d, Train Loss: %.5f, AUC: %.4f, KS: %.4f, PRE: %.4f, RECALL: %.4f F1:%.4f, ACC: %.4f" % (
                t, epoch, train_loss, results[0], results[1], results[2], results[3], results[4], results[5])
            )

            model.eval()
            test_loss = 0.
            results = []
            with torch.no_grad():
                for input_nodes, output_nodes, blocks in tqdm(dataloader_test):
                    user_feature = blocks[0].nodes['user'].data['feature']
                    node_features = {'user': user_feature}
                    logits = model(blocks, node_features)
                    prob = logits
                    pred = torch.where(prob > 0.5, 1, 0)
                    test_label = label[output_nodes['user'].long()].reshape(-1, 1)
                    loss = criterion(prob, test_label)
                    test_loss += loss.item()
                    auc, ks, acc = calc_auc(test_label, prob.detach()), calc_ks(test_label, prob.detach()), calc_acc(test_label, pred)
                    pre, recall, f1 = calc_f1(test_label, pred)
                    results.append([auc, ks, pre, recall, f1, acc])
                results = np.array(results)
                results = np.mean(results, axis=0)

                if epoch == max_epochs:
                    for input_nodes, output_nodes, blocks in tqdm(dataloader_snapshot):
                        user_feature = blocks[0].nodes['user'].data['feature']
                        node_features = {'user': user_feature}
                        h = model(blocks, node_features, encode=True)
                        embed_table[t-1][output_nodes['user'].long()] = h

                t1 = time()
                log.record("Snapshot: %d, Epoch: %d, Test Loss: %.5f, AUC: %.4f, KS: %.4f, PRE: %.4f, RECALL: %.4f F1:%.4f, ACC: %.4f, Time: %.1f" % (
                    t, epoch, test_loss, results[0], results[1], results[2], results[3], results[4], results[5], t1-t0)
                )

    torch.save(model.state_dict(), '/home/zqxu/MHTGNN/model_save/IHG_Xinye_pretrain_params.pth')
    torch.save(embed_table, "/home/zqxu/MHTGNN/data/embedding/IHG_Xinye_pretrain_embed.pt")
