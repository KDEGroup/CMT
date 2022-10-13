# coding=utf-8

from Input import loadXinyeDataHetero
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
from time import time
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

log_dir = "/home/zqxu/MHTGNN/log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir, 'RGCN.log')
log = Logging(log_path)

parser = argparse.ArgumentParser(description='RGCN')
parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
parser.add_argument("--batch_size", type=int, default=1024*8, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for regularization")
parser.add_argument("--epochs", type=int, default=100, help="Epochs for training")
parser.add_argument("--agg_fn", type=str, default='mean', help="Aggregation function for RGCN")
parser.add_argument("--agg_room", type=str, default='mean', help="Aggregation for room feature definition")
parser.add_argument("--model", type=str, default='RGCN', help="Model name RGCN/IG_RGCN")

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
        h = self.predict(h)
        return h


if __name__ == '__main__':
    set_seed(42)
    hetero_graph, n_hetero_features, train_user_idx, test_user_idx, label = loadXinyeDataHetero()
    model = RGCN(n_hetero_features, args.hid_dim, hetero_graph.etypes, args.agg_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss(reduction='sum')

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader_train = dgl.dataloading.NodeDataLoader(hetero_graph, {'user': train_user_idx}, sampler, batch_size=args.batch_size, shuffle=True, drop_last=False)
    dataloader_test = dgl.dataloading.NodeDataLoader(hetero_graph, {'user': test_user_idx}, sampler, batch_size=args.batch_size, shuffle=False, drop_last=False)

    bestepoch = 0
    bestauc = 0.0
    for epoch in range(args.epochs):
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
        log.record("Epoch: %d, Train Loss: %.5f, AUC: %.4f, KS: %.4f, PRE: %.4f, RECALL: %.4f F1:%.4f, ACC: %.4f" % (
            epoch, train_loss, results[0], results[1], results[2], results[3], results[4], results[5])
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
            t1 = time()
            log.record("Epoch: %d, Test Loss: %.5f, AUC: %.4f, KS: %.4f, PRE: %.4f, RECALL: %.4f F1:%.4f, ACC: %.4f, Time: %.1f" % (
                epoch, test_loss, results[0], results[1], results[2], results[3], results[4], results[5], t1-t0)
            )

        if results[0] > bestauc:
            bestepoch = epoch
            bestauc = results[0]
            torch.save(model.state_dict(), '/home/zqxu/MHTGNN/model_save/RGCN_Xinye_params.pth')

    log.record("Best Epoch[%d] Best AUC Score[%.4f]" % (bestepoch, bestauc))
