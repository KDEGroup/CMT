# coding=utf-8
import os
import sys
import argparse

from Input import loadXinyeDataHomo
from Utils import set_seed, Logging
from Metrics import *
from tqdm import tqdm
from time import time
import numpy as np
import dgl, dgl.nn.pytorch.conv
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='GAT')
parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
parser.add_argument("--num_heads", type=int, default=3, help="Number of heads")
parser.add_argument("--batch_size", type=int, default=1024*8, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for regularization")
parser.add_argument("--epochs", type=int, default=100, help="Epochs for training")

log_dir = "/home/zqxu/MHTGNN/log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir, 'GAT.log')
log = Logging(log_path)

args = parser.parse_args()
log.record(args)

class GAT(nn.Module):
    def __init__(self, in_feats, hid_feats, num_heads):
        super().__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.num_heads = num_heads
        self.conv1 = dgl.nn.pytorch.conv.GATConv(hid_feats, hid_feats, num_heads)
        self.conv2 = dgl.nn.pytorch.conv.GATConv(num_heads*hid_feats, hid_feats, num_heads)

        self.embed = nn.Linear(in_feats, hid_feats)

        self.predict = nn.Sequential(
            nn.Linear(num_heads*hid_feats, hid_feats),
            nn.Linear(hid_feats, 1),
            nn.Sigmoid()
        )

    def forward(self, blocks, x):
        x = self.embed(x)
        h = F.relu(self.conv1(blocks[0], x))
        h = h.reshape(-1, self.num_heads*self.hid_feats)
        h = F.relu(self.conv2(blocks[1], h))
        h = h.reshape(-1, self.num_heads*self.hid_feats)
        h = self.predict(h)
        return h

if __name__ == "__main__":
    set_seed(42)
    graph, n_homo_features, train_user_idx, test_user_idx, label = loadXinyeDataHomo()

    model = GAT(n_homo_features, args.hid_dim, args.num_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss(reduction='sum')
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader_train = dgl.dataloading.NodeDataLoader(graph[0], train_user_idx, sampler, batch_size=args.batch_size, shuffle=True, drop_last=False)
    dataloader_test = dgl.dataloading.NodeDataLoader(graph[1], test_user_idx, sampler, batch_size=args.batch_size, shuffle=False, drop_last=False)

    bestepoch = 0
    bestauc = 0.0
    for epoch in range(args.epochs):
        t0 = time()
        model.train()
        train_loss = 0.
        results = []
        for input_nodes, output_nodes, blocks in tqdm(dataloader_train):
            optimizer.zero_grad()
            input_features = blocks[0].srcdata['feature']
            train_label = label[output_nodes.long()].reshape(-1, 1)
            logits = model(blocks, input_features)
            prob = logits
            pred = torch.where(prob > 0.5, 1, 0)
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
                input_features = blocks[0].srcdata['feature']
                test_label = label[output_nodes.long()].reshape(-1, 1)
                logits = model(blocks, input_features)
                prob = logits
                pred = torch.where(prob > 0.5, 1, 0)
                loss = criterion(logits, test_label)
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
            torch.save(model.state_dict(), '/home/zqxu/MHTGNN/model_save/GAT_Xinye_params.pth')

    log.record("Best Epoch[%d] Best AUC Score[%.4f]" % (bestepoch, bestauc))
