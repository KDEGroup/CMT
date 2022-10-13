# coding=utf-8
import os, argparse
import sys

sys.path.append("/home/zqxu/MHTGNN/code")

from Utils import *
from Metrics import *
from Input import loadXinyeDataHetero
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl, dgl.nn.pytorch.conv, dgl.nn.pytorch.hetero
import dgl.function as fn
import numpy as np
from tqdm import tqdm
from time import time
import torch.optim

import warnings
warnings.filterwarnings("ignore")

log_dir = "/home/zqxu/MHTGNN/log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir, 'concat_tss_cl_rs.log')
log = Logging(log_path)

parser = argparse.ArgumentParser(description='RGCN')
parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
parser.add_argument("--batch_size", type=int, default=1024*8, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for regularization")
parser.add_argument("--epochs", type=int, default=50, help="Epochs for training")
parser.add_argument("--agg_fn", type=str, default='stack', help="Aggregation function for RGCN")
parser.add_argument("--agg_room", type=str, default='mean', help="Aggregation for room feature definition")
parser.add_argument("--model", type=str, default='IG_RGCN', help="Model name RGCN/IG_RGCN")
parser.add_argument("--path", type=str, default="/home/zqxu/MHTGNN/data/", help="Dataset path")

args = parser.parse_args()
log.record(args)

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

class IG_RGCN(nn.Module):
    def __init__(self, tss_embed, rs_embed, in_feats, hid_feats, rel_names, agg_fn):
        super().__init__()
        
        self.tss_embed = tss_embed
        tss_embed_size = tss_embed.shape[-1]
        self.rs_embed = rs_embed
        rs_embed_size = rs_embed.shape[-1]
        self.embed = nn.Linear(in_feats, hid_feats)
        self.conv1 = dgl.nn.pytorch.hetero.HeteroGraphConv({
            rel: IGConv(hid_feats+tss_embed_size+rs_embed_size, hid_feats)
            for rel in rel_names}, aggregate=agg_fn)
        self.conv2 = dgl.nn.pytorch.hetero.HeteroGraphConv({
            rel: IGConv(hid_feats+tss_embed_size+rs_embed_size, hid_feats)
            for rel in rel_names}, aggregate=agg_fn)
        self.attn = SemanticAttention(hid_feats, hid_feats)

        self.predict = nn.Sequential(
            nn.Linear(hid_feats, 1),
            nn.Sigmoid()
        )

    def forward(self, blocks, x):
        # inputs are features of nodes
        user_original_feature = self.embed(x['user'])
        user_tss_feature = self.tss_embed[blocks[0].srcdata[dgl.NID]]
        user_rs_feature = self.rs_embed[blocks[0].srcdata[dgl.NID]]
        x['user'] = torch.cat(
            [
                user_original_feature, 
                user_tss_feature,
                user_rs_feature
            ], dim=1)
        h = self.conv1(blocks[0], x)
        h = {k: F.relu(self.attn(v), inplace=True) for k, v in h.items()}

        user_tss_feature = self.tss_embed[blocks[1].srcdata[dgl.NID]]
        user_rs_feature = self.rs_embed[blocks[1].srcdata[dgl.NID]]
        user_original_feature = h['user']
        h['user'] = torch.cat(
            [
                user_original_feature, 
                user_tss_feature,
                user_rs_feature
            ], dim=1)      
        h = self.attn(self.conv2(blocks[1], h)['user'])
        h = self.predict(h)

        return h

class IGConv(nn.Module):
    def __init__(self, in_feats, hid_feats):
        super(IGConv, self).__init__()
        self.fc1 = nn.Linear(in_feats, hid_feats)
        self.fc2 = nn.Linear(in_feats*3, hid_feats)
        self.fc3 = nn.Linear(hid_feats*2, hid_feats)

    def forward(self, graph, feat):
        aggregate_fn = fn.copy_u('h', 'm')
        graph.srcdata['h'] = feat[0]
        graph.dstdata['h'] = feat[1]
        graph.update_all(aggregate_fn, fn.max('m', 'max'))
        graph.update_all(aggregate_fn, fn.mean('m', 'mean'))
        graph.update_all(aggregate_fn, fn.sum('m', 'sum'))
        max_mean_sum_msg = torch.cat([graph.dstdata['max'], graph.dstdata['mean'], graph.dstdata['sum']], dim=1)
        input_msg = torch.cat([self.fc2(max_mean_sum_msg), self.fc1(graph.dstdata['h'])], dim=1)
        input_msg = F.relu(input_msg, inplace=True)
        graph.dstdata['h'] = self.fc3(input_msg)
        return graph.dstdata['h']


def run():
    set_seed(42)
    hetero_graph, n_hetero_features, train_user_idx, test_user_idx, label = loadXinyeDataHetero()
    tss_embed = torch.load("/home/zqxu/MHTGNN/data/embedding/tfencoder_Xinye_tss_Npair_embed.pt")
    rs_embed = torch.load("/home/zqxu/MHTGNN/data/embedding/tfencoder_Xinye_rs_embed.pt")
    model = IG_RGCN(tss_embed, rs_embed, n_hetero_features, args.hid_dim, hetero_graph[1].etypes, args.agg_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss(reduction='sum')

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader_train = dgl.dataloading.NodeDataLoader(hetero_graph[0], {'user': train_user_idx}, sampler, batch_size=args.batch_size, shuffle=True, drop_last=False)
    dataloader_test = dgl.dataloading.NodeDataLoader(hetero_graph[1], {'user': test_user_idx}, sampler, batch_size=args.batch_size, shuffle=False, drop_last=False)
    bestauc = 0.0
    bestepoch = 0

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
                bestauc = results[0]
                bestepoch = epoch
                torch.save(model.state_dict(), "/home/zqxu/MHTGNN/model_save/IHG_tss_cl_rs_%.4f_prarams.pth" % bestauc)

    log.record("Best Epoch[%d] Best AUC Score[%.4f]" % (bestepoch, bestauc))

if __name__ == '__main__':
    run()
    