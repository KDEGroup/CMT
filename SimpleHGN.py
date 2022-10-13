import os, sys, argparse
from tkinter.ttk import Label
sys.path.append("/home/zqxu/MHTGNN/code/")

import dgl
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import dgl.function as fn
import torch.nn.functional as F
from dgl.ops import edge_softmax
from dgl.nn import TypedLinear
from time import time
from tqdm import tqdm
from Utils import set_seed, Logging
from Input import loadXinyeDataHetero
from Metrics import *

log_dir = "/home/zqxu/MHTGNN/log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir, 'SimpleHGN.log')
log = Logging(log_path)

parser = argparse.ArgumentParser(description='SimpleHGN')
parser.add_argument("--h_dim", type=int, default=64, help="hidden dimension")
parser.add_argument("--edge_dim", type=int, default=64, help="Edge feature dimension")
parser.add_argument("--batch_size", type=int, default=1024*8, help="Batch size")
parser.add_argument("--n_layers", type=int, default=3, help="number of hidden layers")
parser.add_argument("--num_heads", type=int, default=3, help="number of heads in multi-head attention")
parser.add_argument("--num_classes", type=int, default=2, help="number of classes")
parser.add_argument("--k", type=int, default=1, help="number of hops")
parser.add_argument("--feats_drop_rate", type=float, default=0.2, help="dropout rate")
parser.add_argument("--slope", type=float, default=0.05, help="slope of the leaky relu")
parser.add_argument("--beta", type=float, default=0.05, help="beta of the leaky relu")
parser.add_argument("--residual", type=bool, default=True, help="residual connection")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for regularization")
parser.add_argument("--epochs", type=int, default=100, help="Epochs for training")
parser.add_argument("--agg_room", type=str, default='mean', help="Aggregation for room feature definition")
parser.add_argument("--path", type=str, default="/home/zqxu/MHTGNN/data/", help="Dataset path")

args = parser.parse_args()
log.record(args)

class SimpleHGN(nn.Module):
    def __init__(self, edge_dim, num_etypes, in_dim, hidden_dim, num_classes,
                num_layers, heads, feat_drop, negative_slope, residual, beta):
        super().__init__()
        self.num_layers = num_layers
        self.hgn_layers = nn.ModuleList()
        self.activation = F.elu

        self.hgn_layers.append(
            SimpleHGNConv(
                edge_dim,
                in_dim,
                hidden_dim,
                heads[0],
                num_etypes,
                feat_drop,
                negative_slope,
                False,
                self.activation,
                beta=beta,
            )
        )

        for l in range(1, num_layers - 1): 
            self.hgn_layers.append(
                SimpleHGNConv(
                    edge_dim,
                    hidden_dim * heads[l - 1],
                    hidden_dim,
                    heads[l],
                    num_etypes,
                    feat_drop,
                    negative_slope,
                    residual,
                    self.activation,
                    beta=beta,
                )
            )

        self.hgn_layers.append(
            SimpleHGNConv(
                edge_dim,
                hidden_dim * heads[-2],
                num_classes,
                heads[-1],
                num_etypes,
                feat_drop,
                negative_slope,
                residual,
                None,
                beta=beta,
            )
        )

    def forward(self, hg, user_feature):
        h_dict = user_feature
        with hg.local_scope():
            hg.ndata['h'] = h_dict
            g = dgl.to_homogeneous(hg, ndata = 'h')
            h = g.ndata['h']
            for l in range(self.num_layers):
                h = self.hgn_layers[l](g, h, g.edata['_TYPE'])
                h = h.flatten(1)

        h_dict = to_hetero_feat(h, g.ndata['_TYPE'], hg.ntypes)

        return h_dict

class SimpleHGNConv(nn.Module):
    def __init__(self, edge_dim, in_dim, out_dim, num_heads, num_etypes, feat_drop=0.0, 
                negative_slope=0.2, residual=True, activation=F.elu, beta=0.0):
        super().__init__()
        self.edge_dim = edge_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_etypes = num_etypes

        self.edge_emb = nn.Parameter(torch.empty(size=(num_etypes, edge_dim)))
        self.W = nn.Parameter(torch.FloatTensor(in_dim, out_dim * num_heads))
        self.W_r = TypedLinear(edge_dim, edge_dim * num_heads, num_etypes)

        self.a_l = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.a_r = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.a_e = nn.Parameter(torch.empty(size=(1, num_heads, edge_dim)))

        nn.init.xavier_uniform_(self.edge_emb, gain=1.414)
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a_l.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_r.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_e.data, gain=1.414)

        self.feat_drop = nn.Dropout(feat_drop)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.activation = activation

        if residual:
            self.residual = nn.Linear(in_dim, out_dim * num_heads)
        else:
            self.register_buffer("residual", None)

        self.beta = beta

    def forward(self, g, h, etype):
        emb = self.feat_drop(h)
        emb = torch.matmul(emb, self.W).view(-1, self.num_heads, self.out_dim)
        emb[torch.isnan(emb)] = 0.0

        edge_emb = self.W_r(self.edge_emb[etype], etype).view(-1, self.num_heads, self.edge_dim)

        row = g.edges()[0]
        col = g.edges()[1]

        h_l = (self.a_l * emb).sum(dim=-1)[row]
        h_r = (self.a_r * emb).sum(dim=-1)[col]
        h_e = (self.a_e * edge_emb).sum(dim=-1)

        edge_attention = self.leakyrelu(h_l + h_r + h_e)
        edge_attention = edge_softmax(g, edge_attention)

        if 'alpha' in g.edata.keys():
            res_attn = g.edata['alpha']
            edge_attention = edge_attention * (1 - self.beta) + res_attn * self.beta
        if self.num_heads == 1:
            edge_attention = edge_attention[:, 0]
            edge_attention = edge_attention.unsqueeze(1)

        with g.local_scope():
            emb = emb.permute(0, 2, 1).contiguous()
            g.edata['alpha'] = edge_attention
            g.srcdata['emb'] = emb
            g.update_all(fn.u_mul_e('emb', 'alpha', 'm'),
                            fn.sum('m', 'emb'))
            h_output = g.ndata['emb'].view(-1, self.out_dim * self.num_heads)

        g.edata['alpha'] = edge_attention
        if self.residual:
            res = self.residual(h)
            h_output += res
        if self.activation is not None:
            h_output = self.activation(h_output)

        return h_output


def to_hetero_feat(h, type, name):
    """Feature convert API.
    
    It uses information about the type of the specified node
    to convert features ``h`` in homogeneous graph into a heteorgeneous
    feature dictionay ``h_dict``.
    
    Parameters
    ----------
    h: Tensor
        Input features of homogeneous graph
    type: Tensor
        Represent the type of each node or edge with a number.
        It should correspond to the parameter ``name``.
    name: list
        The node or edge types list.
    
    Return
    ------
    h_dict: dict
        output feature dictionary of heterogeneous graph
    
    Example
    -------
    
    >>> h = torch.tensor([[1, 2, 3],
                          [1, 1, 1],
                          [0, 2, 1],
                          [1, 3, 3],
                          [2, 1, 1]])
    >>> print(h.shape)
    torch.Size([5, 3])
    >>> type = torch.tensor([0, 1, 0, 0, 1])
    >>> name = ['author', 'paper']
    >>> h_dict = to_hetero_feat(h, type, name)
    >>> print(h_dict)
    {'author': tensor([[1, 2, 3],
    [0, 2, 1],
    [1, 3, 3]]), 'paper': tensor([[1, 1, 1],
    [2, 1, 1]])}
    
    """
    h_dict = {}
    sigm = nn.Sigmoid()
    for index, ntype in enumerate(name):
        h_dict[ntype] = sigm(h[torch.where(type == index)])

    return h_dict

def run():
    set_seed(42)
    hetero_graph, n_hetero_features, train_user_idx, test_user_idx, label = loadXinyeDataHetero()
    heads = [args.num_heads] * args.n_layers + [1]
    hidden_dim = args.h_dim*args.num_heads
    model = SimpleHGN(args.edge_dim, len(hetero_graph[1].etypes), n_hetero_features, args.h_dim, args.num_classes, 
                    args.n_layers, heads, args.feats_drop_rate, args.slope, args.residual, args.beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss(reduction='sum')
    fanouts = dict()
    for etype in hetero_graph[1].etypes:
        fanouts[etype] = -1
    sampler = dgl.dataloading.ShaDowKHopSampler([fanouts])
    # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader_train = dgl.dataloading.NodeDataLoader(hetero_graph[0], {'user': train_user_idx}, sampler, batch_size=args.batch_size, shuffle=True, drop_last=False)
    dataloader_test = dgl.dataloading.NodeDataLoader(hetero_graph[1], {'user': test_user_idx}, sampler, batch_size=args.batch_size, shuffle=False, drop_last=False)

    bestepoch = 0
    bestauc = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        t0 = time()
        train_loss = 0.
        results = []
        for input_nodes, output_nodes, hg in tqdm(dataloader_train):
            optimizer.zero_grad()
            user_feature = hg.nodes['user'].data['feature']
            batch_size = output_nodes['user'].shape[0]

            logits = model(hg, user_feature)['user'][:batch_size]
            logits = torch.max(logits, dim=1).values.reshape(-1, 1)
            prob = logits
            pred = torch.where(prob > 0.5, 1, 0)
            train_label = label[output_nodes['user'].long()].reshape(-1, 1)
            loss = criterion(logits, train_label)
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
            for input_nodes, output_nodes, hg in tqdm(dataloader_test):
                user_feature = hg.nodes['user'].data['feature']
                batch_size = output_nodes['user'].shape[0]
                
                logits = model(hg, user_feature)['user'][:batch_size]
                logits = torch.max(logits, dim=1).values.reshape(-1, 1)
                prob = logits
                pred = torch.where(prob > 0.5, 1, 0)
                test_label = label[output_nodes['user'].long()].reshape(-1, 1)
                loss = criterion(logits, test_label)
                optimizer.step()
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
            torch.save(model.state_dict(), '/home/zqxu/MHTGNN/model_save/SimpleHGN_Xinye_params.pth')

    log.record("Best Epoch[%d] Best AUC Score[%.4f]" % (bestepoch, bestauc))

if __name__ == '__main__':
    run()
    