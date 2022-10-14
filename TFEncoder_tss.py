# coding=utf-8
import os, sys, argparse

from time import time
import torch
import torch.nn as nn
import torch.optim
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import random
from tqdm import tqdm
from Utils import Logging, set_seed
from Metrics import *

import warnings
warnings.filterwarnings("ignore")

log_dir = "XXX"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir, 'TFEncoder_tss.log')
log = Logging(log_path)

parser = argparse.ArgumentParser(description='tfencoder_tss')
parser.add_argument("--hid_dim", type=int, default=32, help="Hidden layer dimension")
parser.add_argument("--n_layers", type=int, default=2, help="Number of layers")
parser.add_argument("--batch_size", type=int, default=1024*4)
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for regularization")
parser.add_argument("--epochs", type=int, default=25, help="Epochs for training")
parser.add_argument("--path", type=str, default="XXX", help="Dataset path")
parser.add_argument("--data", type=str, default="XXX", help="Dataset")

args = parser.parse_args()
log.record(args)

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class SnapshotDataset(Dataset):
    def __init__(self, is_train_set=True):
        # shape: [timespan, num_nodes, feat_dim] -> [num_nodes, timespan, feat_dim]
        embed_table = torch.load(args.data)
        embed_table = torch.transpose(embed_table, dim0=0, dim1=1)
        with open("/home/zqxu/MHTGNN/data/u_train_test_Xinye.pickle", "rb") as fp:
            X_train_p, X_train_n, X_test_p, X_test_n = pickle.load(fp)
        X_train = X_train_p + X_train_n
        X_test = X_test_p + X_test_n
        self.num_nodes = embed_table.shape[0]
        self.time_span = embed_table.shape[1]
        assert self.time_span == 6
        self.feat_dim = embed_table.shape[2]
        if is_train_set:
            self.len = len(X_train)
            self.label = torch.zeros(max(X_train) + 1, dtype=torch.int64)
            self.label[torch.LongTensor(X_train_p)] = 1
            self.label = self.label[X_train].reshape(-1, 1).float()
            self.embed_table = embed_table[X_train]
        else:
            self.len = len(X_test)
            self.label = torch.zeros(max(X_test) + 1, dtype=torch.int64)
            self.label[torch.LongTensor(X_test_p)] = 1
            self.label = self.label[X_test].reshape(-1, 1).float()
            self.embed_table = embed_table[X_test]
        idx = torch.randperm(self.embed_table.shape[0])
        self.embed_table = self.embed_table[idx]
        self.label = self.label[idx]


    def __getitem__(self, index):
        return self.embed_table[index], self.label[index]

    def __len__(self):
        return self.len

class PretrainDataset(Dataset):
    def __init__(self):
        embed_table = torch.load(args.data)
        self.embed_table = torch.transpose(embed_table, dim0=0, dim1=1)
        self.num_nodes = self.embed_table.shape[0]
        self.time_span = self.embed_table.shape[1]
        assert self.time_span == 6

    def __getitem__(self, index):
        return index, self.embed_table[index]

    def __len__(self):
        return self.num_nodes

class TFEncoderClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, time_span, batch_size, n_layers=1, bidirectional=False):
        super(TFEncoderClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.time_span = time_span
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.n_directions = 2 if bidirectional else 1
        self.predict = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        self.position_encoder = nn.Embedding(time_span, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(p=0.5)
        self.embedding = nn.Linear(input_size, hidden_size)
        transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=512)
        if n_layers > 1:
            self.transformer_encoder = nn.TransformerEncoder(transformer_encoder, num_layers=n_layers)
        else:
            self.transformer_encoder = transformer_encoder
        
    def add_position_embedding(self, input_seq):
        # input_seq: [batch_size, time_span, hidden_size]
        batch_size = input_seq.shape[0]
        time_span = input_seq.shape[1]
        position_ids = np.array(batch_size*list(range(time_span))).reshape(batch_size, time_span)
        position_ids = torch.LongTensor(position_ids)
        item_embeddings = self.embedding(input_seq)
        position_embeddings = self.position_encoder(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def forward(self, input, encode=False):
        # input shape: [batch_size, time_span(seq_len), hidden_size] -> [time_span(seq_len), batch_size, hidden_size]
        input = self.add_position_embedding(input)
        # input = self.embedding(input)
        input = torch.transpose(input, dim0=0, dim1=1)
        # 加入self-attention
        diag_val = torch.ones(input.shape[0], input.shape[0])
        mask = torch.triu(diag_val)
        # attn_output, attn_output_weights = self.attn(query=input, key=input, value=input, attn_mask=mask)
        # output, (hidden, _) = self.lstm(attn_output, (self.hidden, self.cell))
        if self.n_layers > 1:
            tf_output = self.transformer_encoder(src=input, mask=mask)
        else:
            tf_output = self.transformer_encoder(src=input, src_mask=mask)
        hidden_cat = tf_output[-1]
        if encode:
            return hidden_cat
        logits = self.predict(hidden_cat)
        return logits

def run():
    set_seed(42)
    trainset = SnapshotDataset(is_train_set=True)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    testset = SnapshotDataset(is_train_set=False)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = TFEncoderClassifier(trainset.feat_dim, args.hid_dim, trainset.time_span, args.batch_size, args.n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss(reduction='sum')

    bestauc = 0.
    bestepoch = 0
    totaltime = 0.
    for epoch in range(args.epochs):
        t0 = time()
        train_loss = 0.
        model.train()
        results = []
        for _, (embeds, labels) in enumerate(tqdm(trainloader)):
            optimizer.zero_grad()
            logits = model(embeds)
            prob = logits
            pred = torch.where(prob > 0.5, 1, 0)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            auc, ks, acc = calc_auc(labels, prob.detach()), calc_ks(labels, prob.detach()), calc_acc(labels, pred)
            pre, recall, f1 = calc_f1(labels, pred)
            results.append([auc, ks, pre, recall, f1, acc])
        results = np.array(results)
        results = np.mean(results, axis=0)
        t1 = time()
        log.record("Epoch: %d, Train Loss: %.2f, AUC: %.4f, KS: %.4f, PRE: %.4f, RECALL: %.4f F1:%.4f, ACC: %.4f" % (
            epoch, train_loss, results[0], results[1], results[2], results[3], results[4], results[5])
        )

        test_loss = 0.
        model.eval()
        results = []
        with torch.no_grad():
            for _, (embeds, labels) in enumerate(tqdm(testloader)):
                logits = model(embeds)
                prob = logits
                pred = torch.where(prob > 0.5, 1, 0)
                loss = criterion(logits, labels)
                test_loss += loss.item()
                auc, ks, acc = calc_auc(labels, prob.detach()), calc_ks(labels, prob.detach()), calc_acc(labels, pred)
                pre, recall, f1 = calc_f1(labels, pred)
                results.append([auc, ks, pre, recall, f1, acc])
            results = np.array(results)
            results = np.mean(results, axis=0)

            if results[0] > bestauc:
                bestauc = results[0]
                bestepoch = epoch
                torch.save(model.state_dict(), 'XXX')

            t1 = time()
            log.record("Epoch: %d, Test Loss: %.2f, AUC: %.4f, KS: %.4f, PRE: %.4f, RECALL: %.4f F1:%.4f, ACC: %.4f, Time: %.1f" % (
                epoch, test_loss, results[0], results[1], results[2], results[3], results[4], results[5], t1-t0)
            )
            totaltime += (t1-t0)
    
    log.record("Best Epoch[%d] Best AUC Score[%.4f] Total Time[%.1f]" % (bestepoch, bestauc, totaltime))
    

def get_embed():
    trainset = SnapshotDataset(is_train_set=True)
    pretrainset = PretrainDataset()
    pretrainloader = DataLoader(pretrainset, batch_size=32*args.batch_size, shuffle=False,drop_last=False)
    print("Num_nodes: ", pretrainset.num_nodes)
    embed_table = torch.zeros(trainset.num_nodes, args.hid_dim)

    model = TFEncoderClassifier(trainset.feat_dim, args.hid_dim, trainset.time_span, args.batch_size, args.n_layers)
    model.load_state_dict(torch.load('XXX.pth'))
    with torch.no_grad():
        for _, (idx, embeds) in enumerate(tqdm(pretrainloader)):
            h = model(embeds, encode=True)
            embed_table[idx] = h
    torch.save(embed_table, "XXX/tfencoder_Xinye_tss_embed.pt")

if __name__ == '__main__':   
    run()
    get_embed()
