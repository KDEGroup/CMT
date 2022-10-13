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

log_dir = "/home/zqxu/MHTGNN/log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir, 'LSTM_TSS.log')
log = Logging(log_path)

parser = argparse.ArgumentParser(description='node_seq')
parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
parser.add_argument("--n_layers", type=int, default=2, help="Number of layers")
parser.add_argument("--batch_size", type=int, default=1024*4)
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for regularization")
parser.add_argument("--epochs", type=int, default=100, help="Epochs for training")
parser.add_argument("--data", type=str, default="/home/zqxu/MHTGNN/data/embedding/IHG_Xinye_pretrain_embed.pt", help="Dataset")

args = parser.parse_args()
log.record(args)

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
        self.feat_dim = embed_table.shape[2]
        assert self.time_span == 13
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
        assert self.time_span == 13

    def __getitem__(self, index):
        return index, self.embed_table[index]

    def __len__(self):
        return self.num_nodes

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, bidirectional=False):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1
        self.predict = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        self.embedding = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, bidirectional=bidirectional, dropout=0.5)
        
    
    def forward(self, input, encode=False):
        # input shape: [batch_size, timespan(seq_len), feat_dim] -> [timespan(seq_len), batch_size, feat_dim]
        batch_size = input.shape[0]
        self.hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        self.cell = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        input = self.embedding(input)
        input = torch.transpose(input, dim0=0, dim1=1)
        output, (hidden, _) = self.lstm(input, (self.hidden, self.cell))
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]
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
    pretrainset = PretrainDataset()
    pretrainloader = DataLoader(pretrainset, batch_size=32*args.batch_size, shuffle=False,drop_last=False)
    print("Num_nodes: ", pretrainset.num_nodes)
    embed_table = torch.zeros(trainset.num_nodes, args.hid_dim)

    model = LSTMClassifier(trainset.feat_dim, args.hid_dim, args.n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss(reduction='sum')
    bestauc = 0.
    bestepoch = 0
    totaltime = 0.
    for epoch in range(1, args.epochs+1):
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

            if epoch > 20 and results[0] > bestauc:
                bestauc = results[0]
                bestepoch = epoch
                for _, (idx, embeds) in enumerate(tqdm(pretrainloader)):
                    h = model(embeds, encode=True)
                    embed_table[idx] = h

            t1 = time()
            log.record("Epoch: %d, Test Loss: %.2f, AUC: %.4f, KS: %.4f, PRE: %.4f, RECALL: %.4f F1:%.4f, ACC: %.4f, Time: %.1f" % (
                epoch, test_loss, results[0], results[1], results[2], results[3], results[4], results[5], t1-t0)
            )
            totaltime += (t1-t0)
    
    log.record("Best Epoch[%d] Best AUC Score[%.4f] Total Time[%.1f]" % (bestepoch, bestauc, totaltime))
    torch.save(model.state_dict(), '/home/zqxu/MHTGNN/model_save/LSTM_tss_params.pth')
    torch.save(embed_table, "/home/zqxu/MHTGNN/data/embedding/Xinye_lstm_aggregate_embed.pt")

if __name__ == '__main__':   
    run()