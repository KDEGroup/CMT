# coding=utf-8
import os, sys, argparse

from time import time
import torch
import torch.nn as nn
import torch.optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pickle
import random
from Utils import Logging, set_seed
from Metrics import *
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

log_dir = "XXX"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir, 'TFEncoder_urs.log')
log = Logging(log_path)

parser = argparse.ArgumentParser(description='tfencoder_rs')
parser.add_argument("--hid_dim", type=int, default=32, help="Hidden layer dimension")
parser.add_argument("--n_layers", type=int, default=2, help="Number of layers")
parser.add_argument("--batch_size", type=int, default=1024*4)
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for regularization")
parser.add_argument("--epochs", type=int, default=25, help="Epochs for training")
parser.add_argument("--path", type=str, default="XXX", help="Dataset path")
parser.add_argument("--data", type=str, default="XXX/sampled_user_seq_complete.pickle", help="Dataset path")
parser.add_argument("--pos_embed", type=bool, default=False, help="Whether to use position embedding")

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
        with open(args.data, 'rb') as f:
            user_seq_table = pickle.load(f)
        with open(args.path + "u_train_test_Xinye.pickle", "rb") as fp:
            X_train_p, X_train_n, X_test_p, X_test_n = pickle.load(fp)
        X_train = X_train_p + X_train_n
        X_test = X_test_p + X_test_n
        self.num_nodes = len(user_seq_table)

        graph_data = np.load("XXX/phase1_gdata.npz")
        user_features = torch.FloatTensor(graph_data['x'])

        self.num_features = user_features.shape[1]
        self.user_features = user_features

        self.time_span = 14
        if is_train_set:
            self.seq_table = [user_seq_table[i] for i in X_train]
            self.users = np.array(X_train)
            self.len = len(X_train)
            self.label = torch.zeros(max(X_train) + 1, dtype=torch.int64)
            self.label[torch.LongTensor(X_train_p)] = 1
            self.label = self.label[X_train].reshape(-1, 1).float()
        else:
            self.seq_table = [user_seq_table[i] for i in X_test]
            self.users = np.array(X_test)
            self.len = len(X_test)
            self.label = torch.zeros(max(X_test) + 1, dtype=torch.int64)
            self.label[torch.LongTensor(X_test_p)] = 1
            self.label = self.label[X_test].reshape(-1, 1).float()
        data = list(zip(self.seq_table, self.label))
        random.shuffle(data)
        self.seq_table, self.label = zip(*data)

    def __getitem__(self, idx):
        user_seqs = []
        for seq in self.seq_table[idx]:
            if seq not in user_seqs:
                user_seqs.append(seq)
        label = self.label[idx]
        labels = torch.repeat_interleave(label, len(user_seqs))
        user_seq_input = []
        for seq in user_seqs:
            seq_len = len(seq)
            input = torch.empty(seq_len, self.num_features)
            for j in range(len(seq)):
                input[j] = self.user_features[int(seq[j])]
            user_seq_input.append(input)
        return user_seq_input, labels

    def __len__(self):
        return self.len

class PretrainDataset(Dataset):
    def __init__(self):
        with open(args.data, 'rb') as f:
            user_seq_table = pickle.load(f)
        self.seq_table = user_seq_table
        self.num_nodes = len(user_seq_table)
        graph_data = np.load("XXX/phase1_gdata.npz")
        user_features = torch.FloatTensor(graph_data['x'])

        self.num_features = user_features.shape[1]
        self.user_features = user_features

    def __getitem__(self, index):
        user_seqs = []
        for seq in self.seq_table[index]:
            if seq not in user_seqs:
                user_seqs.append(seq)
        user_seq_input = []
        for seq in user_seqs:
            seq_len = len(seq)
            input = torch.empty(seq_len, self.num_features)
            for j in range(len(seq)):
                input[j] = self.user_features[int(seq[j])]
            user_seq_input.append(input)
        return user_seq_input

    def __len__(self):
        return self.num_nodes


class TFEncoderClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, time_span, n_layers=1, bidirectional=False):
        super(TFEncoderClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_span = time_span
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1
        self.epsilon = torch.FloatTensor([1e-12])

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
        # input_seq: [timespan(seq_len), batch_size, hidden_size]
        batch_size = input_seq.shape[1]
        time_span = input_seq.shape[0]
        position_ids = np.array(batch_size*list(range(time_span))).reshape(batch_size, time_span)
        position_ids = torch.LongTensor(position_ids)
        position_embeddings = self.position_encoder(position_ids)
        input_seq = torch.transpose(input_seq, dim0=0, dim1=1)
        sequence_emb = input_seq + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def get_embed(self, input_seq):
        # input_seq: [timespan(seq_len), batch_size, hidden_size]
        for i in range(len(input_seq)):
            input_seq[i] = self.embedding(input_seq[i])
        input_seq = pad_sequence(input_seq)     
        if args.pos_embed:
            input_embed = torch.transpose(self.add_position_embedding(input_seq), dim0=0, dim1=1)
        else:
            input_embed = input_seq
        diag_val = torch.ones(input_embed.shape[0], input_embed.shape[0])
        mask = torch.triu(diag_val)
        if self.n_layers > 1:
            z = self.transformer_encoder(src=input_embed, mask=mask)[-1]
        else:
            z = self.transformer_encoder(src=input_embed, src_mask=mask)[-1]
        return z

    def forward(self, input_seq, encode=False):
        # input shape: [timespan(seq_len), batch_size, hidden_size]
        for i in range(len(input_seq)):
            input_seq[i] = self.embedding(input_seq[i])
        input_seq = pad_sequence(input_seq)
        # input shape: [timespan(seq_len), batch_size, hidden_size]      
        if args.pos_embed:
            input_embed = torch.transpose(self.add_position_embedding(input_seq), dim0=0, dim1=1)
        else:
            input_embed = input_seq
        # 加入self-attention
        diag_val = torch.ones(input_embed.shape[0], input_embed.shape[0])
        mask = torch.triu(diag_val)
        if self.n_layers > 1:
            tf_output = self.transformer_encoder(src=input_embed, mask=mask)
        else:
            tf_output = self.transformer_encoder(src=input_embed, src_mask=mask)
        hidden_cat = tf_output[-1]
        if encode:
            return hidden_cat
        logits = self.predict(hidden_cat)
        return logits    

def run():
    set_seed(42)
    trainset = SnapshotDataset(is_train_set=True)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=lambda x:x)
    testset = SnapshotDataset(is_train_set=False)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=lambda x:x)
    model = TFEncoderClassifier(trainset.num_features, args.hid_dim, trainset.time_span, args.n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss(reduction='sum')

    bestauc = 0.
    bestepoch = 0
    for epoch in range(1, args.epochs+1):
        t0 = time()
        train_loss = 0.
        model.train()
        results = []
        for batch_id, batch_data in enumerate(tqdm(trainloader)):
            input_seq = []
            train_labels = []
            for i in range(len(batch_data)):
                for initial_seq in batch_data[i][0]:
                    input_seq.append(initial_seq)
                train_labels.append(batch_data[i][1])
            train_labels = torch.cat(train_labels, dim=0).reshape(-1, 1)
            optimizer.zero_grad()
            logits = model(input_seq)
            prob = logits
            pred = torch.where(prob > 0.5, 1, 0)
            loss = criterion(logits, train_labels)
            loss.backward() 
            optimizer.step()
            train_loss += loss.item()
            auc, ks, acc = calc_auc(train_labels, prob.detach()), calc_ks(train_labels, prob.detach()), calc_acc(train_labels, pred)
            pre, recall, f1 = calc_f1(train_labels, pred)
            results.append([auc, ks, pre, recall, f1, acc])
        results = np.array(results)
        results = np.mean(results, axis=0)
        log.record("Epoch: %d, Train Loss: %.2f, AUC: %.4f, KS: %.4f, PRE: %.4f, RECALL: %.4f F1:%.4f, ACC: %.4f" % (
            epoch, train_loss, results[0], results[1], results[2], results[3], results[4], results[5])
        )

        test_loss = 0.
        model.eval()
        results = []
        with torch.no_grad():
            for batch_id, batch_data in enumerate(tqdm(testloader)):
                input_seq = []
                test_labels = []
                for i in range(len(batch_data)):
                    for initial_seq in batch_data[i][0]:
                        input_seq.append(initial_seq)
                    test_labels.append(batch_data[i][1])
                test_labels = torch.cat(test_labels, dim=0).reshape(-1, 1)
                logits = model(input_seq)
                prob = logits
                pred = torch.where(prob > 0.5, 1, 0)
                loss = criterion(logits, test_labels)
                test_loss += loss.item()
                auc, ks, acc = calc_auc(test_labels, prob.detach()), calc_ks(test_labels, prob.detach()), calc_acc(test_labels, pred)
                pre, recall, f1 = calc_f1(test_labels, pred)
                results.append([auc, ks, pre, recall, f1, acc])
            results = np.array(results)
            results = np.mean(results, axis=0)

            if results[0] > bestauc:
                bestauc = results[0]
                bestepoch = epoch
                torch.save(model.state_dict(), 'XXX/TFEncoder_rs_params.pth')

            t1 = time()
            log.record("Epoch: %d, Test Loss: %.2f, AUC: %.4f, KS: %.4f, PRE: %.4f, RECALL: %.4f F1:%.4f, ACC: %.4f, Time: %.1f" % (
                epoch, test_loss, results[0], results[1], results[2], results[3], results[4], results[5], t1-t0)
            )
    
    log.record("Best Epoch[%d] Best AUC Score[%.4f]" % (bestepoch, bestauc))


def generate_embed():
    trainset = SnapshotDataset(is_train_set=True)
    pretrainset = PretrainDataset()
    pretrainloader = DataLoader(pretrainset, batch_size=32*args.batch_size, shuffle=False, drop_last=False, collate_fn=lambda x:x)
    model = TFEncoderClassifier(trainset.num_features, args.hid_dim, trainset.time_span, args.n_layers)
    model.load_state_dict(torch.load('XXX/TFEncoder_rs_params.pth'))
    embed_table = torch.empty(pretrainset.num_nodes, args.hid_dim)
    with torch.no_grad():
        user = 0
        for batch_id, batch_data in enumerate(tqdm(pretrainloader)):
            t0 = time()
            input_seq = []
            batch_size = len(batch_data)
            for i in range(batch_size):
                for initial_seq in batch_data[i]:
                    input_seq.append(initial_seq)
            h = model.get_embed(input_seq)
            for i in range(batch_size):
                embed_table[user] = h[i]
                user+=1
            t1 = time()
            # print("%d, %d, %d, %.2f" % (batch_id, user, pretrainset.num_nodes, t1-t0))
    torch.save(embed_table, "XXX/tfencoder_Xinye_rs_embed.pt")


if __name__ == '__main__':
    run()   
    generate_embed()
