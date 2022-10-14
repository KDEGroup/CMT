# coding=utf-8
import os, sys, argparse
sys.path.append("XXX")

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
import copy
from tqdm import tqdm
from Utils import Logging, set_seed
from Metrics import *
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

log_dir = "XXX"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir, 'Contrastive_tss.log')
log = Logging(log_path)

parser = argparse.ArgumentParser(description='node_seq')
parser.add_argument("--hid_dim", type=int, default=32, help="Hidden layer dimension")
parser.add_argument("--n_layers", type=int, default=2, help="Number of layers")
parser.add_argument("--batch_size", type=int, default=1024*8, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--beta", type=float, default=0.4, help="Ratio for shuffling sub sequences")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for regularization")
parser.add_argument("--epochs", type=int, default=25, help="Epochs for training")
parser.add_argument("--path", type=str, default="XXX", help="Dataset path")
parser.add_argument("--pos_embed", type=bool, default=False, help="Whether to use position embedding")
parser.add_argument("--data", type=str, default="XXX/IHG_Xinye_pretrain_embed.pt", help="Dataset")

args = parser.parse_args()
log.record(args)


class SnapshotDataset(Dataset):
    def __init__(self, is_train_set=True):
        # shape: [timespan, num_nodes, feat_dim] -> [num_nodes, timespan, feat_dim]
        embed_table = torch.load(args.data)
        embed_table = torch.transpose(embed_table, dim0=0, dim1=1)
        with open("XXX/u_train_test_Xinye.pickle", "rb") as fp:
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
        
class NPairDiscriminator(nn.Module):
    def __init__(self, hid_dim, tau=0.5):
        super(NPairDiscriminator, self).__init__()
        self.tau = tau
        self.fc1 = torch.nn.Linear(hid_dim, hid_dim)
        self.fc2 = torch.nn.Linear(hid_dim, hid_dim)

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    
    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def forward(self, z1, z2):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        
        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)

        ret = (l1 + l2) * 0.5
        return ret.mean()


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


class Reorder(object):
    """Randomly shuffle a continuous sub-sequence"""

    def __init__(self, beta=0.2):
        self.beta = beta

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        sub_seq_length = int(self.beta * len(copied_sequence))
        start_index = random.randint(0, len(copied_sequence) - sub_seq_length - 1)
        sub_seq = copied_sequence[start_index : start_index + sub_seq_length]
        random.shuffle(sub_seq)
        reordered_seq = copied_sequence[:start_index] + sub_seq + copied_sequence[start_index + sub_seq_length :]
        assert len(copied_sequence) == len(reordered_seq)
        return reordered_seq


class MHTGNN(nn.Module):
    def __init__(self, input_size, time_span, hidden_size, n_layers, batch_size):
        super(MHTGNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.position_encoder = nn.Embedding(time_span, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(p=0.5)
        transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=512)
        if n_layers > 1:
            self.transformer_encoder = nn.TransformerEncoder(transformer_encoder, num_layers=n_layers)
        else:
            self.transformer_encoder = transformer_encoder        
        self.embedding = nn.Linear(input_size, hidden_size)
        self.predict = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.disc = NPairDiscriminator(hidden_size)

    def add_position_embedding(self, input_seq):
        # input shape: [batch_size, timespan(seq_len), feat_dim]
        input_seq = self.embedding(input_seq)
        batch_size = input_seq.shape[0]
        time_span = input_seq.shape[1]
        position_ids = np.array(batch_size*list(range(time_span))).reshape(batch_size, time_span)
        position_ids = torch.LongTensor(position_ids)
        position_embeddings = self.position_encoder(position_ids)
        sequence_emb = input_seq + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb


    def forward(self, input_embed, aug_embed1, aug_embed2):
        # input shape: [batch_size, timespan(seq_len), feat_dim]
        if args.pos_embed:
            input_embed = self.add_position_embedding(input_embed)
            aug_embed1 = self.add_position_embedding(aug_embed1)
            aug_embed2 = self.add_position_embedding(aug_embed2)
        else:
            input_embed = self.embedding(input_embed)
            aug_embed1 = self.embedding(aug_embed1)
            aug_embed2 = self.embedding(aug_embed2)
        # input shape: [timespan(seq_len), batch_size, feat_dim]
        input_embed = torch.transpose(input_embed, dim0=0, dim1=1)
        aug_embed1 = torch.transpose(aug_embed1, dim0=0, dim1=1)
        aug_embed2 = torch.transpose(aug_embed2, dim0=0, dim1=1)
        diag_val = torch.ones(input_embed.shape[0], input_embed.shape[0])
        mask = torch.triu(diag_val)
        if self.n_layers > 1:
            z = self.transformer_encoder(src=input_embed, mask=mask)[-1]
            z1 = self.transformer_encoder(src=aug_embed1, mask=mask)[-1]
            z2 = self.transformer_encoder(src=aug_embed2, mask=mask)[-1]
        else:
            z = self.transformer_encoder(src=input_embed, src_mask=mask)[-1]
            z1 = self.transformer_encoder(src=aug_embed1, src_mask=mask)[-1]
            z2 = self.transformer_encoder(src=aug_embed2, src_mask=mask)[-1]

        contrastive_loss = self.disc(z1, z2)
        logits = self.predict(z)
        return logits, contrastive_loss

    def get_embed(self, input_embed):
        # input shape: [batch_size, timespan(seq_len), feat_dim]
        if args.pos_embed:
            input_embed = self.add_position_embedding(input_embed)
        else:
            input_embed = self.embedding(input_embed)
        # input shape: [timespan(seq_len), batch_size, feat_dim]
        input_embed = torch.transpose(input_embed, dim0=0, dim1=1)
        diag_val = torch.ones(input_embed.shape[0], input_embed.shape[0])
        mask = torch.triu(diag_val)
        if self.n_layers > 1:
            z = self.transformer_encoder(src=input_embed, mask=mask)[-1]
        else:
            z = self.transformer_encoder(src=input_embed, src_mask=mask)[-1]
        
        return z

def run():
    set_seed(42)
    trainset = SnapshotDataset(is_train_set=True)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    testset = SnapshotDataset(is_train_set=False)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    pretrainset = PretrainDataset()
    pretrainloader = DataLoader(pretrainset, batch_size=16*args.batch_size, shuffle=False,drop_last=False)
    print("Num_nodes: ", pretrainset.num_nodes)
    embed_table = torch.empty(trainset.num_nodes, args.hid_dim)

    model = MHTGNN(trainset.feat_dim, trainset.time_span, args.hid_dim, args.n_layers, args.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss(reduction='sum')
    augment = Reorder(beta=args.beta)

    bestauc = 0.
    bestepoch = 0
    totaltime = 0.
    for epoch in range(1, args.epochs+1):
        t0 = time()
        train_loss = 0.
        model.train()
        results = []
        for _, (input_seqs, train_labels) in enumerate(tqdm(trainloader)):
            optimizer.zero_grad()
            idx1 = []
            idx2 = []
            for seq in input_seqs:
                coordinates = torch.nonzero(seq)
                if coordinates.nelement() == 0:
                    start = trainset.time_span
                else:
                    start = coordinates[0][0].item()
                pre = [i for i in range(start)]
                sub = [i for i in range(start, trainset.time_span)]
                if sub:
                    aug1, aug2 = augment(sub), augment(sub)
                    idx1.append(pre + aug1)
                    idx2.append(pre + aug2)
                else:
                    idx1.append(pre)
                    idx2.append(pre)
            aug_seqs1 = torch.empty_like(input_seqs)
            aug_seqs2 = torch.empty_like(input_seqs)
            for i in range(input_seqs.shape[0]):
                aug_seqs1[i] = input_seqs[i][torch.tensor(idx1[i]), :]
                aug_seqs2[i] = input_seqs[i][torch.tensor(idx2[i]), :]
            logits, contrastive_loss = model(input_seqs, aug_seqs1, aug_seqs2)
            prob = logits
            pred = torch.where(prob > 0.5, 1, 0)
            classify_loss = criterion(logits, train_labels)
            loss = classify_loss + contrastive_loss
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
            for _, (input_seqs, test_labels) in enumerate(tqdm(testloader)):
                idx1 = []
                idx2 = []
                for seq in input_seqs:
                    coordinates = torch.nonzero(seq)
                    if coordinates.nelement() == 0:
                        start = trainset.time_span
                    else:
                        start = coordinates[0][0].item()
                    pre = [i for i in range(start)]
                    sub = [i for i in range(start, trainset.time_span)]
                    if sub:
                        aug1, aug2 = augment(sub), augment(sub)
                        idx1.append(pre + aug1)
                        idx2.append(pre + aug2)
                    else:
                        idx1.append(pre)
                        idx2.append(pre)
                aug_seqs1 = torch.empty_like(input_seqs)
                aug_seqs2 = torch.empty_like(input_seqs)
                for i in range(input_seqs.shape[0]):
                    aug_seqs1[i] = input_seqs[i][torch.tensor(idx1[i]), :]
                    aug_seqs2[i] = input_seqs[i][torch.tensor(idx2[i]), :]
                logits, contrastive_loss = model(input_seqs, aug_seqs1, aug_seqs2)
                prob = logits
                pred = torch.where(prob > 0.5, 1, 0)
                classify_loss = criterion(logits, test_labels)
                loss = classify_loss + contrastive_loss
                test_loss += loss.item()
                auc, ks, acc = calc_auc(test_labels, prob.detach()), calc_ks(test_labels, prob.detach()), calc_acc(test_labels, pred)
                pre, recall, f1 = calc_f1(test_labels, pred)
                results.append([auc, ks, pre, recall, f1, acc])
            results = np.array(results)
            results = np.mean(results, axis=0)
            if results[0] > bestauc:
                bestauc = results[0]
                bestepoch = epoch
                torch.save(model.state_dict(), 'XXX/TFEncoder_tss_Npair_params.pth')

            t1 = time()
            log.record("Epoch: %d, Test Loss: %.2f, AUC: %.4f, KS: %.4f, PRE: %.4f, RECALL: %.4f F1:%.4f, ACC: %.4f, Time: %.1f" % (
                epoch, test_loss, results[0], results[1], results[2], results[3], results[4], results[5], t1-t0)
            )
            totaltime += (t1-t0)

    log.record("Best Epoch[%d] Best AUC Score[%.4f] Total Time[%.1f]" % (bestepoch, bestauc, totaltime))

def get_embed():
    trainset = SnapshotDataset(is_train_set=True)
    pretrainset = PretrainDataset()
    pretrainloader = DataLoader(pretrainset, batch_size=16*args.batch_size, shuffle=False, drop_last=False)
    print("Num_nodes: ", pretrainset.num_nodes)
    embed_table = torch.empty(trainset.num_nodes, args.hid_dim)

    model = MHTGNN(trainset.feat_dim, trainset.time_span, args.hid_dim, args.n_layers, args.batch_size)
    model.load_state_dict(torch.load('XXX/TFEncoder_tss_Npair_params.pth'))
    with torch.no_grad():
        for _, (idx, input_seqs) in enumerate(tqdm(pretrainloader)):
            h = model.get_embed(input_seqs)
            embed_table[idx] = h
    torch.save(embed_table, "XXX/tfencoder_Xinye_tss_Npair_embed.pt")

if __name__ == '__main__':   
    run()
    get_embed()
