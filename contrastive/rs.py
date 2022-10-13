# coding=utf-8
import os, sys, argparse
sys.path.append("/home/zqxu/MHTGNN/code/")

from time import time
from tqdm import tqdm
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
import random, copy
from Utils import Logging, set_seed
from Metrics import *
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

log_dir = "/home/zqxu/MHTGNN/log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir, 'Contrastive_rs.log')
log = Logging(log_path)

parser = argparse.ArgumentParser(description='tfencoder_rs')
parser.add_argument("--hid_dim", type=int, default=32, help="Hidden layer dimension")
parser.add_argument("--n_layers", type=int, default=2, help="Number of layers")
parser.add_argument("--batch_size", type=int, default=1024*8)
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for regularization")
parser.add_argument("--epochs", type=int, default=25, help="Epochs for training")
parser.add_argument("--path", type=str, default="/home/zqxu/MHTGNN/data/", help="Dataset path")
parser.add_argument("--data", type=str, default="/home/zqxu/MHTGNN/data/sampled_user_seq_complete.pickle", help="Dataset path")
parser.add_argument("--pos_embed", type=bool, default=False, help="Whether to use position embedding")
parser.add_argument("--substitute_ratio", type=float, default=0.8, help="Substitute ratio")

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

class SnapshotDataset(Dataset):
    def __init__(self, is_train_set=True):
        # shape: [timespan, num_nodes, feat_dim] -> [num_nodes, timespan, feat_dim]
        with open(args.data, 'rb') as f:
            user_seq_table = pickle.load(f)
        with open(args.path + 'node_augment_hash_with_hetero.pickle', 'rb') as f:
            node_augment_hash = pickle.load(f)
        with open("/home/zqxu/MHTGNN/data/u_train_test_Xinye.pickle", "rb") as fp:
            X_train_p, X_train_n, X_test_p, X_test_n = pickle.load(fp)
        X_train = X_train_p + X_train_n
        X_test = X_test_p + X_test_n
        self.num_nodes = len(user_seq_table)

        graph_data = np.load("/home/zqxu/MHTGNN/data/phase1_gdata.npz")
        user_features = torch.FloatTensor(graph_data['x'])

        self.num_features = user_features.shape[1]
        self.user_features = user_features
        self.node_augment_hash = node_augment_hash
        self.reorder = Reorder(beta=0.5)

        self.time_span = 7
        self.augment_ratio = args.substitute_ratio
        
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

    def replace(self, seq):
        copied_seq = copy.deepcopy(seq)
        seq_len = len(copied_seq)
        num_replace = int(seq_len * self.augment_ratio)
        replace_idx = random.sample(range(seq_len), num_replace)
        for idx in replace_idx:
            if self.node_augment_hash[copied_seq[idx]]:
                node_dict = self.node_augment_hash[copied_seq[idx]]
                aug_node = random.choice(list(node_dict.keys()))
                if node_dict[aug_node]:
                    copied_seq[idx] = random.choice(node_dict[aug_node])
        return copied_seq

    def input_feature_construct(self, seq):
        seq_len = len(seq)
        input = torch.empty(seq_len, self.num_features)
        for j in range(len(seq)):
            input[j] = self.user_features[int(seq[j])]
        return input

    def __getitem__(self, idx):
        user_seqs = []
        for seq in self.seq_table[idx]:
            if seq and seq not in user_seqs:
                user_seqs.append(seq)
        label = self.label[idx]
        labels = torch.repeat_interleave(label, len(user_seqs))
        user_seq_input = []
        user_seq_input_replace1 = []
        user_seq_input_replace2 = []
        for seq in user_seqs:
            input = self.input_feature_construct(seq)
            user_seq_input.append(input)

            aug_seq1, aug_seq2 = self.replace(seq), self.reorder(seq)
            aug_input1, aug_input2 = self.input_feature_construct(aug_seq1), self.input_feature_construct(aug_seq2)
            user_seq_input_replace1.append(aug_input1)
            user_seq_input_replace2.append(aug_input2)

        return user_seq_input, user_seq_input_replace1, user_seq_input_replace2, labels

    def __len__(self):
        return self.len

class PretrainDataset(Dataset):
    def __init__(self):
        with open(args.data, 'rb') as f:
            user_seq_table = pickle.load(f)
        self.seq_table = user_seq_table
        self.num_nodes = len(user_seq_table)
        graph_data = np.load("/home/zqxu/MHTGNN/data/phase1_gdata.npz")
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

class TFEncoderClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, time_span, n_layers=1, bidirectional=False):
        super(TFEncoderClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_span = time_span
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1
        self.epsilon = torch.FloatTensor([1e-12])
        self.disc = NPairDiscriminator(hidden_size)

        self.predict = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.position_encoder = nn.Embedding(time_span, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(p=0.5)
        self.embedding = nn.Linear(input_size, hidden_size)
        transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=256)
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

    def forward(self, input_seq, input_aug_seq1, input_aug_seq2, encode=False):
        # input shape: [timespan(seq_len), batch_size, hidden_size]
        for i in range(len(input_seq)):
            input_seq[i] = self.embedding(input_seq[i])
            input_aug_seq1[i] = self.embedding(input_aug_seq1[i])
            input_aug_seq2[i] = self.embedding(input_aug_seq2[i])
        input_seq = pad_sequence(input_seq)
        input_aug_seq1 = pad_sequence(input_aug_seq1)
        input_aug_seq2 = pad_sequence(input_aug_seq2)
        # input shape: [timespan(seq_len), batch_size, hidden_size]      
        if args.pos_embed:
            input_embed = torch.transpose(self.add_position_embedding(input_seq), dim0=0, dim1=1)
            input_aug_embed1 = torch.transpose(self.add_position_embedding(input_aug_seq1), dim0=0, dim1=1)
            input_aug_embed2 = torch.transpose(self.add_position_embedding(input_aug_seq2), dim0=0, dim1=1)
        else:
            input_embed = input_seq
            input_aug_embed1 = input_aug_seq1
            input_aug_embed2 = input_aug_seq2
        # 加入self-attention
        diag_val = torch.ones(input_embed.shape[0], input_embed.shape[0])
        mask = torch.triu(diag_val)
        if self.n_layers > 1:
            z = self.transformer_encoder(src=input_embed, mask=mask)[-1]
            z1 = self.transformer_encoder(src=input_aug_embed1, mask=mask)[-1]
            z2 = self.transformer_encoder(src=input_aug_embed2, mask=mask)[-1]
        else:
            z = self.transformer_encoder(src=input_embed, src_mask=mask)[-1]
            z1 = self.transformer_encoder(src=input_aug_embed1, src_mask=mask)[-1]
            z2 = self.transformer_encoder(src=input_aug_embed2, src_mask=mask)[-1]
        if encode:
            return z
        
        contrastive_loss = self.disc(z1, z2)
        logits = self.predict(z)
        return logits, contrastive_loss   

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
            input_aug_seq1 = []
            input_aug_seq2 = []
            train_labels = []
            for i in range(len(batch_data)):
                for initial_seq in batch_data[i][0]:
                    input_seq.append(initial_seq)
                for aug_seq in batch_data[i][1]:
                    input_aug_seq1.append(aug_seq)
                for aug_seq in batch_data[i][2]:
                    input_aug_seq2.append(aug_seq)
                train_labels.append(batch_data[i][3])
            train_labels = torch.cat(train_labels, dim=0).reshape(-1, 1)
            optimizer.zero_grad()
            logits, contrastive_loss = model(input_seq, input_aug_seq1, input_aug_seq2)
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
            for batch_id, batch_data in enumerate(tqdm(testloader)):
                input_seq = []
                input_aug_seq1 = []
                input_aug_seq2 = []
                test_labels = []
                for i in range(len(batch_data)):
                    for initial_seq in batch_data[i][0]:
                        input_seq.append(initial_seq)
                    for aug_seq in batch_data[i][1]:
                        input_aug_seq1.append(aug_seq)
                    for aug_seq in batch_data[i][2]:
                        input_aug_seq2.append(aug_seq)
                    test_labels.append(batch_data[i][3])
                test_labels = torch.cat(test_labels, dim=0).reshape(-1, 1)
                optimizer.zero_grad()
                logits, contrastive_loss = model(input_seq, input_aug_seq1, input_aug_seq2)
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
                torch.save(model.state_dict(), '/home/zqxu/MHTGNN/model_save/TFEncoder_rs_Npair_params_%.1f.pth' % args.substitute_ratio)

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
    model.load_state_dict(torch.load('/home/zqxu/MHTGNN/model_save/TFEncoder_rs_Npair_params_%.1f.pth' % args.substitute_ratio))
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
            print("%d, %d, %d, %.2f" % (batch_id, user, pretrainset.num_nodes, t1-t0))
    torch.save(embed_table, "/home/zqxu/MHTGNN/data/embedding/tfencoder_Xinye_rs_Npair_embed.pt")


if __name__ == '__main__':   
    run()
    generate_embed()
