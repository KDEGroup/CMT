import argparse, os, sys
sys.path.append("/home/zqxu/MHTGNN/code/")

from time import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from pathlib import Path
import pickle
import dgl

from Utils import Logging, set_seed
from models.clf_model import Classifier
from models.dci import DCI
from sklearn.cluster import KMeans
from Metrics import *

import warnings
warnings.filterwarnings("ignore")

log_dir = "/home/zqxu/MHTGNN/log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir, 'DCI.log')
log = Logging(log_path)

parser = argparse.ArgumentParser(description='PyTorch deep cluster infomax')
parser.add_argument('--dataset', type=str, default="Xinye",
                    help='name of dataset (default: wiki)')
parser.add_argument('--path', type=str, default="/home/zqxu/MHTGNN/model_save/",
                    help='path of dataset')
parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train (default: 50)')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of layers (default: 2)')
parser.add_argument('--num_mlp_layers', type=int, default=2,
                    help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
parser.add_argument('--hidden_dim', type=int, default=64,
                    help='number of hidden units (default: 64)')
parser.add_argument('--batch_size', type=int, default=1024*8,
                    help='Batch size')
parser.add_argument('--finetune_epochs', type=int, default=200,
                    help='number of finetune epochs (default: 100)')
parser.add_argument('--num_folds', type=int, default=10,
                    help='number of folds (default: 10)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.01)')
parser.add_argument('--num_cluster', type=int, default=25,
                    help='number of clusters (default: 2)')
parser.add_argument('--recluster_interval', type=int, default=20,
                    help='the interval of reclustering (default: 20)')
parser.add_argument('--final_dropout', type=float, default=0.4,
                    help='final layer dropout (default: 0.5)')
parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average"],
                    help='Pooling for over neighboring nodes: sum or average')
parser.add_argument('--training_scheme', type=str, default="decoupled", choices=["decoupled", "joint"],
                    help='Training schemes: decoupled or joint')
args = parser.parse_args()
log.record(args)

def preprocess_neighbors_sumavepool(edge_index, nb_nodes, device):
    adj_idx = edge_index.T
        
    adj_idx_2 = torch.cat([torch.unsqueeze(adj_idx[1], 0), torch.unsqueeze(adj_idx[0], 0)], 0)
    adj_idx = torch.cat([adj_idx, adj_idx_2], 1)

    self_loop_edge = torch.LongTensor([range(nb_nodes), range(nb_nodes)])
    adj_idx = torch.cat([adj_idx, self_loop_edge], 1)
        
    adj_elem = torch.ones(adj_idx.shape[1])

    adj = torch.sparse.FloatTensor(adj_idx, adj_elem, torch.Size([nb_nodes, nb_nodes]))

    return adj.to(device)

def loadXinyeData():
    graph_data = np.load("/home/zqxu/MHTGNN/data/phase1_gdata.npz")
    edge_index = graph_data['edge_index']
    feats = graph_data['x']
    nb_nodes = feats.shape[0]
    labels = torch.FloatTensor(graph_data['y'])
    # users = np.array([i for i in range(nb_nodes)]).reshape(-1, 1)
    # labels = np.concatenate((users, labels), axis=1)

    with open("/home/zqxu/MHTGNN/data/u_train_test_Xinye.pickle", "rb") as fp:
        X_train_p, X_train_n, X_test_p, X_test_n = pickle.load(fp)
    train_user_idx = X_train_p + X_train_n
    test_user_idx = X_test_p + X_test_n

    return edge_index, feats, train_user_idx, test_user_idx, labels, nb_nodes

def cal_metrics(output, node, idx, label, criterion, test_graph):
    logits = output[node]
    prob = logits
    pred = torch.where(prob > 0.5, 1, 0)
    label = label.reshape(-1, 1)
    loss = criterion(prob, label)
    auc, ks, f1, acc = calc_auc(label, prob.detach()), calc_ks(label, prob.detach()), calc_f1(label, pred), calc_acc(label, pred)

    return loss, auc, ks, f1, acc

def finetune(args, model_pretrain, device, test_graph, feats_num):
    # initialize the joint model
    model = Classifier(args.num_layers, args.num_mlp_layers, feats_num, args.hidden_dim, args.final_dropout, args.neighbor_pooling_type, device).to(device)
    # replace the encoder in joint model with the pre-trained encoder
    pretrained_dict = model_pretrain.state_dict()
    model_dict = model.state_dict()
    pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion_tune = nn.BCELoss(reduction='sum')

    res = []
    # test_graph = (feats, adj, train_idx, test_idx, label)
    train_idx = test_graph[2]
    node_train = test_graph[-1][train_idx, 0].astype('int')
    label_train = torch.FloatTensor(test_graph[-1][train_idx, 1]).to(device)
    test_idx = test_graph[3]
    node_test = test_graph[-1][test_idx, 0].astype('int')
    label_test = torch.FloatTensor(test_graph[-1][test_idx, 1]).to(device)
    print("X_train: %d, X_test: %d" % (len(train_idx), len(test_idx)))
    bestauc = 0.
    bestepoch = 0
    for finetune_epoch in range(1, args.finetune_epochs+1):
        t2 = time()
        model.train()
        # test_graph = (feats, graph, train_idx, test_idx, label)
        output = model(test_graph[0], test_graph[1])
        loss, auc, ks, f1, acc = cal_metrics(output, node_train, train_idx, label_train, criterion_tune, test_graph)
        t3 = time()
        log.record("epoch[%d],TrainLoss[%.2f],AUC[%.4f],KS[%.4f],F1[%.4f,%.4f,%.4f],ACC[%.4f],time[%.1f]" % (finetune_epoch, loss.item(), auc, ks, f1[0], f1[1], f1[2], acc, t3-t2))
        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            # testing
            model.eval()
            loss, auc, ks, f1, acc = cal_metrics(output, node_test, test_idx, label_test, criterion_tune, test_graph)
            t4 = time()
            res.append(auc)
            log.record("epoch[%d],TestLoss[%.2f],AUC[%.4f],KS[%.4f],F1[%.4f,%.4f,%.4f],ACC[%.4f],time[%.1f]" % (finetune_epoch, loss.item(), auc, ks, f1[0], f1[1], f1[2], acc, t4-t3))

            if auc > bestauc:
                bestauc = auc
                bestepoch = finetune_epoch
                torch.save(model.state_dict(), args.path + "DCI_Xinye_params.pth")
    log.record("Best Epoch[%d], Best AUC[%.4f]" % (bestepoch, bestauc))
    return np.max(res)

def main():
    set_seed(42)
    device = torch.device("cpu")
    # device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    edge_index, feats, train_user_idx, test_user_idx, labels, nb_nodes = loadXinyeData()

    src, dst = torch.tensor(edge_index[:, 0]), torch.tensor(edge_index[:, 1])
    graph = dgl.graph((src, dst), num_nodes=feats.shape[0])
    graph = dgl.to_bidirected(graph)
    graph = dgl.add_self_loop(graph)
    graph.ndata['feature'] = torch.FloatTensor(feats)
    random.shuffle(train_user_idx)
    random.shuffle(test_user_idx)

    test_uid = set(test_user_idx)
    edges = pd.DataFrame(edge_index, columns=['src', 'dst'])
    edges_train = edges[~edges['src'].isin(test_uid) & ~edges['dst'].isin(test_uid)].to_numpy()
    src, dst = torch.tensor(edges_train[:, 0]), torch.tensor(edges_train[:, 1])
    graph_train = dgl.graph((src, dst), num_nodes=feats.shape[0])
    graph_train = dgl.to_bidirected(graph_train)
    graph_train = dgl.add_self_loop(graph_train)
    graph_train.ndata['feature'] = torch.FloatTensor(feats)

    # pre-training process
    model_pretrain = DCI(args.num_layers, args.num_mlp_layers, feats.shape[1], args.hidden_dim, args.neighbor_pooling_type, device).to(device)
    t0 = time()
    pretrain_file = Path(args.path + "DCI_Xinye_pretrain_params.pth")
    if not pretrain_file.exists():
        kmeans = KMeans(n_clusters=args.num_cluster, random_state=0).fit(feats)
        ss_label = kmeans.labels_
        cluster_info = [list(np.where(ss_label==i)[0]) for i in range(args.num_cluster)]
        idx = np.random.permutation(nb_nodes)
        shuf_feats = feats[idx, :]
        feats = torch.FloatTensor(feats).to(device)
        shuf_feats = torch.FloatTensor(shuf_feats).to(device)
        if args.training_scheme == 'decoupled':
            optimizer_train = optim.Adam(model_pretrain.parameters(), lr=args.lr)
            for epoch in range(1, args.epochs + 1):
                model_pretrain.train()
                t2 = time()
                pretrain_loss = 0.
                loss_pretrain = model_pretrain(feats, shuf_feats, graph, None, None, None, cluster_info, args.num_cluster)
                pretrain_loss += loss_pretrain.item()
                if optimizer_train is not None:
                    optimizer_train.zero_grad()
                    loss_pretrain.backward()         
                    optimizer_train.step()
                # re-clustering
                if epoch % args.recluster_interval == 0 and epoch < args.epochs:
                    model_pretrain.eval()
                    emb = model_pretrain.get_emb(feats, graph)
                    kmeans = KMeans(n_clusters=args.num_cluster, random_state=0).fit(emb.detach().cpu().numpy())
                    ss_label = kmeans.labels_
                    cluster_info = [list(np.where(ss_label==i)[0]) for i in range(args.num_cluster)]
                t3 = time()
                log.record("epoch[%d],PretrainLoss[%.2f],Time[%.1f]" % (epoch, pretrain_loss, t3-t2))
            
            log.record('Pre-training Down!')
        torch.save(model_pretrain.state_dict(), args.path + "DCI_Xinye_pretrain_params.pth")
    #fine-tuning process
    model_pretrain.load_state_dict(torch.load(args.path + "DCI_Xinye_pretrain_params.pth"))
    graph = [graph_train, graph]

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader_train = dgl.dataloading.NodeDataLoader(graph[0], train_user_idx, sampler, batch_size=args.batch_size, shuffle=True, drop_last=False)
    dataloader_test = dgl.dataloading.NodeDataLoader(graph[1], test_user_idx, sampler, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = Classifier(args.num_layers, args.num_mlp_layers, feats.shape[1], args.hidden_dim, args.final_dropout, args.neighbor_pooling_type, device).to(device)
    pretrained_dict = model_pretrain.state_dict()
    model_dict = model.state_dict()
    pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss(reduction='sum')

    t1 = time()
    log.record("Total Time[%.1f] min" % ((t1-t0)/60))

    bestepoch = 0
    bestauc = 0.0
    for finetune_epoch in range(1, args.finetune_epochs+1):
        t0 = time()
        model.train()
        train_loss = 0.
        results = []
        for input_nodes, output_nodes, blocks in tqdm(dataloader_train):
            optimizer.zero_grad()
            input_features = blocks[0].srcdata['feature']
            train_label = labels[output_nodes.long()].reshape(-1, 1)
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
            finetune_epoch, train_loss, results[0], results[1], results[2], results[3], results[4], results[5])
        )

        model.eval()
        test_loss = 0.
        results = []
        with torch.no_grad():
            for input_nodes, output_nodes, blocks in tqdm(dataloader_test):
                input_features = blocks[0].srcdata['feature']
                test_label = labels[output_nodes.long()].reshape(-1, 1)
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
                finetune_epoch, test_loss, results[0], results[1], results[2], results[3], results[4], results[5], t1-t0)
            )

        if results[0] > bestauc:
            bestepoch = finetune_epoch
            bestauc = results[0]
            torch.save(model.state_dict(), '/home/zqxu/MHTGNN/model_save/DCI_Xinye_params.pth')

    log.record("Best Epoch[%d] Best AUC Score[%.4f]" % (bestepoch, bestauc))


if __name__ == '__main__':
    main()
