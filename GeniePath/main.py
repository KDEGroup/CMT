import argparse, os, sys
sys.path.append("/home/zqxu/MHTGNN/code/")

import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import dgl
from model import GeniePath, GeniePathLazy
from Metrics import *
from Utils import Logging, set_seed
from Input import loadXinyeDataHomo
from time import time
from tqdm import tqdm

parser = argparse.ArgumentParser(description='GeniePath')
parser.add_argument("--gpu", type=int, default=-1, help="GPU Index. Default: -1, using CPU.")
parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
parser.add_argument("--batch_size", type=int, default=1024*8, help="Batch size")
parser.add_argument("--dataset", type=str, default='Xinye', help="Name of dataset")
parser.add_argument("--num_layers", type=int, default=3, help="Number of GeniePath layers")
parser.add_argument("--max_epoch", type=int, default=50, help="The max number of epochs. Default: 300")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate. Default: 0.0004")
parser.add_argument("--num_heads", type=int, default=3, help="Number of head in breadth function. Default: 1")
parser.add_argument("--residual", type=bool, default=False, help="Residual in GAT or not")
parser.add_argument("--lazy", type=bool, default=False, help="Variant GeniePath-Lazy")

log_dir = "/home/zqxu/MHTGNN/log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir, 'GeniePath.log')
log = Logging(log_path)

args = parser.parse_args()
log.record(args)


def main_with_sample(args):
    set_seed(42)
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'

    graph, n_homo_features, train_user_idx, test_user_idx, label = loadXinyeDataHomo()

    # Step 2: Create model =================================================================== #
    if args.lazy:
        model = GeniePathLazy(in_dim=n_homo_features,
                              hid_dim=args.hid_dim,
                              num_layers=args.num_layers,
                              num_heads=args.num_heads,
                              residual=args.residual)
    else:
        model = GeniePath(in_dim=n_homo_features,
                          hid_dim=args.hid_dim,
                          num_layers=args.num_layers,
                          num_heads=args.num_heads,
                          residual=args.residual)

    model = model.to(device)

    # Step 3: Create training components ===================================================== #
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader_train = dgl.dataloading.NodeDataLoader(
        graph[0], 
        train_user_idx, 
        sampler, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=False
    )
    dataloader_test = dgl.dataloading.NodeDataLoader(
        graph[1], 
        test_user_idx, 
        sampler, 
        batch_size=args.batch_size, 
        shuffle=False, 
        drop_last=False
    )
    criterion = nn.BCELoss(reduction='sum')

    t0 = time()
    bestauc = 0.
    bestepoch = 0
    for epoch in range(1, args.max_epoch+1):
        # Training and validation
        model.train()
        train_loss = 0.
        for input_nodes, output_nodes, blocks in tqdm(dataloader_train):
            optimizer.zero_grad()
            feature = blocks[0].srcdata['feature']
            logits = model(blocks, feature)
            prob = logits
            pred = torch.where(prob > 0.5, 1, 0)
            train_label = label[output_nodes.long()].reshape(-1, 1)
            loss = criterion(prob, train_label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            auc, ks, acc = calc_auc(train_label, prob.detach()), calc_ks(train_label, prob.detach()), calc_acc(train_label, pred)
            pre, recall, f1 = calc_f1(train_label, pred)
        log.record("Epoch: %d, Train Loss: %.5f, AUC: %.4f, KS: %.4f, PRE: %.4f, RECALL: %.4f F1:%.4f, ACC: %.4f" % (
            epoch, train_loss, auc, ks, pre, recall, f1, acc)
        )

        model.eval()
        test_loss = 0.
        with torch.no_grad():
            for input_nodes, output_nodes, blocks in tqdm(dataloader_test):
                feature = blocks[0].srcdata['feature']
                logits = model(blocks, feature)
                prob = logits
                pred = torch.where(prob > 0.5, 1, 0)
                test_label = label[output_nodes.long()].reshape(-1, 1)
                loss = criterion(prob, test_label)
                test_loss += loss.item()
                auc, ks, acc = calc_auc(test_label, prob.detach()), calc_ks(test_label, prob.detach()), calc_acc(test_label, pred)
                pre, recall, f1 = calc_f1(test_label, pred)
            if auc > bestauc:
                bestauc = auc
                bestepoch = epoch
                torch.save(model.state_dict(), "/home/zqxu/MHTGNN/model_save/GeniePath_Xinye_params.pth")
            log.record("Epoch: %d, Test Loss: %.5f, AUC: %.4f, KS: %.4f, PRE: %.4f, RECALL: %.4f F1:%.4f, ACC: %.4f" % (
                epoch, test_loss, auc, ks, pre, recall, f1, acc)
            )
    t1 = time()
    log.record("Total Time: %.1f min, Best Epoch: %d, Best AUC: %.4f" % ((t1-t0)/60, bestepoch, bestauc))


if __name__ == '__main__':
    main_with_sample(args)