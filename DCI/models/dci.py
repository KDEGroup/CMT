import torch
import torch.nn as nn
from layers.graphcnn import GraphCNN, GIN
from layers.readout import AvgReadout
from layers.discriminator import Discriminator
import sys

class DCI(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, device):
        super(DCI, self).__init__()
        self.device = device
        self.gin = GIN(input_dim, hidden_dim)
        # self.gin = GraphCNN(num_layers, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, device)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(hidden_dim)

    def forward(self, seq1, seq2, graph, msk, samp_bias1, samp_bias2, cluster_info, cluster_num):
        # seq1: feats, seq2: shuffle_feats
        h_1 = self.gin(seq1, graph)
        h_2 = self.gin(seq2, graph)

        loss = 0
        batch_size = 1
        criterion = nn.BCELoss(reduction='sum')
        for i in range(cluster_num):
            node_idx = cluster_info[i]

            h_1_block = torch.unsqueeze(h_1[node_idx], 0)
            c_block = self.read(h_1_block, msk)
            c_block = self.sigm(c_block)
            h_2_block = torch.unsqueeze(h_2[node_idx], 0)

            lbl_1 = torch.ones(batch_size, len(node_idx))
            lbl_2 = torch.zeros(batch_size, len(node_idx))
            lbl = torch.cat((lbl_1, lbl_2), 1).to(self.device)

            ret = self.disc(c_block, h_1_block, h_2_block, samp_bias1, samp_bias2)
            loss_tmp = criterion(ret, lbl)
            loss += loss_tmp

        return loss / cluster_num
    
    def get_emb(self, seq1, graph):
        h_1 = self.gin(seq1, graph)
        return h_1