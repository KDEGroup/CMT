import sys
sys.path.append("/home/zqxu/MHTGNN/code/DCI")

import torch.nn as nn
from layers.graphcnn import GraphCNN, GIN
from layers.mlp import MLP
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, final_dropout, neighbor_pooling_type, device):
        super(Classifier, self).__init__()
        self.gin = GIN(input_dim, hidden_dim)
        # self.gin = GraphCNN(num_layers, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, device)
        self.predict = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.final_dropout = final_dropout
        
    def forward(self, blocks, x):
        h_1 = self.gin(blocks, x)
        # score_final_layer = F.dropout(h_1, self.final_dropout, training = self.training)
        return self.predict(h_1)