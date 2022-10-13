import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl, dgl.nn.pytorch.conv

from layers.mlp import MLP

class GIN(nn.Module):
    def __init__(self, in_feats, hid_feats):
        super().__init__()
        self.conv1 = dgl.nn.pytorch.conv.GINConv(torch.nn.Linear(hid_feats, hid_feats), 'mean')
        self.conv2 = dgl.nn.pytorch.conv.GINConv(torch.nn.Linear(hid_feats, hid_feats), 'mean')
        self.embed = nn.Linear(in_feats, hid_feats)
        
    def forward(self, blocks, x):
        h = self.embed(x)
        h = F.relu(self.conv1(blocks[0], h))
        h = F.relu(self.conv2(blocks[1], h))
        return h

class GraphCNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, device):
        '''
            num_layers: number of layers in the neural networks
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            device: which device to use
        '''

        super(GraphCNN, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.neighbor_pooling_type = neighbor_pooling_type

        ###List of MLPs
        self.mlps = nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = nn.ModuleList()

        for layer in range(self.num_layers):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def next_layer(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ###pooling neighboring nodes and center nodes altogether  
        
        #If sum or average pooling
        pooled = torch.spmm(Adj_block, h)
        if self.neighbor_pooling_type == "average":
            #If average pooling
            degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
            
            pooled = pooled/degree

        #representation of neighboring and center nodes 
        pooled_rep = self.mlps[layer](pooled)

        h = self.batch_norms[layer](pooled_rep)

        #non-linearity
        h = F.relu(h, inplace=True)
        return h

    
    def forward(self, feats, adj):
        h = feats
        for layer in range(self.num_layers):
            h = self.next_layer(h, layer, Adj_block = adj)

        return h
