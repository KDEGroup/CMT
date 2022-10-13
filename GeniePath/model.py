import torch as th
import torch.nn as nn
from dgl.nn.pytorch.conv import GATConv
from torch.nn import LSTM


class GeniePathConv(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_heads=1, residual=False):
        super(GeniePathConv, self).__init__()
        self.breadth_func = GATConv(in_dim, hid_dim, num_heads=num_heads, residual=residual)
        self.depth_func = LSTM(hid_dim, out_dim)

    def forward(self, blocks, x, h, c):
        x = self.breadth_func(blocks[0], x)
        x = th.mean(x, dim=1)
        x = self.breadth_func(blocks[1], x)
        x = th.mean(x, dim=1)
        x, (h, c) = self.depth_func(x.unsqueeze(0), (h, c))
        x = x[0]
        return x, (h, c)


class GeniePath(nn.Module):
    def __init__(self, in_dim, hid_dim, num_layers, num_heads, residual=False):
        super(GeniePath, self).__init__()
        self.hid_dim = hid_dim
        self.embed_dim = 32
        self.embed = nn.Linear(in_dim, self.embed_dim)
        self.linear1 = nn.Linear(self.embed_dim, hid_dim)
        self.predict = nn.Sequential(
            nn.Linear(hid_dim, 1),
            nn.Sigmoid()
        )
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(GeniePathConv(hid_dim, hid_dim, hid_dim, num_heads=num_heads, residual=residual))

    def forward(self, blocks, x):
        h = th.zeros(1, blocks[1].num_dst_nodes(), self.hid_dim).to(x.device)
        c = th.zeros(1, blocks[1].num_dst_nodes(), self.hid_dim).to(x.device)

        x = self.linear1(self.embed(x))
        feats = x
        for layer in self.layers:
            x, (h, c) = layer(blocks, feats, h, c)
        x = self.predict(x)

        return x


class GeniePathLazy(nn.Module):
    def __init__(self, in_dim, hid_dim=16, num_layers=2, num_heads=1, residual=False):
        super(GeniePathLazy, self).__init__()
        self.hid_dim = hid_dim
        self.embed_dim = 32
        self.embed = nn.Linear(in_dim, self.embed_dim)
        self.linear1 = nn.Linear(self.embed_dim, hid_dim)
        self.predict = nn.Sequential(
            nn.Linear(hid_dim, 1),
            nn.Sigmoid()
        )
        self.breaths = nn.ModuleList()
        self.depths = nn.ModuleList()
        for i in range(num_layers):
            self.breaths.append(GATConv(hid_dim, hid_dim, num_heads=num_heads, residual=residual))
            self.depths.append(LSTM(hid_dim*2, hid_dim))

    def forward(self, blocks, x):
        h = th.zeros(1, blocks[1].num_dst_nodes(), self.hid_dim).to(x.device)
        c = th.zeros(1, blocks[1].num_dst_nodes(), self.hid_dim).to(x.device)

        x = self.linear1(self.embed(x))
        h_tmps = []
        for layer in self.breaths:
            tmp = th.mean(layer(blocks[0], x), dim=1)
            tmp = th.mean(layer(blocks[1], tmp), dim=1)
            h_tmps.append(tmp)
        x = self.linear1(self.embed(blocks[1].dstdata['feature']))
        x = x.unsqueeze(0)
        for h_tmp, layer in zip(h_tmps, self.depths):
            in_cat = th.cat((h_tmp.unsqueeze(0), x), -1)
            x, (h, c) = layer(in_cat, (h, c))
        x = self.predict(x[0])

        return x