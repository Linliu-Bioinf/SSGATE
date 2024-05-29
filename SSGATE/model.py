#!/bin/env python3
import torch
import torch.nn.functional as F
import torch.nn as nn
from .GATConv import GATConv


class ssmiDGATE(torch.nn.Module):
    def __init__(self, in_dims1, hidden_dims1, in_dims2, hidden_dims2, out_dims, dropout=0.0):
        super(ssmiDGATE, self).__init__()

        dim_concat = hidden_dims1 + hidden_dims2

        self.conv1_x = GATConv(in_dims1, hidden_dims1, heads=1, concat=False, dropout=0, add_self_loops=False, bias=False)
        self.conv2_x = GATConv(hidden_dims1, hidden_dims1, heads=1, concat=False, dropout=0, add_self_loops=False, bias=False)
        self.conv3_x = GATConv(out_dims, hidden_dims1, heads=1, concat=False, dropout=0, add_self_loops=False, bias=False)
        self.conv4_x = GATConv(hidden_dims1, in_dims1, heads=1, concat=False, dropout=0, add_self_loops=False, bias=False)
        self.connectlayer = nn.Linear(dim_concat, out_dims)
        self.detach_x = nn.Linear(out_dims, hidden_dims1)
        self.detach_y = nn.Linear(out_dims, hidden_dims2)

        self.conv1_y = GATConv(in_dims2, hidden_dims2, heads=1, concat=False, dropout=0, add_self_loops=False, bias=False)
        self.conv2_y = GATConv(hidden_dims2, hidden_dims2, heads=1, concat=False, dropout=0, add_self_loops=False, bias=False)
        self.conv3_y = GATConv(out_dims, hidden_dims2, heads=1, concat=False, dropout=0, add_self_loops=False, bias=False)
        self.conv4_y = GATConv(hidden_dims2, in_dims2, heads=1, concat=False, dropout=0, add_self_loops=False, bias=False)

    def forward(self, x, y, edge_x, edge_y):
        h_1_x = F.relu(self.conv1_x(x, edge_x))
        h_x = F.tanh(self.conv2_x(h_1_x, edge_x, attention = False))
        h_1_y = F.relu(self.conv1_y(y, edge_y))
        h_y = F.tanh(self.conv2_y(h_1_y, edge_y, attention = False))
        h = torch.concat([h_x, h_y], 1)

        z = self.connectlayer(h)
        z_x = self.detach_x(z)
        z_y = self.detach_y(z)

        self.conv3_x.lin_src.data = self.conv2_x.lin_src.transpose(0, 1)
        self.conv3_x.lin_dst.data = self.conv2_x.lin_dst.transpose(0, 1)
        self.conv4_x.lin_src.data = self.conv1_x.lin_src.transpose(0, 1)
        self.conv4_x.lin_dst.data = self.conv1_x.lin_dst.transpose(0, 1)

        self.conv3_y.lin_src.data = self.conv2_y.lin_src.transpose(0, 1)
        self.conv3_y.lin_dst.data = self.conv2_y.lin_dst.transpose(0, 1)
        self.conv4_y.lin_src.data = self.conv1_y.lin_src.transpose(0, 1)
        self.conv4_y.lin_dst.data = self.conv1_y.lin_dst.transpose(0, 1)
        
        h3_x = F.relu(self.conv3_x(z_x, edge_x, attention=True, tied_attention=self.conv1_x.attentions))
        encode_x = self.conv4_x(h3_x, edge_x, attention=False)

        h3_y = F.relu(self.conv3_y(z_y, edge_y, attention=True, tied_attention=self.conv1_y.attentions))
        encode_y = self.conv4_y(h3_y, edge_y, attention=False)

        return h_x, h_y, z, z_x, z_y, encode_x, encode_y
