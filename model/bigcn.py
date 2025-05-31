import copy
import torch
from torch.nn import functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv

from config import DEVICE

class GCNsBlock(torch.nn.Module):
    def __init__(self,
                 in_channels:int=32,
                 hidden_size:int=64,
                 out_channels:int=64,
                 dropout_p:float=0.2,
                 is_BU:bool=False,
                 ):
        super(GCNsBlock, self).__init__()

        self.in_proj_1 = torch.nn.Linear(1, in_channels)
        self.in_proj_2 = torch.nn.Linear(in_channels, in_channels * 4)
        self.in_proj_3 = torch.nn.Linear(in_channels * 4, in_channels)

        self.conv1 = GCNConv(in_channels, hidden_size)
        self.conv2 = GCNConv(hidden_size + in_channels, out_channels)

        self.dropout_p = dropout_p
        self.is_BU = is_BU

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index

        # Reverse edge direction (if needed)
        if self.is_BU:
            edge_index = edge_index[[1, 0], :]

        # Apply edge dropout (if needed)
        if self.training:
            mask = torch.bernoulli(torch.ones(edge_index.shape[1], dtype=torch.float) * (1 - self.dropout_p)) > 0
            edge_index = edge_index[:, mask]

        x = self.in_proj_1(x)
        x = self.in_proj_2(x)
        x = self.in_proj_3(x)

        x1=copy.copy(x)
        x = self.conv1(x, edge_index)
        x2=copy.copy(x)

        #──────────# Root Features Skip #──────────#
        root_index = data.root_index
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(DEVICE)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[root_index[num_batch]]
        x = torch.cat((x,root_extend), 1)


        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        #──────────# Root Features Skip #──────────#
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(DEVICE)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[root_index[num_batch]]
        x = torch.cat((x,root_extend), 1)


        x= scatter_mean(x, data.batch, dim=0)


        return x
    


class BiGCN(torch.nn.Module):
    def __init__(self,
                 in_channels:int=1,
                 hidden_size:int=64,
                 out_channels:int=64,
                 dropout_p:float=0.2,
                 out_dim: int=64,
                 ):
        super(BiGCN, self).__init__()
        self.TDrumorGCN = GCNsBlock(in_channels, hidden_size, out_channels, dropout_p)
        self.BUrumorGCN = GCNsBlock(in_channels, hidden_size, out_channels, dropout_p, is_BU=True)
        self.fc = torch.nn.Linear((out_channels+hidden_size) * 2, out_dim)

    def forward(self, data):
        x = torch.cat(
            (
                self.TDrumorGCN(data),
                self.BUrumorGCN(data),
            ), dim=1)
        x = self.fc(x)
        return x