from constants import VANILLA
from models.variants import SingleBaseModel
from torch import optim

class VanillaModel(SingleBaseModel):
    def __init__(self, args, data):
        super(VanillaModel, self).__init__(args, data)

        model = GCN(nfeat=data.x.shape[1],
                    nhid=args.hidden,
                    nclass=args.num_classes,
                    dropout=args.dropout)

        self.model = model
        self.edge_index = data.edge_index
        optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.optimizer_model = optimizer_model

    def _get_best_condition(self, loss_val, tradeoff_val):
        return loss_val < self.best_loss

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.name = VANILLA
        self.body = GCN_Body(nfeat,nhid,dropout)
        self.fc = nn.Linear(nhid, nclass)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        h = self.body(x, edge_index)
        x = self.fc(h)
        return x, h


class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_Body, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        return x
