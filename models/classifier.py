import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, nhid, nclass, name=""):
        super(Classifier, self).__init__()
        self.name = name
        self.lin1 = nn.Linear(nhid, nhid // 2)
        self.lin2 = nn.Linear(nhid // 2, nhid // 4)
        self.lin3 = nn.Linear(nhid // 4, nclass)

    def forward(self, h, edge_index=None):
        h = self.lin1(h)
        h = self.lin2(h)
        h = self.lin3(h)

        return h