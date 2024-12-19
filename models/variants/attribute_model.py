from torch import optim
from constants import ATTRIBUTE
from models import GCN
from models.variants import SingleBaseModel


class AttributeModel(SingleBaseModel):
    def __init__(self, args, data):
        super(AttributeModel, self).__init__(args, data)
        model = GCN(name=ATTRIBUTE,
                    nfeat=data.x.shape[1],
                    nhid=args.hidden,
                    nclass=args.num_classes,
                    layers=args.layers,
                    dropout=args.dropout,
                    enable_bn=args.enable_bn)
        edge_index = data.knn_edge_index
        
        self.model = model
        self.edge_index = edge_index
        optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.optimizer_model = optimizer_model
