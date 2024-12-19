from torch import optim
from constants import STRUCTURAL
from models import GCN_free_embedding
from models.variants import SingleBaseModel


class StructuralModel(SingleBaseModel):
    def __init__(self, args, data):
        super(StructuralModel, self).__init__(args, data)
        model = GCN_free_embedding(name=STRUCTURAL,
                                   nsamples=data.x.shape[0],
                                   nfeat=data.x.shape[1],
                                   nhid=args.hidden,
                                   nclass=args.num_classes,
                                   layers=args.layers,
                                   dropout=args.dropout,
                                   enable_bn=args.enable_bn)

        self.model = model
        self.edge_index = data.edge_index
        optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.optimizer_model = optimizer_model
