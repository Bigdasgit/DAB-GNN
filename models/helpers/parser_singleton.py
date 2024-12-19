import argparse
import torch
import yaml
from pathlib import Path


# Singleton class for parsing arguments
class ParserSingleton(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ParserSingleton, cls).__new__(cls)
            args = cls._instance.__create_args()
            cls._instance.args = args

        return cls._instance

    def __create_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--cuda', type=int, default=1, help='cuda index.')
        parser.add_argument('--no-wandb_log', action='store_true', default=False,
                            help='Disables wandb_log.')
        parser.add_argument('--wandb_sweep', action='store_true', default=False,
                            help='Disables wandb_sweep.')
        parser.add_argument('--no-wd_loss', action='store_true', default=False,
                            help='Disables wd_loss.')
        parser.add_argument('--no-dis_loss', action='store_true', default=False,
                            help='Disables dis_loss.')
        parser.add_argument('--seed', type=int, default=11, help='Random seed.')
        parser.add_argument('--epochs', type=int, default=1000,
                            help='Number of epochs to train.')
        parser.add_argument('--lr', type=float, default=0.0005,
                            help='Initial learning rate.')
        parser.add_argument('--lambda_gp', type=float, default=10,
                            help='determines the contribution of gradient penalty.')
        parser.add_argument('--dis', type=float, default=0.01,
                            help='Initial disentanglement loss weight.')
        parser.add_argument('--l_dis', type=float, default=0.01,
                            help='Initial latent disentanglement loss weight.')
        parser.add_argument('--s_lr', type=float, default=0.0005,
                            help='learning rate of structural debiasing module.')
        parser.add_argument('--a_lr', type=float, default=0.001,
                            help='learning rate of attribute debiasing module.')
        parser.add_argument('--l_lr', type=float, default=0.001,
                            help='learning rate of attribute debiasing module.')
        parser.add_argument('--c_lr', type=float, default=0.003,
                            help='learning rate of combined module.')
        parser.add_argument('--w_lr', type=float, default=0.001,
                            help='learning rate of wasserstein distance approximator.')
        parser.add_argument('--s_alpha', type=float, default=1,
                            help='learning rate of structural debiasing module.')
        parser.add_argument('--a_alpha', type=float, default=1,
                            help='learning rate of structural debiasing module.')
        parser.add_argument('--l_alpha', type=float, default=1,
                            help='learning rate of structural debiasing module.')
        parser.add_argument('--alpha', type=float, default=1,
                            help='determines the contribution of wd_loss.')
        parser.add_argument('--k', type=int, default=5,
                            help='Number of hidden units.')
        parser.add_argument('--layers', type=int, default=3,
                            help='Number of GCN module layers.')
        parser.add_argument('--weight_decay', type=float, default=1e-5,
                            help='Weight decay (L2 loss on parameters).')
        parser.add_argument('--hidden', type=int, default=16,
                            help='Number of hidden units.')
        parser.add_argument('--dropout', type=float, default=0.5,
                            help='Dropout rate (1 - keep probability).')
        parser.add_argument('--dataset', type=str, default='bail',
                            choices=['bail'])
        parser.add_argument('--model', type=str, default='non_linear',
                            choices=['non_linear', 'vanilla'])
        parser.add_argument('--save', action='store_true', default=True, help='saves model')
        parser.add_argument('--load', type=str, default=None, help='loads model')
        parser.add_argument('--without_acc', action='store_true', default=False)

        args = parser.parse_known_args()[0]

        args.wandb_log = not args.no_wandb_log
        if args.wandb_sweep:
            args.wandb_log = False
        args.wd_loss = not args.no_wd_loss
        args.dis_loss = not args.no_dis_loss
        args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
        args.wandb = args.wandb_log or args.wandb_sweep
        args.with_acc = not args.without_acc

        config_dir = Path('./configs')
        
        with (config_dir / f'{args.dataset}.yml').open('r') as file:
            print(f'Loading config from {config_dir / f"{args.dataset}.yml"}')
            config = yaml.safe_load(file)

        for key, value in config.items():
            setattr(args, key, value)

        return args
