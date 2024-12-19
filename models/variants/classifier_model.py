import numpy as np
import wandb
from sklearn.metrics import roc_auc_score
from constants import CLASSIFIER
from metrics import eval_metric
from models import Classifier
import torch.optim as optim
import torch.nn.functional as F
from models.helpers import WandbSingleton
import os
import torch
from copy import deepcopy


class ClassifierModel:
    def __init__(self, args, data, nhid, name=CLASSIFIER):
        self.args = args
        self.data = data
        self.model = Classifier(name=name,
                                nhid=nhid,
                                nclass=args.num_classes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.best_loss = float('inf')
        self.best_tradeoff_val = float('-inf')
        self.best_epoch = 0
        self.best_sp = float('inf')
        self.best_eo = float('inf')
        self.best_state_dict = None

        self.log_dict = {}
        self.summary_dict = {}
        self.run_name = WandbSingleton().run_name
        directory = f"./saved_models/{self.run_name}"
        os.makedirs(directory, exist_ok=True)

    def load(self, run_name):
        self.model.load_state_dict(torch.load(f"saved_models/{run_name}/{self.model.name}_{self.args.dataset}.pt"))
        self.to(self.args.device)
        self.best_state_dict = deepcopy(self.model.state_dict())

    def save(self, state_dict):
        torch.save(state_dict, f"saved_models/{self.run_name}/{self.model.name}_{self.args.dataset}.pt")

    def to(self, device):
        self.model.to(device)

    def forward(self, combined_embedding):
        data = self.data
        args = self.args
        model = self.model

        model.train()

        output = model(combined_embedding)

        preds = (output.squeeze() > 0).type_as(data.y)
        bce_loss_train = F.binary_cross_entropy_with_logits(output[data.idx_train],
                                                            data.y[data.idx_train].unsqueeze(1).float().to(
                                                                args.device))

        auc_roc_train = roc_auc_score(data.y[data.idx_train].cpu().numpy(),
                                      output[data.idx_train].detach().cpu().numpy())

        return bce_loss_train, auc_roc_train

    def eval(self, combined_embedding, epoch):
        model = self.model
        model.eval()
        data = self.data
        model_name = model.name

        output = model(combined_embedding)

        bce_loss_val, auc_roc_val, parity_val, equality_val, f1_val, accuracy_val = eval_metric(
            output=output,
            labels=data.y,
            sens=data.sens,
            idx=data.idx_val,
            args=self.args)
        _, auc_roc_test, parity_test, equality_test, f1_test, accuracy_test = eval_metric(
            output=output,
            labels=data.y,
            sens=data.sens,
            idx=data.idx_test,
            args=self.args)

        self.__log_preds_ratio(output=output, idx=data.idx_val, annotation='val')
        self.__log_preds_ratio(output=output, idx=data.idx_test, annotation='test')

        tradeoff_val = self._get_tradeoff(accuracy_val, auc_roc_val, f1_val, parity_val, equality_val)

        self.log_dict.update({
            f"{model_name}_logits": wandb.Histogram((output.squeeze() > 0).type_as(data.y).cpu()),
            f"{model_name}_bce_loss_val": bce_loss_val,
            f"{model_name}_tradeoff_val": tradeoff_val,
            f"{model_name}_f1_val": f1_val,
            f"{model_name}_auc_val": auc_roc_val,
            f"{model_name}_acc_val": accuracy_val,
            f"{model_name}_sp_val": parity_val,
            f"{model_name}_eo_val": equality_val,
            f"{model_name}_f1_test": f1_test,
            f"{model_name}_auc_test": auc_roc_test,
            f"{model_name}_acc_test": accuracy_test,
            f"{model_name}_sp_test": parity_test,
            f"{model_name}_eo_test": equality_test,
        })

        if self._get_best_condition(bce_loss_val, tradeoff_val):
            self.best_loss = bce_loss_val.item()
            self.best_tradeoff_val = tradeoff_val
            self.best_epoch = epoch
            self.best_sp = parity_val
            self.best_eo = equality_val

            WandbSingleton().wandb_log_without_step_inc({f"{model_name}_best_tradeoff_val": tradeoff_val})
            if self.args.save:
                self.best_state_dict = deepcopy(self.model.state_dict())

            self.summary_dict.update({
                f'{model_name}_best_bce_loss_val': bce_loss_val,
                f'{model_name}_best_acc_val': accuracy_val,
                f'{model_name}_best_f1_val': f1_val,
                f'{model_name}_best_auc_val': auc_roc_val,
                f'{model_name}_best_sp_val': parity_val,
                f'{model_name}_best_eo_val': equality_val,
                f'{model_name}_best_tradeoff_val': tradeoff_val,
                f'{model_name}_best_acc_test': accuracy_test,
                f'{model_name}_best_f1_test': f1_test,
                f'{model_name}_best_auc_test': auc_roc_test,
                f'{model_name}_best_sp_test': parity_test,
                f'{model_name}_best_eo_test': equality_test,
            })

        return bce_loss_val, auc_roc_val

    def get_results(self, combined_embedding):
        self.model.eval()
        output = self.model(combined_embedding)
        _, auc_roc_test, parity_test, equality_test, f1_test, accuracy_test = eval_metric(
            output=output,
            labels=self.data.y,
            sens=self.data.sens,
            idx=self.data.idx_test,
            args=self.args)

        return auc_roc_test, parity_test, equality_test, f1_test, accuracy_test

    def _get_best_condition(self, loss_val, tradeoff_val):
        if self.args.wd_loss:
            return tradeoff_val > self.best_tradeoff_val
        else:
            return loss_val < self.best_loss

    def _get_tradeoff(self, accuracy, auc_roc, f1, parity, equality):
        if self.args.with_acc:
            tradeoff = accuracy + auc_roc + f1 - (parity + equality)
        else:
            tradeoff = auc_roc + f1 - (parity + equality)
        return tradeoff

    def __log_preds_ratio(self, output, idx, annotation=""):
        data = self.data

        output_preds = (output.squeeze() > 0).type_as(data.y)
        data_sens = data.sens[idx].cpu().numpy()
        data_y = data.y[idx].cpu().numpy()
        preds = output_preds[idx].cpu().numpy()

        idx_s0 = data_sens == 0
        idx_s1 = data_sens == 1
        idx_s0_y1 = np.bitwise_and(idx_s0, data_y == 1)
        idx_s1_y1 = np.bitwise_and(idx_s1, data_y == 1)

        model_name = self.model.name
        self.log_dict.update({
            f"{model_name}_pred_s0_{annotation}": sum(preds[idx_s0]),
            f'{model_name}_pred_s1_{annotation}': sum(preds[idx_s1]),
            f"{model_name}_pred_s0 div s0_{annotation}": sum(preds[idx_s0]) / sum(idx_s0),
            f'{model_name}_pred_s1 div s1_{annotation}': sum(preds[idx_s1]) / sum(idx_s1),
            f"{model_name}_pred_s0_y1_{annotation}": sum(preds[idx_s0_y1]),
            f'{model_name}_pred_s1_y1_{annotation}': sum(preds[idx_s1_y1]),
            f"{model_name}_pred_s0_y1 div s0_y1_{annotation}": sum(preds[idx_s0_y1]) / sum(idx_s0_y1),
            f'{model_name}_pred_s1_y1 div s1_y1_{annotation}': sum(preds[idx_s1_y1]) / sum(idx_s1_y1),
        })
