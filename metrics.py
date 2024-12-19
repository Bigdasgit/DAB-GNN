import torch
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score
import torch.nn.functional as F


def eval_metric(output, labels, sens, idx, args):
    loss = F.binary_cross_entropy_with_logits(output[idx],
                                              labels[idx].unsqueeze(1).float().to(args.device))

    output_preds = (output.squeeze() > 0).type_as(labels)

    auc_roc = roc_auc_score(labels[idx].cpu().numpy(),
                            output[idx].detach().cpu().numpy())
    parity, equality = fair_metric(output_preds[idx].cpu().numpy(),
                                   labels[idx].cpu().numpy(),
                                   sens[idx].cpu().numpy())

    f1 = f1_score(labels[idx].cpu().numpy(),
                  output_preds[idx].cpu().numpy())
    accuracy = output_preds[idx].eq(labels[idx]).sum().item() / idx.shape[0]

    return loss, auc_roc, parity, equality, f1, accuracy


def fair_metric(pred, labels, sens):
    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)

    parity = abs(sum(pred[idx_s0]) / sum(idx_s0) - sum(pred[idx_s1]) / sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred[idx_s1_y1]) / sum(idx_s1_y1))
    return parity.item(), equality.item()


def metric_wd(feature, flag, plt_show=False):
    flag = flag.detach().cpu()
    feature = (feature / feature.norm(dim=0)).detach().cpu().numpy()
    emd_distances = []

    for i in range(feature.shape[1]):
        class_1 = feature[torch.eq(flag, 0), i]
        class_2 = feature[torch.eq(flag, 1), i]
        emd = wasserstein_distance(class_1, class_2)
        emd_distances.append(emd)

    if plt_show:
        print('Attribute bias : ')
        print("Sum of all Wasserstein distance value across feature dimensions: " + str(sum(emd_distances)))
        print(
            "Average of all Wasserstein distance value across feature dimensions: " + str(
                np.mean(np.array(emd_distances))))

        sns.distplot(np.array(emd_distances).squeeze(), rug=True, hist=True, label='EMD value distribution')
        plt.legend()
        plt.show()

        num_list1 = emd_distances.cpu().numpy()
        x = range(len(num_list1))

        plt.bar(x, height=num_list1, width=0.4, alpha=0.8, label="Wasserstein distance on reachability")
        plt.ylabel("Wasserstein distance")
        plt.legend()
        plt.show()

    return sum(emd_distances)
