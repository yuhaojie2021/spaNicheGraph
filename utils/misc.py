import os
import random

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)


def print_results(nmi_result, ari_result, ami_result, args, state):
    print(state)
    print(f"name: {args.name}")
    best_idx = int(np.argmax(nmi_result))
    print("NMI : {:.4f}".format(float(nmi_result[best_idx])))
    print("ARI : {:.4f}".format(float(ari_result[best_idx])))
    print("AMI  : {:.4f}".format(float(ami_result[best_idx])))


def eva(y_true, y_pred, epoch=0):
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")
    ari = adjusted_rand_score(y_true, y_pred)
    homogeneity = homogeneity_score(y_true, y_pred)
    completeness = completeness_score(y_true, y_pred)
    v_measure = v_measure_score(y_true, y_pred)
    print(
        "Epoch_{:3d} , nmi {:.4f} , ari {:.4f} , ami {:.4f} , "
        "homogeneity {:.4f} , completeness {:.4f} , v_measure {:.4f}".format(
            int(epoch), nmi, ari, ami, homogeneity, completeness, v_measure
        )
    )
    return nmi, ari, ami, homogeneity, completeness, v_measure


def set_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def hungarian_align_labels(y_true, y_pred):
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    true_classes = np.unique(y_true)
    pred_classes = np.unique(y_pred)
    cost = np.zeros((len(true_classes), len(pred_classes)), dtype=np.int64)
    for i, true_label in enumerate(true_classes):
        true_mask = y_true == true_label
        for j, pred_label in enumerate(pred_classes):
            pred_mask = y_pred == pred_label
            cost[i, j] = -np.sum(true_mask & pred_mask)
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {pred_classes[col]: true_classes[row] for row, col in zip(row_ind, col_ind)}
    aligned = np.asarray([mapping.get(label, label) for label in y_pred], dtype=object)
    return aligned, mapping
