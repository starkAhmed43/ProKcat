import math
from math import sqrt
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch.autograd import Variable

from feature_utils import load_pickle


# NOTE: This module is a minimal copy of utility behavior used by the authors' notebook
# path in code/run_train_test.ipynb via `from train_functions import *`.
# We keep it local here to avoid depending on the DLTKcat baseline runner script.


def load_data(path, has_label, split_name):
    p = Path(path)
    compounds = np.array(load_pickle(p / f"{split_name}_features/compounds.pkl"), dtype=object)
    adjacencies = np.array(load_pickle(p / f"{split_name}_features/adjacencies.pkl"), dtype=object)
    fps = np.array(load_pickle(p / f"{split_name}_features/fps.pkl"), dtype=object)
    proteins = np.array(load_pickle(p / f"{split_name}_features/proteins.pkl"), dtype=object)
    inv_temp = np.array(load_pickle(p / f"{split_name}_features/inv_Temp.pkl"), dtype=object)
    temp = np.array(load_pickle(p / f"{split_name}_features/Temp.pkl"), dtype=object)

    if has_label:
        targets = np.array(load_pickle(p / f"{split_name}_features/log10_kcat.pkl"), dtype=object)
        return [compounds, adjacencies, fps, proteins, inv_temp, temp, targets]
    return [compounds, adjacencies, fps, proteins, inv_temp, temp]


def batch_pad(arr):
    n = max([a.shape[0] for a in arr])
    if arr[0].ndim == 1:
        new_arr = np.zeros((len(arr), n))
        new_mask = np.zeros((len(arr), n))
        for i, a in enumerate(arr):
            m = a.shape[0]
            new_arr[i, :m] = a + 1
            new_mask[i, :m] = 1
        return new_arr, new_mask

    if arr[0].ndim == 2:
        new_arr = np.zeros((len(arr), n, n))
        new_mask = np.zeros((len(arr), n, n))
        for i, a in enumerate(arr):
            m = a.shape[0]
            new_arr[i, :m, :m] = a
            new_mask[i, :m, :m] = 1
        return new_arr, new_mask

    raise ValueError("Unsupported tensor rank in batch_pad")


def batch2tensor(batch_data, has_label, device):
    # NOTE: Mirrors DLTKcat_batch2tensor behavior used by notebook training path.
    atoms_pad, atoms_mask = batch_pad(batch_data[0])
    adj_pad, _ = batch_pad(batch_data[1])

    fps = batch_data[2]
    temp_arr = np.zeros((len(fps), 1024))
    for i, a in enumerate(fps):
        temp_arr[i, :] = np.array(list(a), dtype=int)
    fps = temp_arr

    amino_pad, amino_mask = batch_pad(batch_data[3])

    atoms_pad = Variable(torch.LongTensor(atoms_pad)).to(device)
    atoms_mask = Variable(torch.FloatTensor(atoms_mask)).to(device)
    adj_pad = Variable(torch.LongTensor(adj_pad)).to(device)
    fps = Variable(torch.FloatTensor(fps)).to(device)
    amino_pad = Variable(torch.LongTensor(amino_pad)).to(device)
    amino_mask = Variable(torch.FloatTensor(amino_mask)).to(device)

    inv_temp = batch_data[4]
    temp_arr = np.zeros((len(inv_temp), 1))
    for i, a in enumerate(inv_temp):
        temp_arr[i, :] = a
    inv_temp = torch.FloatTensor(temp_arr).to(device)

    temp = batch_data[5]
    temp_arr = np.zeros((len(temp), 1))
    for i, a in enumerate(temp):
        temp_arr[i, :] = a
    temp = torch.FloatTensor(temp_arr).to(device)

    if not has_label:
        return atoms_pad, atoms_mask, adj_pad, fps, amino_pad, amino_mask, inv_temp, temp

    label = batch_data[6]
    temp_arr = np.zeros((len(label), 1))
    for i, a in enumerate(label):
        temp_arr[i, :] = a
    label = torch.FloatTensor(temp_arr).to(device)

    return atoms_pad, atoms_mask, adj_pad, fps, amino_pad, amino_mask, inv_temp, temp, label


def scores_metrics(label, pred):
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    rmse = sqrt(((label - pred) ** 2).mean(axis=0))
    r2 = r2_score(label, pred)
    pcc = np.corrcoef(label, pred)[0, 1]
    mae = mean_absolute_error(label, pred)
    return round(rmse, 6), round(r2, 6), round(pcc, 6), round(mae, 6)


def iterate_batches(data_pack, batch_size, shuffle=False):
    n = len(data_pack[0])
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)

    for i in range(math.ceil(n / batch_size)):
        sel = idx[i * batch_size : (i + 1) * batch_size]
        yield [data_pack[d][sel] for d in range(len(data_pack))]
