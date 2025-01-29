# Bacis libraries #
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import yaml
from sklearn.model_selection import train_test_split

# Pytorch #
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from datetime import datetime

# Personal scripts #
path_src = './src'
if path_src not in sys.path:
    sys.path.insert(0,path_src)
from preprocessing import *
from callbacks import *
from transformer import AnalysisObjectTransformer, Embedding
from losses import BCEDecorrelatedLoss
from plotting import plot_roc, plot_confusion_matrix

## Specify dataset files to run over ##
path = "/cephfs/dice/projects/CMS/Hinv/ml_datasets_ul/UL{year}_ml_inputs/{dataset}.parquet"

datasets = [
    'ttH_HToInvisible_M125',
    'TTToSemiLeptonic',
]
years = ['2018']

files = [
    path.format(year=year, dataset=dataset)
    for dataset in datasets
    for year in years
]

## Data preprocessing ##
df = load_from_parquet(files)
df = remove_negative_events(df)
df["target"] = create_target_labels(df["dataset"])
apply_reweighting_per_class(df)
reweighting = torch.Tensor(df['weight_training'].values)
weight_nom = torch.Tensor(df['weight_nominal'].values)
df["target"] = create_target_labels(df["dataset"])

X, y, pad_mask = awkward_to_inputs_parallel(df, n_processes=8, target_length=10)

event_level = get_event_level(df)

# Now save X,y, pad_mask

path = '/home/pk21271/prep_data/ttH_ttSL'

torch.save(X, os.path.join(path, 'X.pt'))
torch.save(y, os.path.join(path, 'y.pt'))
torch.save(pad_mask, os.path.join(path,'pad_mask.pt'))
torch.save(reweighting, os.path.join(path,'reweighting.pt'))
torch.save(weight_nom, os.path.join(path,'weight_nom.pt'))
torch.save(event_level, os.path.join(path,'event_level.pt'))