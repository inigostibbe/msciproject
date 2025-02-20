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
# path = "/cephfs/dice/projects/CMS/Hinv/ml_datasets_ul/UL{year}_ml_inputs/{dataset}.parquet"

path = '/cephfs/dice/projects/CMS/Hinv/ml_datasets_ul_241111/ml_inputs_UL2018/{dataset}.parquet' # This is the newer datasets

datasets = [
    # Signal 0
    'ttH_HToInvisible_M125', 

    # ttbar processes 1
    'TTToSemiLeptonic',
    'TTTo2L2Nu',
    'TTToHadronic',

    # Z+jets processes 2
    'ZJetsToNuNu_HT-100To200',
    'ZJetsToNuNu_HT-200To400',
    'ZJetsToNuNu_HT-400To600',
    'ZJetsToNuNu_HT-600To800',
    'ZJetsToNuNu_HT-800To1200',
    'ZJetsToNuNu_HT-1200To2500',
    'ZJetsToNuNu_HT-2500ToInf',

    # W+jets processes 2
    'WJetsToLNu_HT-70To100',
    'WJetsToLNu_HT-100To200',
    'WJetsToLNu_HT-200To400',
    'WJetsToLNu_HT-400To600',
    'WJetsToLNu_HT-600To800',
    'WJetsToLNu_HT-800To1200',
    'WJetsToLNu_HT-1200To2500',
    'WJetsToLNu_HT-2500ToInf'
]

years = ['2018']

files = [
    path.format(year=year, dataset=dataset)
    for dataset in datasets
    for year in years
]

## Data preprocessing ##
df = load_from_parquet(files, regions = [0,6]) # Including region 
df = remove_negative_events(df)

# df["target"] = create_target_labels(df["dataset"])# NEED TO CHANGE THIS FOR MULTICLASS
# print(len(df[df['target'] == 0]))
# print(len(df[df['target'] == 1]))

# ...existing code...

# Define class mappings
class_mappings = {
    'ttH_HToInvisible_M125': 0,  # Signal
    'TTTo': 1,                   # ttbar
    'ZJetsToNuNu': 2,           # Z+jets
    'WJetsToLNu': 2,            # W+jets
}

# Initialize class column
df['class'] = -1

# Assign classes based on dataset patterns
for pattern, class_label in class_mappings.items():
    df.loc[df["dataset"].str.contains(pattern), "class"] = class_label

# Verify all datasets are assigned
unassigned = df[df['class'] == -1]
if len(unassigned) > 0:
    print("Warning: Some datasets not assigned a class:")
    print(unassigned['dataset'].unique())

# Print class distribution
print("\nClass distribution:")
print(df.groupby('class')['dataset'].nunique())
print("\nSamples per class:")
print(df['class'].value_counts())

apply_reweighting_per_class_multi(df)
reweighting = torch.Tensor(df['weight_training'].values)
weight_nom = torch.Tensor(df['weight_nominal'].values)

# df["target"] = create_target_labels(df["dataset"]) 

X, y, pad_mask = awkward_to_inputs_parallel_multi(df, n_processes=8, target_length=10)    

event_level = get_event_level(df) # This is already inputMET_pt

phi = torch.tensor(df['InputMet_phi'].values)

# Save the target varible for the regressor which is GenMET_pt
target_reg = torch.tensor(df['GenMET_pt'].values)


# Now save X,y, pad_mask

path = '/home/pk21271/prep_data/ttH_ttbar_jets_regressor'

torch.save(X, os.path.join(path, 'X.pt'))
torch.save(y, os.path.join(path, 'y.pt'))
torch.save(pad_mask, os.path.join(path,'pad_mask.pt'))
torch.save(reweighting, os.path.join(path,'reweighting.pt'))
torch.save(weight_nom, os.path.join(path,'weight_nom.pt'))
torch.save(event_level, os.path.join(path,'event_level.pt')) # This is already inputMET_pt
torch.save(phi, os.path.join(path,'phi.pt'))
torch.save(target_reg, os.path.join(path,'target_reg.pt'))