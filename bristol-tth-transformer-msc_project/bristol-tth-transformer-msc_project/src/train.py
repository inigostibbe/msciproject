# Autoreload 
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

# Lightning #
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CometLogger

# Bacis libraries #
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import yaml
from sklearn.model_selection import train_test_split

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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


torch.set_float32_matmul_precision('medium')
accelerator = 'gpu' if torch.cuda.is_available() else "cpu"
print (f"Accelerator : {accelerator}")

# Make output directory #
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = f"./model_training/AOTransformer_{current_time}"
os.makedirs(outdir, exist_ok=False)

# Get training config
with open("training_config.yaml", "r") as f:
    training_config = yaml.load(f, Loader=yaml.SafeLoader)

## Specify dataset files to run over ##
path = "/cephfs/dice/projects/CMS/Hinv/ml_datasets_ul/UL{year}_ml_inputs/{dataset}.parquet"
years = training_config["years"]
datasets = training_config["datasets"]
files = []
for y in years:
    for d in datasets:
        files.append(path.format(year=y, dataset=d))

## Data preprocessing ##
df = load_from_parquet(files)
df = remove_negative_events(df)
df["target"] = create_target_labels(df["dataset"])
df = apply_reweighting_per_class(df)
reweighting = torch.Tensor(df['weight_training'].values)

X, y, pad_mask = awkward_to_inputs_parallel(df, n_processes=8, target_length=training_config["sequence_length"])
print(X.shape)
#X, y, pad_mask = awkward_to_inputs(df, target_length=6)

event_level = get_event_level(df)
# split_masks = kfold_split(df, k=2)      # returns a list with a mask for each fold, so if only training one select it in line below
# split = split_masks[0]

## Create training datasets ##
train_X, val_X, train_y, val_y, train_weights, val_weights, train_mask, val_mask, train_event, val_event = train_test_split(
    X, 
    y, 
    reweighting, 
    pad_mask, 
    event_level, 
    test_size=0.2,  
    random_state=42,
)

train_dataset = TensorDataset(train_X, train_y, train_weights, train_mask, train_event)
valid_dataset = TensorDataset(val_X, val_y, val_weights, val_mask, val_event)

## Create loaders ##
def custom_collate(batch):
    inputs, labels, weights, mask, event = zip(*batch)
    return torch.stack(inputs), torch.stack(labels), torch.stack(weights), torch.stack(mask), torch.stack(event)

batch_size = training_config["training"]["batch_size"]

train_loader = DataLoader(
    dataset = train_dataset, 
    batch_size = batch_size, 
    shuffle = True, 
    #collate_fn = custom_collate, 
    num_workers = 127,
)
valid_loader = DataLoader(
    dataset = valid_dataset, 
    batch_size = 10000, # can use larger batches for the GPU 
    shuffle = False, 
    #collate_fn = custom_collate, 
    num_workers = 127,
)

fig = plot_inputs_per_multiplicity(X,y,pad_mask,bins=100,log=True,show=True)
fig = plot_inputs_per_label(X,y,pad_mask,bins=100,log=True,show=True)
fig = plot_inputs_per_label(X,y,pad_mask,bins=100,weights=reweighting,log=True,show=True)

# Define model #
loss_config = training_config["loss_fn"]
loss_function = BCEDecorrelatedLoss(lam = loss_config["lam"], weighted=loss_config["weighted"])

embedding = Embedding(
    input_dim = train_X.shape[-1],
    embed_dims = training_config["embedding"]["embed_dims"],
    normalize_input = training_config["embedding"]["normalize_input"],
)

model_config = training_config["model"]
model = AnalysisObjectTransformer(
    embedding = embedding,
    embed_dim = embedding.dim,
    output_dim = model_config["output_dim"],
    expansion_factor = model_config["expansion_factor"],
    encoder_layers = model_config["encoder_layers"],
    class_layers = model_config["class_layers"],
    dnn_layers = model_config["dnn_layers"],
    num_heads = model_config["num_heads"],
    hidden_activation = nn.GELU, 
    output_activation = None, #nn.Sigmoid,
    dropout = model_config["dropout"],
    loss_function = loss_function,
)
print (model)
# Quick benchmark test the model #
batch = next(iter(train_loader))

inputs, labels, weights, mask, event = batch
print (inputs.dtype,labels.dtype,weights.dtype,mask.dtype,event.dtype)

outputs = model(inputs,padding_mask=mask)
print ('outputs',outputs.shape)
loss_values = loss_function(outputs,labels,event,weights)
print ('losses',loss_values)

##### Parameters #####
epochs = training_config["training"]["epochs"]
lr = training_config["training"]["lr"]

steps_per_epoch_train = math.ceil(len(train_dataset)/train_loader.batch_size)
steps_per_epoch_valid = math.ceil(len(valid_dataset)/valid_loader.batch_size)

print (f'Training   : Batch size = {train_loader.batch_size} => {steps_per_epoch_train} steps per epoch')
print (f'Validation : Batch size = {valid_loader.batch_size} => {steps_per_epoch_valid} steps per epoch')

# Setup optimizer #
optimizer = optim.Adam(model.parameters(), lr=lr)
model.set_optimizer(optimizer)

# Setup scheduler #
schedule_config = training_config["lr_sched"]
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
   optimizer = optimizer,
   mode = schedule_config["mode"], 
   factor = schedule_config["factor"], 
   patience = schedule_config["patience"], 
   threshold = schedule_config["threshold"], 
   threshold_mode = schedule_config["threshold_mode"], 
   cooldown = schedule_config["cooldown"], 
   min_lr = schedule_config["min_lr"]
)
model.set_scheduler_config(
    {
        'scheduler' : scheduler,
        'interval' : 'step' if isinstance(scheduler,optim.lr_scheduler.OneCycleLR) else 'epoch',
        'frequency' : 1,
        'monitor' : 'val/loss_tot',
        'strict' : True,
        'name' : 'lr',
    }
)

# Setup callbacks #
checkpoint_callback = ModelCheckpoint(
    dirpath=outdir,  # Directory where checkpoints will be saved
    filename="best-checkpoint",  # Checkpoint filename
    monitor="val/loss_tot",  # Monitor validation loss
    mode="min",  # Save the best model with the minimum validation loss
    save_top_k=1  # Save only the best model
)

early_stopping = EarlyStopping(
    monitor='val/loss_tot',  # Metric to monitor
    patience=training_config["early_stopping"]["patience"],          # Number of epochs with no improvement after which training will be stopped
    verbose=True,
    mode='min'           # 'min' for loss, 'max' for accuracy
)

log_bar = L.pytorch.callbacks.TQDMProgressBar(refresh_rate=steps_per_epoch_train//100)

plots_callback = EpochEndCallback(
    data = valid_loader,
    frequency = 1,
    subcallbacks = [
        ScoreSubCallback(name='score',bins=100,log=True),
        CorrelationSubCallback(name='corr',bins=100,log=True),
        ROCSubCallback(name='ROC'),
        ConfusionMatrixSubCallback(name='CM'),
    ]
)

## Logger ##
logger = CometLogger(
    api_key = os.environ.get("COMET_API_KEY"),
    project_name = "AnalysisObjectTransformer",
    experiment_name = "Setup",
    save_dir = "./comet_logs",  # Specify where to save Comet logs if offline
    offline = False  # Set to True for offline mode
)

## Trainer ##
trainer = L.Trainer(
    default_root_dir = outdir,
    accelerator = accelerator,
    devices = [1],
    max_epochs = epochs,  # Specify the number of epochs
    log_every_n_steps = steps_per_epoch_train,
    check_val_every_n_epoch = 1,  # Check validation every n epochs
    callbacks = [
        checkpoint_callback, 
        early_stopping,
        log_bar,
        plots_callback,
    ],
    logger = logger,
    # limit_train_batches = 10,
    # limit_val_batches = 1,
    # limit_test_batches = 1,
)

trainer.fit(
    model = model, 
    train_dataloaders = train_loader, 
    val_dataloaders = valid_loader,
)
trainer.save_checkpoint(f"{outdir}/model.pt")

## Testing model performance on validation set ##
preds = trainer.predict(model=model, dataloaders=valid_loader)
preds = torch.cat(preds, dim=0)

inputs, labels, weights, mask, event = valid_dataset.tensors

df_preds = pd.DataFrame({
    "preds": preds.numpy().flatten(),
    "labels": labels.numpy().flatten(),
    "weights": weights.numpy().flatten(),
    "event": event.numpy().flatten()
})
df_preds.to_csv(f"{outdir}/inference_information.csv")

with open(f"{outdir}/training_config.yaml", "w") as f:
    yaml.safe_dump(training_config, f)

fig = plot_score(labels,preds,outdir=outdir,bins=100,log=False)
fig = plot_score(labels,preds,weights=weights,outdir=outdir,bins=100,log=True)
fig = plot_confusion_matrix(labels, preds, outdir=outdir,show=True)
fig = plot_correlation(labels,preds,event,outdir=outdir,log=True,bins=40)
fig = plot_roc(labels, preds, outdir=outdir,show=True)
# fig = plot_roc(labels, preds, outdir=outdir,show=True, sample_weights=weights)

print("Done")