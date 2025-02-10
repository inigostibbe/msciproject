# Script is based off https://docs.ray.io/en/latest/tune/examples/pbt_guide.html and the training_multi.ipynb notebook.

import os
import sys
from datetime import datetime

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

# Ray Tune imports
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback

# Import your custom modules.
# Ensure that your PYTHONPATH or sys.path includes the location of your source files.
path_src = '../src'
if path_src not in sys.path:
    sys.path.insert(0, path_src)
from transformer import AnalysisObjectTransformer, Embedding
from losses import CrossEntropyWeightedLoss  # or BCEDecorrelatedLoss, if preferred

# Set matmul precision and determine accelerator.
torch.set_float32_matmul_precision('medium')
accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
print(f"Accelerator: {accelerator}")


def get_data_loaders(): 
    """
    Loads the dataset, performs an 80/20 train–validation split,
    and returns DataLoaders for each.
    """
    main_path = '/home/pk21271/prep_data/ttH_ttbar_Zjets_Wjets'
    x_path = os.path.join(main_path, 'X.pt')
    y_path = os.path.join(main_path, 'y.pt')
    pad_mask_path = os.path.join(main_path, 'pad_mask.pt')
    reweighting_path = os.path.join(main_path, 'reweighting.pt')
    weight_nom_path = os.path.join(main_path, 'weight_nom.pt')
    event_level_path = os.path.join(main_path, 'event_level.pt')

    # Load tensors
    X = torch.load(x_path)
    y = torch.load(y_path)
    pad_mask = torch.load(pad_mask_path)
    reweighting = torch.load(reweighting_path)
    weight_nom = torch.load(weight_nom_path)
    event_level = torch.load(event_level_path)

    # Split data into training and validation sets.
    train_X, val_X, train_y, val_y, train_weights, val_weights, train_mask, val_mask, train_event, val_event, train_nom, val_nom = train_test_split(
        X, y, reweighting, pad_mask, event_level, weight_nom, test_size=0.2, random_state=42,
    )

    # Create TensorDatasets.
    train_dataset = TensorDataset(train_X, train_y, train_weights, train_mask, train_event)
    valid_dataset = TensorDataset(val_X, val_y, val_weights, val_mask, val_event)

    batch_size = 1024 # Can make this a hyperparameter if desired.
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=11,  # Adjust as needed for your system.
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=10000,  # Larger batch size is acceptable for validation.
        shuffle=False,
        num_workers=11,
    )
    return train_loader, valid_loader


def train_transformer(config):
    """
    Ray Tune training function.
    Loads data, sets up the embedding and loss, computes input/output dimensions,
    instantiates the model with parameters from config, and trains using Lightning.
    """
    # Retrieve data loaders.
    step = 1
    train_loader, valid_loader = get_data_loaders()

    # ----------------------------
    # Set up the model components.
    # ----------------------------

    loss_function = CrossEntropyWeightedLoss(weighted = False) # The weights refer to the predetermined weights [1.1,1,1,1] in the loss function file

    embedding = Embedding(
        input_dim = train_X.shape[-1],   # Input dimension (Need to figure this out )
        embed_dims = config.get('embed_dims', 64),               # Output dimension
        normalize_input = True,          # Whether to apply batch norm before the embedding
    )

    model = AnalysisObjectTransformer(
        embedding = embedding,           # Embedding instance 
        embed_dim = embedding.dim,       # Embedding dimension
        num_heads = 8,                   # Number of heads for multihead attention (must be a divisor of embed dim) 
        output_dim = output_dim,         # Output dimension (1 : binary classification, >1 : multi classification) 
        expansion_factor = config.get('expansion', 4),      # Multipliying factor for layers in attention block (neurons = embed_dim * expansion_factor) norm is 4
        encoder_layers = config.get('enc', 4),              # Number of encoder layers (self attention on jets)
        class_layers = config.get('cla', 2),                # Number of class layers (cross attention between jets representations and class token)
        dnn_layers = config.get('dnn', 3),                  # Number of layers for DNN after the transformer
        hidden_activation = nn.GELU,     # Hidden activation in transformer and DNN
        output_activation = None,        # DNN output activation (sigmoid for binary, softmax for multiclass, None if applied in the loss)
        dropout = 0.2,                   # Dropout rate - default was 0.1
        loss_function = loss_function,   # Loss function, see above
    )

    optimizer = optim.Adam(model.parameters(), lr=config.get("lr", 0.01))

    # Checkpoint section LOOK AT THIS AND  FILL THIS IN 

    # olDE CODE BELOW

    # Determine the input dimension from the training data.
    sample_input = train_dataset.tensors[0]  # from TensorDataset: (inputs, labels, weights, mask, event)
    input_dim = sample_input.shape[-1]

    # Determine output dimension: if more than two unique classes then use the number of classes, else 1.
    train_labels = train_dataset.tensors[1]
    unique_labels = torch.unique(train_labels)
    output_dim = len(unique_labels) if len(unique_labels) > 2 else 1
    print("Output dimension:", output_dim)


    # (Ensure that your AnalysisObjectTransformer uses the learning rate accordingly.)

    # ----------------------------
    # Set up the Lightning Trainer.
    # ----------------------------
    # The TuneReportCallback will report "val_loss" at the end of each validation epoch.
    tune_report = TuneReportCallback({"val_loss": "val_loss"}, on="validation_end")
    trainer = L.Trainer(
        max_epochs=config.get("max_epochs", 10),
        accelerator=accelerator,
        devices=1,  # Use one GPU (or 1 CPU if accelerator == 'cpu').
        callbacks=[tune_report],
    )

    # Start training.
    trainer.fit(model, train_loader, valid_loader)


def main(): # Here is the main function that will be called by Ray Tune.
    print()

if __name__ == "__main__":
    main()