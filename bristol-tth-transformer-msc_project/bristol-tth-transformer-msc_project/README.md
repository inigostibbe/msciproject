# ttH Particle Transformer

This repository contains the code for developing a Transformer model for the ttH --> Hinv analysis. It aims to separate ttH signals from background processes, primarily from \( t\overline{t} \) and Drell-Yan. The model uses a transformer architecture, based on Particle Transformer, to classify events based on jet features.

## Repository Structure

- **src/**
  - `preprocessing.py`: Functions for data preprocessing.
  - `model.py`: Defines the Particle Transformer model architecture.
  - `lit_transformer.py`: PyTorch Lightning module for training the transformer.
  - `train.py`: Script for training the model, incorporating data preprocessing steps.
  - `test.py`: Script for evaluating model performance on test data.

- **hinv_ml_env.yaml**: Conda environment file listing required dependencies for running the code.

- **model_training/**: Directory where model training logs are stored.

## Getting Started

### Prerequisites

Ensure you have [Conda](https://docs.conda.io/en/latest/miniconda.html) installed. You can then set up the environment with the provided `hinv_ml_env.yaml` file:

```bash
conda env create -f hinv_ml_env.yaml
conda activate hinv_ml_env
```

### Training the Model

To train the model, execute the training script:

```bash
python src/train.py
```

Training logs will be saved in the `./model_training` directory.

### Testing the Model

After training, you can test the model’s performance with:

```bash
python src/test.py
```
