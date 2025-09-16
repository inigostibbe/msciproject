# Detecting Higgs Boson to Dark Matter Decays with Machine Learning

This repository contains the code for the MSci project:  
**“Detecting Higgs boson to Dark Matter decays with Machine learning algorithms”**

## Project Overview

The project applies machine learning techniques to identify events where the Higgs boson decays into Dark Matter.  
The main algorithm is inspired by the Particle Transformer paper, originally developed for b-jet classification.  
The code is primarily written in Jupyter notebooks for ease of experimentation and visualization.

## Key Features

- Uses transformer-based models adapted for particle physics data.
- Focuses on event classification for Higgs to Dark Matter decay.
- Builds on the ideas from the Particle Transformer paper ([arXiv:2202.03772](https://arxiv.org/abs/2202.03772)).

## Structure

- **pre_transformer/**: Data exploration and non-transformer models tested (XGBoost and MLP)
- **bristol-tth-transformer-msc_project/bristol-tth-transformer-msc_project/**: The source code for the transformer, requirements and the notebooks running variations of the transformer.
- **bristol-tth-transformer-msc_project/bristol-tth-transformer-msc_project/notebooks/**: Notebooks for training and testing the transformer for binary and multiclassification and regression tasks. 


## Usage

1. Clone the repository.
2. Install required Python packages (see notebook headers or requirements.txt if available).
3. Run notebooks in the `notebooks/` folder to reproduce results or explore the analysis.

## Reference

- Particle Transformer: [arXiv:2202.03772](https://arxiv.org/abs/2202.03772)

## License

See LICENSE file for details.
