# Prep for data analysis

import pandas as pd
import numpy as np


def prep(signal_data, background_data):

    import pandas as pd
    import numpy as np

    # Remove nan values
    signal_data = signal_data.dropna()
    background_data = background_data.dropna()

    # Remove regions 1-5 as they contain data not suitable for training
    signal_data = signal_data[~signal_data.region.isin([1, 2, 3, 4, 5])]
    background_data = background_data[~background_data.region.isin([1, 2, 3, 4, 5])]

    # Dealing with class imbalance
    total_background_weight = background_data['weight_nominal'].sum() # Step 1: Calculate total weight for the original background
    sampled_background = background_data.sample(len(signal_data), random_state=42) # Step 2: Randomly sample x rows from the background data
    sampled_background_weight = sampled_background['weight_nominal'].sum() # Step 3: Calculate the total weight of the sampled background (before scaling)
    scaling_factor = total_background_weight / sampled_background_weight # Step 4: Compute the scaling factor to adjust the weights
    sampled_background['weight_nominal_scaled'] = sampled_background['weight_nominal'] * scaling_factor # Step 5: Scale the weights of the sampled background rows
    signal_data['weight_nominal_scaled'] = signal_data['weight_nominal'] # Add scaled weight ot the signal data

    # Combine data + add feautures
    signal_data['target'] = 1
    sampled_background['target'] = 0
    data = pd.concat([signal_data, sampled_background])
    weights = data['weight_nominal_scaled'] # saving weights for later use

    # Creation of additional useful features found from 'investigatingjets.ipynb'
    # cleanedJet_eta_std, cleanedJet_eta_range, cleadJet_phi_std, cleanedJet_phi_range

    data['cleanedJet_eta_std'] = data['cleanedJet_eta'].apply(lambda x: np.std(x))
    data['cleanedJet_eta_range'] = data['cleanedJet_eta'].apply(lambda x: np.max(x) - np.min(x))
    data['cleanedJet_phi_std'] = data['cleanedJet_phi'].apply(lambda x: np.std(x))
    data['cleanedJet_phi_range'] = data['cleanedJet_phi'].apply(lambda x: np.max(x) - np.min(x))

    return data