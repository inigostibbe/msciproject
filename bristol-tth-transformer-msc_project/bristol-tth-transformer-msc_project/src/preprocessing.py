import numpy as np
import pandas as pd
import mplhep as hep
import matplotlib.pyplot as plt
import awkward as ak
import torch
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter_regions(df: pd.DataFrame, regions=[0]) -> pd.DataFrame:
    if isinstance(regions, int):
        regions = [regions]
    df = df.loc[df["region"].isin(regions)]
    logging.info(f"Filtered df contains {len(df)} events from regions {regions}")
    return df
    

def load_from_hdf(filenames=None, regions=[0]) -> pd.DataFrame:

    df = pd.DataFrame()
    if filenames:
        logging.info(f"Loading data from {len(filenames)} files...")
        for file in filenames:
            logging.info(f"Reading file: {file}")
            new_df = pd.read_hdf(file)
            logging.info(f"{file} contains {len(new_df)} events")
            new_df = filter_regions(new_df, regions)
            df = pd.concat((df, new_df), axis=0)

    if df.empty:
        logging.warning("DataFrame is empty after loading files.")
    else:
        logging.info(f"DataFrame loaded successfully with {len(df)} entries.")

    return df

def load_from_parquet(filenames=None, regions=[0]) -> pd.DataFrame:

    df = pd.DataFrame()
    if filenames:
        logging.info(f"Loading data from {len(filenames)} parquet files...")
        for file in filenames:
            logging.info(f"Reading file: {file}")
            new_df = pd.read_parquet(file)
            new_df["dataset"] = file.split("/")[-1].split(".")[0]
            logging.info(f"{file} contains {len(new_df)} events")
            new_df = filter_regions(new_df, regions)
            df = pd.concat((df, new_df), axis=0)

    if df.empty:
        logging.warning("DataFrame is empty after loading files.")
    else:
        logging.info(f"DataFrame loaded successfully with {len(df)} entries.")

    return df

def create_target_labels(process: pd.Series) -> pd.Series:
    return (process == "ttH_HToInvisible_M125").astype(int)


def get_event_level(df: pd.DataFrame, features=["InputMet_pt"]) -> torch.Tensor:
    return torch.from_numpy(df[features].values)


def split_dataframe(df: pd.DataFrame, n_chunks: int):
    """Split the DataFrame into n_chunks of approximately equal size, with any remaining rows appended to the final chunk."""
    chunk_size = len(df) // n_chunks
    chunks = [df.iloc[i * chunk_size : (i + 1) * chunk_size] for i in range(n_chunks)]
    
    # Add remaining rows to the final chunk
    remaining_rows = len(df) % n_chunks
    if remaining_rows > 0:
        chunks[-1] = pd.concat([chunks[-1], df.iloc[-remaining_rows:]])
    
    return chunks


def process_chunk(chunk, target_length: int, dtype=np.float32):
    arrays = []
    masks = []
    variables = ['pt','eta','phi','mass','area','btagDeepFlavB']
    for var in variables:
        arr_padded = ak.pad_none(chunk[f"cleanedJet_{var}"], target_length, clip=True)
        mask = ~ak.is_none(arr_padded,axis=1)
        array = ak.fill_none(arr_padded,0).to_numpy()[...,None]
        arrays.append(array)
        masks.append(mask)

    x_chunk = np.concatenate(arrays, axis=-1).astype(dtype) # (chunk, jet, feature)
    #x_chunk = np.reshape(x_chunk, (x_chunk.shape[0], target_length, len(variables))) # (chunk, feature, jet)
    padding_mask_chunk = np.logical_and.reduce(masks)
    y_chunk = chunk['target'].values.astype(dtype)

    return x_chunk, y_chunk, padding_mask_chunk

def process_chunk_multi(chunk, target_length: int, dtype=np.float32): # New version for multi-class
    arrays = []
    masks = []
    variables = ['pt','eta','phi','mass','area','btagDeepFlavB']
    for var in variables:
        arr_padded = ak.pad_none(chunk[f"cleanedJet_{var}"], target_length, clip=True)
        mask = ~ak.is_none(arr_padded,axis=1)
        array = ak.fill_none(arr_padded,0).to_numpy()[...,None]
        arrays.append(array)
        masks.append(mask)

    x_chunk = np.concatenate(arrays, axis=-1).astype(dtype) # (chunk, jet, feature)
    #x_chunk = np.reshape(x_chunk, (x_chunk.shape[0], target_length, len(variables))) # (chunk, feature, jet)
    padding_mask_chunk = np.logical_and.reduce(masks)
    y_chunk = chunk[['is_ttH', 'is_ttbar', 'is_Zjets']].values.astype(dtype)

    return x_chunk, y_chunk, padding_mask_chunk


def awkward_to_inputs_parallel(df, target_length=6, n_processes=4):
    logging.info(f"Converting awkward arrays to inputs [target_length={target_length}]...")

    df_split = split_dataframe(df, n_processes)

    with mp.Pool(processes=n_processes) as pool:
        results = pool.starmap(process_chunk, [(chunk, target_length) for chunk in df_split])

    x_combined = np.concatenate([res[0] for res in results], axis=0)
    y_combined = np.concatenate([res[1] for res in results], axis=0)
    padding_mask_combined = np.concatenate([res[2] for res in results], axis=0)

    logging.info(f"Arrays padded and clipped to target length: {target_length}")

    return torch.from_numpy(x_combined), torch.from_numpy(y_combined).unsqueeze(-1), torch.from_numpy(padding_mask_combined)

def awkward_to_inputs_parallel_multi(df, target_length=6, n_processes=4): # This is the new version for multi-class
    logging.info(f"Converting awkward arrays to inputs [target_length={target_length}]...")

    df_split = split_dataframe(df, n_processes)

    with mp.Pool(processes=n_processes) as pool:
        results = pool.starmap(process_chunk_multi, [(chunk, target_length) for chunk in df_split])

    x_combined = np.concatenate([res[0] for res in results], axis=0)
    y_combined = np.concatenate([res[1] for res in results], axis=0)
    padding_mask_combined = np.concatenate([res[2] for res in results], axis=0)

    logging.info(f"Arrays padded and clipped to target length: {target_length}")

    return torch.from_numpy(x_combined), torch.from_numpy(y_combined), torch.from_numpy(padding_mask_combined)


def awkward_to_inputs(df, target_length=6):
    logging.info("Converting awkward arrays to inputs...")

    # Pad / shorten arrays to target length
    pt = ak.fill_none(ak.pad_none(df[f"cleanedJet_pt"], target_length, clip=True), 0)
    eta = ak.fill_none(ak.pad_none(df[f"cleanedJet_eta"], target_length, clip=True), 0)
    phi = ak.fill_none(ak.pad_none(df[f"cleanedJet_phi"], target_length, clip=True), 0)
    mass = ak.fill_none(ak.pad_none(df[f"cleanedJet_mass"], target_length, clip=True), 0)
    area = ak.fill_none(ak.pad_none(df[f"cleanedJet_area"], target_length, clip=True), 0)
    btag = ak.fill_none(ak.pad_none(df[f"cleanedJet_btagDeepB"], target_length, clip=True), 0)

    logging.info(f"Arrays padded and clipped to target length: {target_length}")

    # Concatenate the features
    x = np.concatenate((pt, eta, pt, phi, area, mass, btag), axis=1)
    padding_mask = (np.array(x) == 0).astype(int)
    y = df['target'].values 

    logging.info("Awkward arrays converted to numpy format.")

    return x, y, padding_mask

def normalize_inputs(x):
    logging.info("Normalizing inputs using StandardScaler...")

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the data and transform it (Z-score normalization)
    normalized_inputs = scaler.fit_transform(x)

    logging.info("Inputs normalized successfully.")

    return normalized_inputs

def remove_negative_events(df, weight_var="weight_nominal") -> pd.DataFrame:

    num_negative = df[df[weight_var]<0].shape[0]
    logging.info(f'Negative events = {num_negative} [{num_negative/df.shape[0]*100:.2f}%]...')
    df = df[df[weight_var]>=0]
    logging.info(f"Negatively weighted events removed")

    return df

def apply_reweighting_per_class(df, weight_var="weight_nominal") -> pd.DataFrame:

    logging.info(f"Applying reweighting using variable: {weight_var}")

    # Set up weight_training column
    df["weight_training"] = df[weight_var]

    # Proportions of samples
    weightings = df[["target", weight_var]].groupby("target").sum() # This adds the weights for each class, so for binary, 2 sets
    counts = df[["target", weight_var]].count() # This counts the number of events in each class

    # Reweight each process separately
    for process in df["target"].unique():
        w_factor = float(counts.iloc[0]) / float(weightings.loc[process].iloc[0]) # Does count sum / weight sum 
        logging.info(f"Reweighting process '{process}' with factor: {w_factor:.0f}")

        # Use loc to modify the dataframe in place
        df.loc[df["target"] == process, "weight_training"] *= w_factor

        # Log the sum for validation
        sum_nominal = df.loc[df["target"] == process, "weight_nominal"].sum()
        sum_training = df.loc[df["target"] == process, "weight_training"].sum()
        logging.info(f"Process '{process}' updated. Sum of 'weight_nominal': {sum_nominal:.5f}, Sum of 'weight_training': {sum_training:.0f}")

#     return df

# def apply_reweighting_per_class_multi(df, weight_var="weight_nominal") -> pd.DataFrame: #  NEW MULTI-CLASS VERSION

#     logging.info(f"Applying reweighting using variable: {weight_var}")

#     # Set up weight_training column
#     df["weight_training"] = df[weight_var]

#     # Proportions of samples
#     # weightings = df[["target", weight_var]].groupby("target").sum() # This adds the weights for each class, so for binary, 2 sets
#     # counts = df[["target", weight_var]].count() # This counts the number of events in each class=

#     # Summing weights for each class
#     weightings = df.groupby(["is_ttH", "is_ttbar", "is_Zjets"])[weight_var].sum()

#     # Sum of weights for each class (returns a DataFrame)
#     weightings = pd.Series({
#         "is_ttH": df.loc[df["is_ttH"] == 1, weight_var].sum(),
#         "is_ttbar": df.loc[df["is_ttbar"] == 1, weight_var].sum(),
#         "is_Zjets": df.loc[df["is_Zjets"] == 1, weight_var].sum()
#     })

#     counts = df[["is_ttH", "is_ttbar", "is_Zjets"]].sum()

#     print(weightings)
#     print(type(weightings))
#     print(counts)
#     print(type(counts))

#     # Reweight each process separately
#     for process in ["is_ttH", "is_ttbar", "is_Zjets"]:
#         # Compute the reweighting factor: count sum / weight sum
#         w_factor = float(counts[process]) / float(weightings[process])
#         logging.info(f"Reweighting process '{process}' with factor: {w_factor:.4f}")

#         # Apply the weight factor where the one-hot column is 1
#         df.loc[df[process] == 1, "weight_training"] *= w_factor

#         # Log the sum for validation
#         sum_nominal = df.loc[df[process] == 1, "weight_nominal"].sum()
#         sum_training = df.loc[df[process] == 1, "weight_training"].sum()
#         logging.info(f"Process '{process}' updated. Sum of 'weight_nominal': {sum_nominal:.5f}, Sum of 'weight_training': {sum_training:.0f}")

#     return df

def apply_reweighting_per_class_multi(df, weight_var="weight_nominal") -> pd.DataFrame:
    logging.info(f"Applying reweighting using variable: {weight_var}")
    
    # Set up weight_training column
    df["weight_training"] = df[weight_var]
    
    # Compute total weight per process
    weightings = pd.Series({
        "is_ttH": df.loc[df["is_ttH"] == 1, weight_var].sum(),
        "is_ttbar": df.loc[df["is_ttbar"] == 1, weight_var].sum(),
        "is_Zjets": df.loc[df["is_Zjets"] == 1, weight_var].sum()
    })
    
    # Get the event counts (one-hot encoded, so just sum the columns)
    counts = df[["is_ttH", "is_ttbar", "is_Zjets"]].sum()
    
    # Choose a fixed reference count, for example the count for 'is_ttH'
    reference_count = counts["is_ttH"]
    
    for process in ["is_ttH", "is_ttbar", "is_Zjets"]:
        # Use the fixed reference for all processes
        w_factor = float(reference_count) / float(weightings[process])
        logging.info(f"Reweighting process '{process}' with factor: {w_factor:.4f}")
    
        # Apply the weight factor where the one-hot column is 1
        df.loc[df[process] == 1, "weight_training"] *= w_factor
    
        # Log the sum for validation
        sum_nominal = df.loc[df[process] == 1, "weight_nominal"].sum()
        sum_training = df.loc[df[process] == 1, "weight_training"].sum()
        logging.info(f"Process '{process}' updated. Sum of 'weight_nominal': {sum_nominal:.5f}, Sum of 'weight_training': {sum_training:.0f}")
    
    return df


def apply_reweighting_per_process(df, weight_var="weight_nominal") -> pd.DataFrame:
    """
    Apply reweighting to the DataFrame by scaling the weights of each dataset.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data and weights.
    weight_var : str, optional (default="weight_nominal")
        The name of the column in the DataFrame that contains the weights to be used for reweighting.

    Returns:
    --------
    pandas.DataFrame
        The modified DataFrame with an updated 'weight_training' column containing the reweighted values.

    Notes:
    ------
    - This function creates a new column `weight_training` in the DataFrame by copying values from the
      specified `weight_var` column.
    - It calculates reweighting factors for each unique dataset and applies them to the `weight_training` column.
    - The reweighting is done by computing the sum of weights for each dataset and scaling them to maintain
      proportionality.
    - After reweighting, the sum of the original and reweighted values for each process is logged for validation.
    """
    logging.info(f"Applying reweighting using variable: {weight_var}")

    # Set up weight_training column
    df["weight_training"] = df[weight_var]

    # Proportions of samples
    weightings = df[["dataset", weight_var]].groupby("dataset").sum()
    counts = df[["dataset", weight_var]].count()

    # Reweight each process separately
    for process in df["dataset"].unique():
        w_factor = float(counts.iloc[0]) / float(weightings.loc[process].iloc[0])
        logging.info(f"Reweighting process '{process}' with factor: {w_factor:.0f}")

        # Use loc to modify the dataframe in place
        df.loc[df["dataset"] == process, "weight_training"] *= w_factor

        # Log the sum for validation
        sum_nominal = df.loc[df["dataset"] == process, "weight_nominal"].sum()
        sum_training = df.loc[df["dataset"] == process, "weight_training"].sum()
        logging.info(f"Process '{process}' updated. Sum of 'weight_nominal': {sum_nominal:.5f}, Sum of 'weight_training': {sum_training:.0f}")

    return df

def plot_weights(self, weight_var="weight_training", outdir="testing"):
    logging.info(f"Plotting weights for variable: {weight_var}")

    # Create the plot
    fig = plt.figure(figsize=(8,7), dpi=200)
    hep.style.use("CMS")
    hep.cms.label("Work In Progress", data=True, lumi=138, year="Run 2", fontsize=16)
    print(self.df["dataset"].unique())
    # Plot weights for each process
    for process in self.df["dataset"].unique():
        print(process)
        sumw = np.sum(self.df[f"{weight_var}"][self.df["dataset"] == process])
        logging.info(f"Process '{process}' - sum of weights ({weight_var}): {sumw:.0f}")

        plt.hist(self.df[f"{weight_var}"][self.df["dataset"] == process],
                    label=f"Sumw {process}: {sumw:.0f}", bins=100, alpha=0.3)

    # Configure plot
    plt.yscale("log")
    plt.xlabel(f"{weight_var}")
    plt.ylabel("Count")
    plt.legend(fontsize=14)
    plt.tight_layout()

    # Save and show the plot
    plot_filename = f"./{outdir}/{weight_var}.png"
    plt.savefig(plot_filename)
    logging.info(f"Plot saved to {plot_filename}")

    plt.show()

def kfold_split(df, k=2, split=False) -> np.ndarray:
    """
    Generate K-fold split masks for cross-validation.

    Parameters:
    -----------
    df : pandas.DataFrame - The input DataFrame to be split.
    k : int, optional (default=2) - The number of folds for splitting the dataset.
    split : int or bool, optional (default=False)
        If False, return all k split masks as a list.
        If an integer (0 <= split < k), return only the mask corresponding to that specific fold.

    Returns:
    --------
    split_mask : list of numpy.ndarray or numpy.ndarray
        If split is False, a list of length k is returned, where each element is a mask for one of the k-folds.
        If split is an integer, the mask for the specific fold is returned.

    Notes:
    ------
    This function is used to create K-fold split masks, where each mask indicates which part of the dataset
    is used for training or validation in a K-fold cross-validation setup.
    """

    n = len(df)
    split_mask = []
    for i in range(k):
        split_mask.append(np.arange(n) % k != i)

    if split:
        assert split < k
        return split_mask[split]

    return np.array(split_mask)

def int_delta(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1 - eta2)**2 + (phi1 - phi2)**2)

def int_kt(pt1, pt2, delta):
    return np.minimum(pt1, pt2) * delta

def int_z(pt1, pt2):
    return np.minimum(pt1, pt2) / (pt1 + pt2)

def interaction_matrix(pt, eta, phi):
    # Create mesh grids for vectorized pairwise operations
    eta1, eta2 = np.meshgrid(eta, eta, indexing='ij')
    phi1, phi2 = np.meshgrid(phi, phi, indexing='ij')
    pt1, pt2 = np.meshgrid(pt, pt, indexing='ij')

    # Calculate interaction terms
    mat_delta = int_delta(eta1, phi1, eta2, phi2)
    mat_kt = int_kt(pt1, pt2, mat_delta)
    mat_z = int_z(pt1, pt2)

    # Ensure diagonals are zero
    np.fill_diagonal(mat_delta, 0)
    np.fill_diagonal(mat_kt, 0)
    np.fill_diagonal(mat_z, 1)

    return np.stack((mat_delta, mat_kt, mat_z))


