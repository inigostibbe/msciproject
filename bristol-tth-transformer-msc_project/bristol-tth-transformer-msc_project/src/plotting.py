import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import pearsonr
import torch


def plot_inputs_per_multiplicity(inputs, labels, mask, bins, weights=None, log=False, outdir=None,show=False):
    n_parts = inputs.size(1)
    n_vars = inputs.size(2)

    fig,axs = plt.subplots(ncols=inputs.size(-1)+1,figsize=(4*(inputs.size(-1)+1),3))
    plt.subplots_adjust(left=0.1,right=0.9,wspace=0.5)

    colors = plt.cm.rainbow(np.linspace(0, 1, n_parts))
    for i in range(n_vars):
        bins_var = np.linspace(inputs[...,i].min(),inputs[...,i].max(),bins)
        for j in range(n_parts):
            axs[i].hist(
                inputs[:,j,i][mask[:,j]],
                bins = bins_var,
                histtype = 'step',
                color = colors[j],
                weights = weights[mask[:,j]] if weights is not None else None,
            )
        axs[i].set_xlabel(f'var {i}')
        if log:
            axs[i].set_yscale('log')
    for j in range(n_parts):
        axs[-1].hist([],color=colors[j],label=f'Jet #{j}')
    axs[-1].legend(loc="center left")
    axs[-1].axis("off")
    if outdir is not None:
        plt.savefig(f"{outdir}/inputs_per_njet.png", dpi=300)
    if show:
        plt.show()
    return fig

def plot_inputs_per_label(inputs, labels, mask, bins, weights=None, log=False, outdir=None,show=False, density=False):
    n_parts = inputs.size(1)
    n_vars = inputs.size(2)

    if weights is not None:
        weights = weights.unsqueeze(-1).expand(-1,n_parts)
    fig,axs = plt.subplots(ncols=inputs.size(-1),figsize=(4*inputs.size(-1),3))
    plt.subplots_adjust(left=0.1,right=0.9,wspace=0.5)

    for i in range(n_vars):
        bins_var = np.linspace(inputs[...,i].min(),inputs[...,i].max(),bins)
        mask_bkg = (labels==0).ravel()
        mask_sig = (labels==1).ravel()
        axs[i].hist(
            inputs[mask_bkg,:,i][mask[mask_bkg]],
            bins = bins_var,
            histtype = 'step',
            color = 'r',
            label = 'Background',
            weights = weights[mask_bkg,:][mask[mask_bkg]] if weights is not None else None,
            density = density
        )
        axs[i].hist(
            inputs[mask_sig,:,i][mask[mask_sig]],
            bins = bins_var,
            histtype = 'step',
            color = 'g',
            label = 'Signal',
            weights = weights[mask_sig,:][mask[mask_sig]] if weights is not None else None,
            density = density,
        )
        axs[i].set_xlabel(f'var {i}')
        if log:
            axs[i].set_yscale('log')
        axs[i].legend()
    if outdir is not None:
        plt.savefig(f"{outdir}/inputs_per_class.png", dpi=300)
    if show:
        plt.show()
    return fig
def plot_inputs_per_label_multi(inputs, labels, mask, bins, weights=None, log=False, outdir=None, show=False, density=False, class_labels=None):
    """
    Plots histograms for each feature (n_vars) split by class labels.
    Assumes labels are already class-coded (e.g. 0, 1, 2, ...).

    Parameters:
    -----------
    inputs : torch.Tensor
        Tensor of shape [batch_size, n_parts, n_vars].
    labels : torch.Tensor
        Tensor containing class labels (integers). Expected shape is [batch_size].
    mask : torch.Tensor
        Boolean tensor of shape [batch_size, n_parts] to select valid particles.
    bins : int
        Number of bins for the histograms.
    weights : torch.Tensor, optional
        Optional per-event weights of shape [batch_size] that will be expanded to [batch_size, n_parts].
    log : bool, optional
        Whether to use a log scale for the y-axis.
    outdir : str, optional
        If provided, the figure will be saved to this directory.
    show : bool, optional
        Whether to display the plot.
    density : bool, optional
        If True, the histogram is normalized to form a probability density.
    class_labels : dict, optional
        Dictionary mapping class indices to a tuple (class_name, color).
        If None, the function auto-generates labels and colors for the unique classes.

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    batch_size, n_parts, n_vars = inputs.shape  # e.g. (batch_size, n_parts, n_vars)

    # Ensure labels is one-dimensional.
    if labels.dim() > 1:
        labels = labels.squeeze()
    
    if weights is not None:
        # If weights are provided per event, expand them to match the particle dimension.
        weights = weights.unsqueeze(-1).expand(-1, n_parts)

    fig, axs = plt.subplots(ncols=n_vars, figsize=(4 * n_vars, 3))
    plt.subplots_adjust(left=0.1, right=0.9, wspace=0.5)

    # If class_labels is not provided, build it automatically using unique classes.
    if class_labels is None:
        unique_classes = torch.unique(labels).cpu().numpy()
        unique_classes = sorted(unique_classes.tolist())
        cmap = plt.get_cmap('tab10')
        class_labels = {cl: (f"Class {cl}", cmap(i % 10)) for i, cl in enumerate(unique_classes)}

    # Iterate over each feature
    for i in range(n_vars):
        # Create bins based on the minimum and maximum of the i-th feature.
        bins_var = np.linspace(inputs[..., i].min().item(), inputs[..., i].max().item(), bins)

        # Plot each class using the provided or auto-generated labels and colors.
        for class_idx, (class_name, color) in class_labels.items():
            # Create a mask for events belonging to the current class.
            # (labels == class_idx) now has shape [batch_size]
            # We add a dimension so it becomes [batch_size, 1] and then expand it to [batch_size, n_parts].
            mask_class = (labels == class_idx).unsqueeze(1).expand(-1, n_parts)
            valid_mask = mask_class & mask  # Combine with the provided mask

            # Apply the mask to select the inputs for feature i.
            selected_inputs = inputs[:, :, i][valid_mask]

            axs[i].hist(
                selected_inputs,
                bins=bins_var,
                histtype="step",
                color=color,
                label=class_name,
                weights=weights[valid_mask] if weights is not None else None,
                density=density,
            )

        axs[i].set_xlabel(f'Feature {i}')
        if log:
            axs[i].set_yscale('log')
        axs[i].legend()

    if outdir is not None:
        plt.savefig(f"{outdir}/inputs_per_class.png", dpi=300)
    if show:
        plt.show()

    return fig



def weighted_roc_curve(y_true, y_scores, sample_weights=None):
    # If no sample weights are provided, set them to 1
    if sample_weights is None:
        sample_weights = np.ones_like(y_true)

    # Sort scores and corresponding labels and weights
    sorted_indices = np.argsort(y_scores)
    y_true = y_true[sorted_indices]
    y_scores = y_scores[sorted_indices]
    sample_weights = sample_weights[sorted_indices]

    # Initialize true positive and false positive counts
    tps = np.cumsum(y_true * sample_weights)
    fps = np.cumsum((1 - y_true) * sample_weights)

    # Total number of positive and negative samples
    total_positive = np.sum(y_true * sample_weights)
    total_negative = np.sum((1 - y_true) * sample_weights)

    # Calculate true positive rate (tpr) and false positive rate (fpr)
    tpr = tps / total_positive
    fpr = fps / total_negative

    # Remove duplicate fpr and tpr values to ensure strict increase
    unique_fpr, unique_indices = np.unique(fpr, return_index=True)
    unique_tpr = tpr[unique_indices]
    thresholds = np.concatenate([[y_scores[sorted_indices[0]] + 1], y_scores[sorted_indices], [0]])

    return unique_fpr, unique_tpr, thresholds


def plot_roc(labels, preds, outdir=None,show=False):

    # fpr, tpr, thresholds = weighted_roc_curve(labels, preds, sample_weights=sample_weights)
    # roc_auc = np.trapezoid(tpr, fpr)

    fpr, tpr, thresholds = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label='ROC curve (auc = %0.2f)' % roc_auc)
    # hep.style.use("CMS")
    # hep.cms.label("Work in Progress", data=True, lumi=138, year="Run 2", fontsize=16)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    if outdir is not None:
        plt.savefig(f"{outdir}/roc_curve.png", dpi=300)
    if show:
        plt.show()
    return fig

def plot_confusion_matrix(labels, preds, normalize='true', outdir=None, show=False):
    # convert preds to binary decisions
    preds = np.round(preds)

    # Compute confusion matrix
    cm = confusion_matrix(labels, preds, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # plot
    fig,ax = plt.subplots(1,1,figsize=(5, 5))
    disp.plot(ax=ax,values_format='.2f')
    #sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues')
    if normalize == 'all':
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
    if normalize == 'true':
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels [normed]')
    if normalize == 'pred':
        plt.xlabel('Predicted Labels [normed]')
        plt.ylabel('True Labels')
    plt.tight_layout()
    if outdir is not None:
        plt.savefig(f"{outdir}/confusion_matrix.png", dpi=300)
    if show:
        plt.show()
    return fig

def plot_score(labels, preds, bins, weights=None, log=False,outdir=None, show=False):
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(5,4))

    labels = labels.numpy().ravel()
    preds = preds.numpy().ravel()

    bins_pred = np.linspace(0,1,bins)

    if weights is None:
        weights = np.ones(len(labels))

    ax.hist(
        preds[labels==0],
        weights = weights[labels==0],
        bins = bins_pred,
        color = 'r',
        linewidth = 2,
        histtype = 'step',
        label = 'Background',
    )
    ax.hist(
        preds[labels==1],
        weights = weights[labels==1],
        bins = bins_pred,
        color = 'g',
        linewidth = 2,
        histtype = 'step',
        label = 'Signal',
    )
    ax.set_xlabel("Score")
    ax.set_ylabel('Count')
    if log:
        ax.set_yscale('log')
        log_name = "log"
    else: log_name = "lin"
    ax.legend(loc='upper center')

    plt.tight_layout()
    if outdir is not None:
        plt.savefig(f"{outdir}/score_distribution_{log_name}.png", dpi=300)
    if show:
        plt.show()
    return fig

def plot_correlation(labels, preds, event, bins, log=False, outdir=None, show=False):
    fig,axs = plt.subplots(ncols=3,nrows=1,figsize=(14,4))

    corr_all = pearsonr(preds,event).statistic[0]

    corr_bkg = pearsonr(preds[labels==0].reshape(-1,1),event[labels==0].reshape(-1,1)).statistic[0]
    corr_sig = pearsonr(preds[labels==1].reshape(-1,1),event[labels==1].reshape(-1,1)).statistic[0]

    labels = labels.numpy().ravel()
    preds = preds.numpy().ravel()
    event = event.numpy().ravel()

    bins_pred = np.linspace(0,1,bins)
    bins_event = np.linspace(event.min(),event.max(),bins)

    h = axs[0].hist2d(
        preds,
        event,
        bins = (bins_pred,bins_event),
        norm = matplotlib.colors.LogNorm() if log else None
    )
    axs[0].text(
        x = 0.2,
        y = 0.95,
        s = f'Pearson corr = {corr_all:0.5f}',
        verticalalignment = 'center',
        transform = axs[0].transAxes,
        color = 'black' if log else 'white',
    )
    plt.colorbar(h[3],ax=axs[0])
    h = axs[1].hist2d(
        preds[labels==0],
        event[labels==0],
        bins = (bins_pred,bins_event),
        norm = matplotlib.colors.LogNorm() if log else None
    )
    axs[1].text(
        x = 0.2,
        y = 0.95,
        s = f'Pearson corr = {corr_bkg:0.5f}',
        verticalalignment = 'center',
        transform = axs[1].transAxes,
        color = 'black' if log else 'white',
    )
    plt.colorbar(h[3],ax=axs[1])
    h = axs[2].hist2d(
        preds[labels==1],
        event[labels==1],
        bins = (bins_pred,bins_event),
        norm = matplotlib.colors.LogNorm() if log else None
    )
    axs[2].text(
        x = 0.2,
        y = 0.95,
        s = f'Pearson corr = {corr_sig:0.5f}',
        verticalalignment = 'center',
        transform = axs[2].transAxes,
        color = 'black' if log else 'white',
    )
    plt.colorbar(h[3],ax=axs[2])

    for ax in axs:
        ax.set_xlabel('Score')
        ax.set_ylabel('MET')
    axs[0].set_title('All events')
    axs[1].set_title('Background events')
    axs[2].set_title('Signal events')

    plt.tight_layout()
    if outdir is not None:
        plt.savefig(f"{outdir}/correlation_plot.png", dpi=300)
    if show:
        plt.show()
    return fig

