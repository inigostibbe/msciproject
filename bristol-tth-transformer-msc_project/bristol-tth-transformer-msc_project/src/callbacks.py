import math
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import lightning as L
from lightning.pytorch.callbacks import Callback
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from plotting import *

class EpochEndCallback(Callback):
    def __init__(self,data,subcallbacks=[],batch_size=1024,frequency=1):
        super().__init__()
        self.frequency = frequency
        self.subcallbacks = subcallbacks

        if isinstance(data,Dataset):
            self.loader = DataLoader(data,batch_size=batch_size,shuffle=False)
        elif isinstance(data,DataLoader):
            self.loader = data
        else:
            raise RuntimeError(f'Type {type(data)} of data not understood')

    def on_validation_epoch_end(self,trainer,pl_module):
        if trainer.sanity_checking:  # optional skip
            return
        if trainer.current_epoch % self.frequency != 0:
           return

        # Obtain all tensors #
        inputs, labels, weights, mask, event = self.loader.dataset.tensors

        # Obtain predictions #
        preds = []
        for batch in self.loader:
            # automatically out batch on model device
            batch = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)
            # save after passing back to cpu #
            preds.append(pl_module(x=batch[0],padding_mask=batch[3]).cpu())
        preds = torch.cat(preds, dim=0)
        preds = F.sigmoid(preds)

        for subcallback in self.subcallbacks:
            subcallback(
                trainer = trainer,
                inputs = inputs,
                labels = labels,
                weights = weights,
                mask = mask,
                event = event,
                preds = preds,
            )


class SubCallback:
    def __init__(self,name):
        self.name = name
        self.epoch = 0

    def set_epoch(self,epoch):
        self.epoch = epoch

    def call(self,inputs,labels,weights,mask,event,preds):
        raise NotImplemented

    def __call__(self,trainer,inputs,labels,weights,mask,event,preds):
        fig = self.call(inputs,labels,weights,mask,event,preds)
        trainer.logger.experiment.log_figure(
            figure_name = f'{self.name}',
            figure = fig,
            overwrite = True,
            step = trainer.current_epoch,
        )
        plt.close(fig)

class ScoreSubCallback(SubCallback):
    def __init__(self,bins,log,**kwargs):
        self.bins = bins
        self.log = log
        super().__init__(**kwargs)

    def call(self,inputs,labels,weights,mask,event,preds):
        return plot_score(labels,preds,bins=self.bins,log=self.log)

class CorrelationSubCallback(SubCallback):
    def __init__(self,bins,log,**kwargs):
        self.bins = bins
        self.log = log
        super().__init__(**kwargs)

    def call(self,inputs,labels,weights,mask,event,preds):
        return plot_correlation(labels,preds,event,bins=self.bins,log=self.log)

class ROCSubCallback(SubCallback):
    def call(self,inputs,labels,weights,mask,event,preds):
        return plot_roc(labels,preds)

class ConfusionMatrixSubCallback(SubCallback):
    def call(self,inputs,labels,weights,mask,event,preds):
        return plot_confusion_matrix(labels,preds)




