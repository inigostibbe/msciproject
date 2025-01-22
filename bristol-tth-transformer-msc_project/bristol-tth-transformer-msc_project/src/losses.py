import torch
from torch import nn

def distance_corr(var_1,var_2,normedweight,power=1):
    """var_1: First variable to decorrelate (eg mass)
    var_2: Second variable to decorrelate (eg classifier output)
    normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
    power: Exponent used in calculating the distance correlation
    va1_1, var_2 and normedweight should all be 1D torch tensors with the same number of entries
    Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
    """
    xx = var_1.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))
    yy = var_1.repeat(len(var_1),1).view(len(var_1),len(var_1))
    amat = (xx-yy).abs()

    xx = var_2.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))
    yy = var_2.repeat(len(var_2),1).view(len(var_2),len(var_2))
    bmat = (xx-yy).abs()

    amatavg = torch.mean(amat*normedweight,dim=1)
    Amat=amat-amatavg.repeat(len(var_1),1).view(len(var_1),len(var_1))\
        -amatavg.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))\
        +torch.mean(amatavg*normedweight)

    bmatavg = torch.mean(bmat*normedweight,dim=1)
    Bmat=bmat-bmatavg.repeat(len(var_2),1).view(len(var_2),len(var_2))\
        -bmatavg.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))\
        +torch.mean(bmatavg*normedweight)

    ABavg = torch.mean(Amat*Bmat*normedweight,dim=1)
    AAavg = torch.mean(Amat*Amat*normedweight,dim=1)
    BBavg = torch.mean(Bmat*Bmat*normedweight,dim=1)

    if(power==1):
        dCorr=(torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight)))
    elif(power==2):
        dCorr=(torch.mean(ABavg*normedweight))**2/(torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))
    else:
        dCorr=((torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))))**power

    return dCorr

class BCEDecorrelatedLoss(nn.Module):
    def __init__(self,lam=0.1,weighted=True):
        super().__init__()

        self.lam = lam
        self.weighted = weighted
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        #self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self,outputs,labels,event,weights):
        if not self.weighted:
            weights = torch.ones((labels.shape[0],1)).to(outputs.device)

        if outputs.dim() == 2 and outputs.shape[1] == 1:
            outputs = outputs[:,0]
        if labels.dim() == 2 and labels.shape[1] == 1:
            labels = labels[:,0]

        bce_loss_value = self.bce_loss(outputs,labels) * weights

        disco_loss_value = distance_corr(
            outputs,
            event,
            weights * len(weights) / sum(weights)
        )
    
        return {
            'bce': bce_loss_value.mean(),
            'disco': disco_loss_value.mean(),
            'tot' : bce_loss_value.mean() + self.lam * disco_loss_value.mean(),
        }
