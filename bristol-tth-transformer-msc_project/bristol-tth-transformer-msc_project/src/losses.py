import torch
from torch import nn
import torch.nn.functional as F


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
    
class CrossEntropyWeightedLoss(nn.Module):  # Without decorrelation
    def __init__(self, weighted=True , class_weights = [1,1,1,1,1]):

        super().__init__()
        self.weighted = weighted
        # Use reduction='none' so we can apply custom weighting before averaging.
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', weight=torch.tensor(class_weights)) # For weighting classes (needs to match no. classes)
        # self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, labels, event, weights=None): # We include the 'event' but this is used for decorrealted loss so we dont use it 

        labels = labels.to(torch.long)

        # If weighting is disabled or no weights are provided, use ones.
        if not self.weighted or weights is None:
            weights = torch.ones((labels.shape[0],), device=outputs.device)
        
        # If labels have an extra singleton dimension, squeeze it.
        if labels.dim() == 2 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        
        # Compute the per-sample cross entropy loss.
        loss = self.ce_loss(outputs, labels)

        # Multiply by weights.
        weighted_loss = loss * weights

        # Return the average loss over the batch.
        return weighted_loss.mean()
    
class CrossEntropyWeightedLoss_selective(nn.Module):  # Selective losses eg taking top 10% of loss values
    def __init__(self, weighted=True):

        super().__init__()
        self.weighted = weighted
        # Use reduction='none' so we can apply custom weighting before averaging.
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, labels, event, weights=None): # We include the 'event' but this is used for decorrealted loss so we dont use it 

        labels = labels.to(torch.long)

        # If weighting is disabled or no weights are provided, use ones.
        if not self.weighted or weights is None:
            weights = torch.ones((labels.shape[0],), device=outputs.device)
        
        # If labels have an extra singleton dimension, squeeze it.
        if labels.dim() == 2 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        
        # Compute the per-sample cross entropy loss.
        loss = self.ce_loss(outputs, labels)  # shape: [batch_size]
        # Multiply by weights.
        weighted_loss = loss * weights

        # Select top 10% of loss values
        k = int(len(weighted_loss) * 0.25)  # Select top 10% of loss values
        topk_loss, _ = torch.topk(weighted_loss, k) # Select top k loss values

        # Return the average loss over the batch.
        return topk_loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, weighted=False):
        super().__init__()

        self.weighted = weighted
        self.gamma = gamma
        
        # Handle class-specific alpha weights
        if alpha is None:
            self.alpha = None
        else:
            # Convert alpha list to tensor and ensure it sums to 1
            self.alpha = torch.tensor(alpha)
            self.alpha = self.alpha / self.alpha.sum()

    def forward(self, outputs, labels, event, weights):

        labels = labels.to(torch.long)
        # If labels have an extra singleton dimension, squeeze it.
        if labels.dim() == 2 and labels.shape[1] == 1:
            labels = labels.squeeze(1)

        # Use log_softmax for numerical stability
        log_softmax = F.log_softmax(outputs, dim=-1)
        log_pt = torch.gather(log_softmax, dim=1, index=labels.unsqueeze(1))
        pt = torch.exp(log_pt).clamp(min=1e-10, max=1.0)  # Clip probabilities

        # Calculate focal loss
        focal_weight = ((1 - pt) ** self.gamma).detach()  # Detach to prevent graph issues
        
        # Apply class-specific alpha weights if provided
        if self.alpha is not None:
            # Move alpha to same device as outputs
            self.alpha = self.alpha.to(outputs.device)
            # Get alpha weight for each sample based on its true class
            alpha_t = self.alpha[labels].unsqueeze(1)
            focal_weight = alpha_t * focal_weight
        
        FL = -focal_weight * log_pt

        if not self.weighted:
            weights = torch.ones((labels.shape[0],), device=outputs.device)

        weighted_loss = FL.squeeze() * weights
        
        return weighted_loss.mean()


class FocalLoss_binary(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, weighted=False):
        
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weighted = weighted

    def forward(self, outputs, labels, event, weights=None):

        # Convert labels to float and squeeze if necessary.
        labels = labels.to(torch.float)
        if labels.dim() == 2 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        
        # If outputs come as (batch,1), squeeze to (batch,)
        if outputs.dim() == 2 and outputs.shape[1] == 1:
            outputs = outputs.squeeze(1)
        
        # Compute probabilities using sigmoid.
        probs = torch.sigmoid(outputs)
        # For each sample, p_t is the probability of the true class.
        pt = torch.where(labels == 1, probs, 1 - probs)
        pt = pt.clamp(min=1e-10, max=1.0)  # Avoid log(0)

        # Compute the focal modulation factor. (Often people do not detach, but we follow your original.)
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha weighting if provided.
        if self.alpha is not None:
            # For positive samples use alpha; for negatives use (1 - alpha)
            alpha_t = torch.where(labels == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_t * focal_weight

        # Compute binary cross entropy components.
        loss = - (labels * torch.log(pt) + (1 - labels) * torch.log(1 - pt))
        loss = focal_weight * loss

        # If per-sample weights are not provided or not used, set them to ones.
        if (not self.weighted) or (weights is None):
            weights = torch.ones_like(labels, device=outputs.device, dtype=outputs.dtype)
        
        weighted_loss = loss * weights

        return weighted_loss.mean()
