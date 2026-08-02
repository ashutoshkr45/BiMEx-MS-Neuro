import torch
import numpy as np
import torch.nn.functional as F

import itertools
    
class Multiclass_SupConLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                base_temperature=0.07, use_cosine_similarity=False):
        super(Multiclass_SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.use_cosine_similarity = use_cosine_similarity

        
    def forward(self, zis, zjs, labels=None, mask=None):
        device = torch.device('cuda')
        
        features = torch.cat([zis.unsqueeze(1), zjs.unsqueeze(1)], dim=1)

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [batch_size, n_views, feature_dim]")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # Default to identity mask (only self-positives)
            mask = torch.eye(batch_size, dtype=torch.float32)
            mask = mask.unsqueeze(2).to(device)
            label_dim = 1
        elif labels is not None:
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(1)
            
            label_dim = labels.shape[1]
            labels = labels.contiguous().view(-1, 1, label_dim)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.permute(1,0,2)).float().to(device)
        else:
            mask = mask.float().to(device)
            label_dim = mask.shape[2]

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown contrast_mode: {self.contrast_mode}")
        
        logits = anchor_feature, contrast_feature

        if self.use_cosine_similarity:
            cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            logits = cosine_similarity(anchor_feature.unsqueeze(1), contrast_feature.unsqueeze(0))
            logits = torch.div(logits, self.temperature)
        else:
            anchor_dot_contrast = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T),
                self.temperature)
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = (anchor_dot_contrast - logits_max.detach()).unsqueeze(2)

        # mask to ignore self-contrast cases
        mask = mask.repeat(anchor_count, contrast_count, 1)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size * anchor_count).unsqueeze(2).to(device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        p_ij = mask.sum(1)
        mean_log_prob_pos_label= (mask * log_prob).sum(1) / p_ij
        mean_log_prob_pos=mean_log_prob_pos_label.sum(1)/label_dim
        
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss

class MultiLabelFocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities for multi-label classification
        probs = torch.sigmoid(inputs)
        
        # Ensure targets are on the same device as inputs
        targets = targets.to(inputs.device)
        
        # Compute focal weights
        focal_weight = torch.where(targets == 1, 1 - probs, probs) ** self.gamma

        # Compute log probabilities
        log_probs = torch.where(targets == 1, torch.log(probs + 1e-8), torch.log(1 - probs + 1e-8))

        # Apply alpha weighting
        if isinstance(self.alpha, (float, int)):
            alpha_weight = self.alpha
        elif isinstance(self.alpha, (list, torch.Tensor)):
            alpha_weight = torch.tensor(self.alpha, device=inputs.device)
            alpha_weight = alpha_weight.unsqueeze(0)  # Broadcast for batch
        else:
            raise ValueError("alpha should be a float, int, list, or tensor")

        loss = -alpha_weight * focal_weight * log_probs

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class AggrementLoss(torch.nn.Module):

    def __init__(self):
        super(AggrementLoss, self).__init__()

    def forward(self, binary_cam, class_cams):
        
        # --- Detach binary CAM to prevent gradient flow ---
        binary_cam = binary_cam.detach()

        # --- Select the Class with the Highest Probability at Each Pixel ---
        max_class_cam, _ = torch.max(class_cams, dim=1, keepdim=True)  # [batch, 1, H, W]

        # --- Compute BCE Loss ---
        loss = F.binary_cross_entropy_with_logits(max_class_cam, binary_cam)

        return loss

class SimMinLoss(torch.nn.Module):

    def __init__(self, metric='cos', reduction='mean',intra=True):
        super(SimMinLoss, self).__init__()
        self.metric = metric
        self.reduction = reduction
        self.intra=intra
        
    def forward(self, embedded_fg, embedded_bg):

        if self.metric == 'cos':
            sim = cos_simi(embedded_fg, embedded_bg, self.intra)
            loss = -torch.log(1 - sim)
        
        else:
            raise NotImplementedError

        if self.reduction == 'mean':    
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            raise ValueError("Reduction must be 'mean' or 'sum'.")

class SimMaxLoss_intraclass(torch.nn.Module):

    def __init__(self, metric='cos', alpha=0.25, reduction='mean'):

        super(SimMaxLoss_intraclass, self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, embedded_bg):
        if self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_bg)
            loss = -torch.log(sim)
            loss[loss < 0] = 0
            
            _, indices = sim.sort(descending=True, dim=2)
            _, rank = indices.sort(dim=2)
            rank = rank - 1
            
            rank_weights = torch.exp(-rank.float() * self.alpha)
            loss = loss * rank_weights

        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
            
        elif self.reduction == 'sum':
            return torch.sum(loss)

def cos_simi(embedded_fg, embedded_bg, intra=True):
    embedded_fg = F.normalize(embedded_fg, dim=2)
    embedded_bg = F.normalize(embedded_bg, dim=2)
    
    if intra:
        embedded_fg = embedded_fg.permute(1, 0, 2)
        embedded_bg = embedded_bg.permute(1, 0, 2)
        sim = torch.bmm(embedded_fg, embedded_bg.permute(0, 2, 1))
        sim = sim.permute(1, 0, 2)

    else:
        pairs = list(itertools.combinations(range(embedded_fg.size(1)), 2))
        sim = [torch.bmm(embedded_fg[:, pair[0]].unsqueeze(1), embedded_bg[:, pair[1]].unsqueeze(2)) for pair in pairs]
        sim = torch.stack(sim, dim=1)

    return torch.clamp(sim, min=0.0005, max=0.9995)
