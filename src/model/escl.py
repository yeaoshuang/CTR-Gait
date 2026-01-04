import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiCenterESCL(nn.Module):
    """
    Multi-Center Emotion-Semantic Contrastive Loss (MultiCenterESCL)
    Each emotion has K VA centers, samples are softly assigned before participating in ESCL
    """
    def __init__(self, num_class, K=2, gamma=0.4, temperature=0.1, 
                 scale_weights=[1.0, 1.0], confidence_threshold=0.8, 
                 kl_weight=0.01, va_lr_factor=1.0, va_weight_decay=0.0, 
                 va_constraint=True, va_constraint_range=1.2):
        super().__init__()
        self.num_class = num_class
        self.K = K
        self.gamma = gamma
        self.temperature = temperature
        self.kl_weight = kl_weight
        self.confidence_threshold = confidence_threshold
        self.scale_weights = scale_weights
        self.va_lr_factor = va_lr_factor
        self.va_weight_decay = va_weight_decay
        self.va_constraint = va_constraint
        self.va_constraint_range = va_constraint_range

        # Initialize: first give each class a "low arousal" + a "high arousal" center
        initial = torch.zeros(num_class, K, 2)
        for c in range(num_class):
            # valence fixed, arousal separated
            base_v = 0.5 if c == 3 else (0.8 if c == 0 else 0.2)  # Simplified prior
            initial[c, :, 0] = base_v
            initial[c, :, 1] = torch.linspace(0.2, 0.8, K)
        self.va_centers = nn.Parameter(initial)  # (C,K,2)

        # Lightweight assignment network: features -> K dimensional logits
        self.assignment_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (N,C,T,V)->(N,C,1,1)
            nn.Conv2d(256, K, 1),     # Output (N,K,1,1)
            nn.Flatten(start_dim=1)   # (N,K)
        )

    def forward(self, features, labels):
        """
        features: (N,C,T,V)  labels: (N,)
        """
        # Handle possible 5D input (N, C, T, V, M)
        if len(features.shape) == 5:
            N, C, T, V, M = features.shape
            # Merge M dimension into V dimension
            features = features.permute(0, 1, 2, 4, 3).contiguous().view(N, C, T, V*M)
        
        N, C, T, V = features.shape
        
        # 1. Predict assignment logits (N,K)
        logits_k = self.assignment_net(features)  # (N,K)
        # Gumbel-Softmax soft assignment (annealing during training, straight-through during inference)
        alpha = 1.0 if self.training else 0.1  # Annealing temperature
        prob_k = F.gumbel_softmax(logits_k, tau=alpha, dim=1)  # (N,K)

        # 2. Expected VA coordinates
        va_centers_batch = self.va_centers[labels]  # (N,K,2)
        va_batch = torch.einsum('nk,nkd->nd', prob_k, va_centers_batch)  # (N,2)

        # 3. Calculate ESCL weight matrix
        dist = torch.cdist(va_batch, va_batch, p=2)  # (N,N)
        w_neg = torch.exp(-dist / self.gamma)

        # 4. Original ESCL contrastive loss (completely consistent with before)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        pos_mask = mask - torch.eye(N, device=features.device)
        neg_mask = 1 - mask

        feats = features.reshape(N, -1)
        feats = F.normalize(feats, dim=1)
        sim = torch.matmul(feats, feats.T) / self.temperature
        exp_sim = torch.exp(sim)

        pos = torch.sum(exp_sim * pos_mask, dim=1)
        neg = torch.sum(exp_sim * neg_mask * w_neg, dim=1) + 1e-8
        confidence = (pos + 1e-8) / (pos + neg)
        high_conf_mask = (confidence >= self.confidence_threshold).float()
        loss = -torch.log((pos + 1e-8) / (pos + neg))
        filtered_loss = loss * high_conf_mask
        if high_conf_mask.sum() == 0:
            escl_loss = torch.tensor(0.0, device=features.device, requires_grad=True)
        else:
            escl_loss = torch.mean(filtered_loss[high_conf_mask.bool()])

        # 5. Sparse regularization: encourage samples to concentrate on one center, prevent uniform distribution
        prior = torch.ones_like(prob_k) / self.K
        kl = F.kl_div(torch.log(prob_k + 1e-8), prior, reduction='batchmean')
        
        # 6. Add VA coordinate regularization loss
        va_reg_loss = 0.0
        if self.va_weight_decay > 0:
            va_reg_loss = self.va_weight_decay * torch.sum(self.va_centers ** 2)
        
        total_loss = escl_loss + self.kl_weight * kl + va_reg_loss

        # 7. Return compatible with old interface
        time_loss = total_loss
        spatial_loss = total_loss  # Simplified example, can calculate spatial in original way
        return (self.scale_weights[0] * time_loss + 
                self.scale_weights[1] * spatial_loss), \
               time_loss, spatial_loss
    
    def get_va_coordinates(self):
        """Get currently learned VA coordinates (return all centers)"""
        return self.va_centers.detach().cpu().numpy()
    
    def get_va_parameters(self):
        """Get VA coordinate parameters for setting different learning rates"""
        return [self.va_centers]

class ESCL(nn.Module):
    """
    Emotion-Semantic Contrastive Loss (ESCL)
    Uses learnable Valence-Arousal (VA) 2D coordinates as soft labels to guide temperature scaling,
    allowing semantically similar emotions to naturally approach each other in feature space.
    """
    def __init__(self, num_classes=4, va_dict=None, gamma=0.4, temperature=0.1, scale_weights=[1.0, 1.0], confidence_threshold=0.8,
                 va_lr_factor=1.0, va_weight_decay=0.0, va_constraint=True, va_constraint_range=1.2):
        super(ESCL, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.temperature = temperature
        self.scale_weights = scale_weights
        self.confidence_threshold = confidence_threshold
        self.va_lr_factor = va_lr_factor
        self.va_weight_decay = va_weight_decay
        self.va_constraint = va_constraint
        self.va_constraint_range = va_constraint_range
        
        # Initialize learnable VA coordinates
        if va_dict is not None:
            # If initial VA dictionary is provided, use it as initial value
            initial_va = torch.tensor([va_dict[e] for e in sorted(va_dict)])
            # Ensure correct shape (num_classes, 2)
            if initial_va.shape[0] != num_classes:
                initial_va = initial_va[:num_classes]
            if initial_va.shape[1] != 2:
                # If not 2D, truncate or extend to 2D
                if initial_va.shape[1] > 2:
                    initial_va = initial_va[:, :2]
                else:
                    padding = torch.zeros(num_classes, 2 - initial_va.shape[1])
                    initial_va = torch.cat([initial_va, padding], dim=1)
        else:
            # If no initial value is provided, use random initialization
            initial_va = torch.randn(num_classes, 2) * 0.1
        
        # Set VA coordinates as learnable parameters
        self.va = nn.Parameter(initial_va)
    
    def forward(self, features, labels):
        """
        Args:
            features: (N, C, T, V, M) Feature tensor, may contain M dimension
            labels: (N,) Label tensor
            
        Returns:
            total_loss: Total loss
            time_loss: Time dimension loss
            spatial_loss: Space dimension loss
        """
        # Handle possible 5D input (N, C, T, V, M)
        if len(features.shape) == 5:
            N, C, T, V, M = features.shape
            # Merge M dimension into V dimension
            features = features.permute(0, 1, 2, 4, 3).contiguous().view(N, C, T, V*M)
        
        N, C, T, V = features.shape
        
        # Calculate contrastive loss in time dimension
        time_features = features.mean(dim=3)
        time_loss = self._compute_escl_loss(time_features, labels)
        
        # Calculate contrastive loss in space dimension
        spatial_features = features.mean(dim=2)
        spatial_loss = self._compute_escl_loss(spatial_features, labels)
        
        total_loss = self.scale_weights[0] * time_loss + self.scale_weights[1] * spatial_loss
        
        # Add VA coordinate regularization loss
        va_reg_loss = 0.0
        if self.va_weight_decay > 0:
            va_reg_loss = self.va_weight_decay * torch.sum(self.va ** 2)
            total_loss = total_loss + va_reg_loss
        
        return total_loss, time_loss, spatial_loss
    
    def _compute_escl_loss(self, features, labels):
        """
        Calculate ESCL loss
        
        Args:
            features: (N, C) Feature tensor
            labels: (N,) Label tensor
            
        Returns:
            loss: ESCL loss
        """
        N, C = features.shape
        
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Calculate similarity matrix
        similarity_matrix = torch.mm(features, features.t())
        
        # Create label mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal elements
        mask = mask - torch.eye(mask.shape[0]).to(mask.device)
        
        # Get current VA coordinates
        current_va = self.va[labels.squeeze()]
        
        # Calculate semantic similarity based on VA distance
        va_similarity = torch.exp(-torch.cdist(current_va, current_va, p=2) / self.temperature)
        
        # Combine feature similarity and semantic similarity
        combined_similarity = similarity_matrix * va_similarity
        
        # Calculate loss
        positive_mask = mask
        negative_mask = 1 - mask - torch.eye(mask.shape[0]).to(mask.device)
        
        positive_loss = -torch.log(torch.exp(combined_similarity) / (torch.sum(torch.exp(combined_similarity), dim=1, keepdim=True) + 1e-8))
        positive_loss = (positive_loss * positive_mask).sum() / (positive_mask.sum() + 1e-8)
        
        negative_loss = torch.log(1 + torch.sum(torch.exp(combined_similarity) * negative_mask, dim=1, keepdim=True))
        negative_loss = (negative_loss * negative_mask).sum() / (negative_mask.sum() + 1e-8)
        
        loss = self.gamma * positive_loss + (1 - self.gamma) * negative_loss
        
        return loss
    
    def get_va_coordinates(self):
        """Get currently learned VA coordinates (return all centers)"""
        return self.va.detach().cpu().numpy()
    
    def get_va_parameters(self):
        """Get VA coordinate parameters for setting different learning rates"""
        return [self.va]

class STContrastiveLoss(nn.Module):
    """ST multi-scale contrastive loss function (dimension bug fixed)"""
    def __init__(self, temperature=0.1, scale_weights=[1.0, 1.0], confidence_threshold=0.8, learnable_temperature=True):
        super(STContrastiveLoss, self).__init__()
        self.scale_weights = scale_weights
        self.confidence_threshold = confidence_threshold
        
        # Set temperature parameter as learnable
        if learnable_temperature:
            # Use log form to ensure temperature is always positive
            self.log_temperature = nn.Parameter(torch.log(torch.ones(1) * temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))
            self.log_temperature = None
    
    @property
    def temperature(self):
        """Get current temperature value (ensure positive)"""
        if self.log_temperature is not None:
            return torch.exp(self.log_temperature)
        else:
            return self.temperature
    
    def forward(self, features, labels):
        N, C, T, V = features.shape
        
        # In time dimension compute contrastive loss
        time_features = features.mean(dim=3)
        time_loss = self._compute_contrastive_loss(time_features, labels)
        
        # In space dimension compute contrastive loss
        spatial_features = features.mean(dim=2)
        spatial_loss = self._compute_contrastive_loss(spatial_features, labels)
        
        total_loss = self.scale_weights[0] * time_loss + self.scale_weights[1] * spatial_loss
        return total_loss, time_loss, spatial_loss
    
    def _compute_contrastive_loss(self, features, labels):
        """Fix version: correctly normalize features in feature dimension, add high confidence selection"""
        N, C, D = features.shape
        
        # L2 normalize (in the last feature dimension)
        features = features.view(N, -1)
        features = F.normalize(features, dim=1)
        
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        mask = mask - torch.diag(torch.diag(mask))
        
        exp_sim = torch.exp(similarity_matrix)
        pos_sum = torch.sum(exp_sim * mask, dim=1)
        neg_sum = torch.sum(exp_sim, dim=1) + 1e-8
        
        # Compute confidence: pos sample similarity / total similarity
        confidence = (pos_sum + 1e-8) / neg_sum
        
        # Create high confidence mask (confidence >= threshold)
        high_conf_mask = (confidence >= self.confidence_threshold).float()
        
        # Compute loss
        loss = -torch.log((pos_sum + 1e-8) / neg_sum)
        
        # Apply high confidence selection
        filtered_loss = loss * high_conf_mask
        
        # If no high confidence samples, return 0 avoid NaN
        if high_conf_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        return torch.mean(filtered_loss[high_conf_mask.bool()])

class CenterLoss(nn.Module):
    """Center loss function"""
    def __init__(self, num_classes, feat_dim, lambda_c=0.003):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        centers_batch = self.centers[labels]
        loss = self.lambda_c * (features - centers_batch).pow(2).sum() / features.size(0)
        return loss