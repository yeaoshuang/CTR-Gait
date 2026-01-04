import torch
import torch.nn as nn
import torch.nn.functional as F

class SadExpert(nn.Module):
    """
    Sad emotion expert module: focuses on detecting and extracting features related to sad emotions
    Modified version: directly processes spatiotemporal features to avoid duplicate data issues
    
    Includes:
    1. Temporal attention mechanism: captures subtle action sequence patterns specific to sad emotions
    2. Spatial attention mechanism: focuses on areas with significant body posture changes (e.g., face, head)
    3. Multi-scale fusion: integrates feature information from different time scales
    """
    def __init__(self, in_channels=256, hidden_channels=128, dropout_rate=0.5, num_classes=4):
        super(SadExpert, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        
        # 1. Temporal attention mechanism - modified to directly process spatiotemporal data
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 2. Spatial attention mechanism - modified to directly process spatiotemporal data
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 3. Multi-scale temporal convolution
        self.multi_scale_temporal = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, hidden_channels, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(inplace=True)
            ) for k in [3, 5, 7]
        ])
        
        # 4. Multi-scale spatial convolution
        self.multi_scale_spatial = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, hidden_channels, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(inplace=True)
            ) for k in [3, 5, 7]
        ])
        
        # 5. Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Conv1d(hidden_channels * 6, hidden_channels * 2, kernel_size=1),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # 6. Feature aggregation layer
        self.feature_aggregator = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels * 2, in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # 7. Classification layer
        self.classifier = nn.Linear(in_channels, num_classes)
        nn.init.xavier_normal_(self.classifier.weight)
        
    def forward(self, x, is_sad=None, train=True):
        """
        Forward propagation function
        Args:
            x: Input features (N, C, T, V) - batch size, channels, time steps, joints
            is_sad: Boolean or boolean tensor indicating which samples are sad emotions
            train: Whether in training mode
        Returns:
            output: Features processed by expert module (N, num_classes)
        """
        N, C, T, V = x.size()
        
        # Process is_sad parameter to ensure it's a boolean tensor
        if is_sad is None:
            is_sad = torch.ones(N, dtype=torch.bool, device=x.device)
        elif isinstance(is_sad, bool):
            # If boolean, convert to tensor
            is_sad = torch.full((N,), is_sad, dtype=torch.bool, device=x.device)
        elif not isinstance(is_sad, torch.Tensor):
            # If other type, convert to boolean tensor
            is_sad = torch.tensor(is_sad, dtype=torch.bool, device=x.device)
            if is_sad.numel() == 1:
                is_sad = is_sad.repeat(N)
        
        # If no sad samples, directly return zero vector
        if not is_sad.any():
            return torch.zeros(N, self.num_classes, device=x.device)
        
        # Reshape data for easier processing
        # Temporal processing: (N, C, T, V) -> (N*V, C, T)
        x_temporal = x.permute(0, 3, 1, 2).contiguous().view(N * V, C, T)
        
        # Spatial processing: (N, C, T, V) -> (N*T, C, V)
        x_spatial = x.permute(0, 2, 1, 3).contiguous().view(N * T, C, V)
        
        # Calculate temporal attention
        temporal_attention = self.temporal_attention(x_temporal)  # (N*V, 1, T)
        temporal_attention = temporal_attention.view(N, V, 1, T).permute(0, 2, 3, 1)  # (N, 1, T, V)
        
        # Calculate spatial attention
        spatial_attention = self.spatial_attention(x_spatial)  # (N*T, 1, V)
        spatial_attention = spatial_attention.view(N, T, 1, V).permute(0, 2, 1, 3)  # (N, 1, T, V)
        
        # Apply attention mechanism
        x_attended = x * temporal_attention * spatial_attention  # (N, C, T, V)
        
        # Multi-scale feature extraction
        multi_scale_features = []
        
        # Temporal multi-scale features
        for conv in self.multi_scale_temporal:
            # Process time series of each joint separately
            feat = conv(x_temporal)  # (N*V, hidden_channels, T)
            feat = feat.view(N, V, feat.size(1), feat.size(2)).permute(0, 2, 3, 1)  # (N, hidden_channels, T, V)
            multi_scale_features.append(feat)
        
        # Spatial multi-scale features
        for conv in self.multi_scale_spatial:
            # Process spatial structure of each time step separately
            feat = conv(x_spatial)  # (N*T, hidden_channels, V)
            feat = feat.view(N, T, feat.size(1), feat.size(2)).permute(0, 2, 1, 3)  # (N, hidden_channels, T, V)
            multi_scale_features.append(feat)
        
        # Feature fusion
        fused_features = torch.cat(multi_scale_features, dim=1)  # (N, hidden_channels*6, T, V)
        
        # Global pooling, fuse spatiotemporal information
        fused_features = fused_features.mean(dim=[2, 3])  # (N, hidden_channels*6)
        
        # Feature fusion and aggregation
        fused_features = fused_features.unsqueeze(-1)  # (N, hidden_channels*6, 1)
        fused_features = self.fusion_layer(fused_features)  # (N, hidden_channels*2, 1)
        aggregated_features = self.feature_aggregator(fused_features)  # (N, in_channels)
        
        # Classification prediction
        logits = self.classifier(aggregated_features)  # (N, num_classes)
        
        # Keep non-zero output only for sad samples
        output = torch.zeros_like(logits)
        output[is_sad] = logits[is_sad]
        
        return output
    
    def get_importance(self):
        """
        Get importance metrics of the expert module
        Can be used for dynamic weight adjustment
        """
        # Here we can add some heuristic methods to calculate importance
        # For example, analyze the distribution of gate values or accuracy of specific emotion recognition
        return 1.0  # Default importance is 1.0