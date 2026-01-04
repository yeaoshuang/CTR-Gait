import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# Import separated modules
from .escl import ESCL, MultiCenterESCL, STContrastiveLoss, CenterLoss
from .expert import SadExpert


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels) 
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1, data_type='joint'):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.data_type = data_type
        
        # Adjust channel configuration based on data type
        if data_type == 'bone':
            # Bone data: usually has fewer channels, need to adjust reduction parameters
            if in_channels <= 3:
                self.rel_channels = max(4, in_channels)
                self.mid_channels = max(8, in_channels * 2)
            else:
                self.rel_channels = max(8, in_channels // rel_reduction)
                self.mid_channels = max(16, in_channels // mid_reduction)
        else:
            # Joint data: maintain original logic
            if in_channels == 3 or in_channels == 9:
                self.rel_channels = 8
                self.mid_channels = 16
            else:
                self.rel_channels = in_channels // rel_reduction
                self.mid_channels = in_channels // mid_reduction
        
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1


class EnhancedTemporalTransformer(nn.Module):
    """Enhanced Temporal Transformer with positional encoding and LayerNorm"""
    def __init__(self, channels, num_heads=8, depth=1, dropout=0.1, 
                 max_seq_len=300, use_positional_encoding=True):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=num_heads, dropout=dropout,
            dim_feedforward=channels*4, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Learnable positional encoding
        self.use_pe = use_positional_encoding
        if use_positional_encoding:
            self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, channels) * 0.02)
        
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (N, C, T, 1) -> (N, T, C)
        N, C, T, _ = x.shape
        x = x.squeeze(-1).transpose(1, 2)
        
        # Add positional encoding
        if self.use_pe:
            x = x + self.pos_embedding[:, :T, :]
        
        # Transformer encoding
        x = self.norm(self.encoder(x))
        
        # Restore dimension (N, T, C) -> (N, C, T, 1)
        return x.transpose(1, 2).unsqueeze(-1)



class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # GCN spatial modeling
        y = self.gcn1(x)
        
        # Temporal downsampling
        if self.stride > 1:
            y = self.time_pool(y)
        
        # Transformer temporal modeling
        N, C, T, V = y.shape
        y_for_transformer = y.mean(-1, keepdim=True)  # (N,C,T,1)
        y_transformed = self.temporal_transformer(y_for_transformer)  # (N,C,T,1)
        
        # Expand back to spatial dimension
        y = y_transformed.expand(-1, -1, -1, V)
        
        # Residual connection
        return self.relu(y + self.residual(x))


class Model(nn.Module):
    def __init__(self, num_class=4, num_point=16, num_person=1, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, dataset='', data_shape=None,  # Add data_shape parameter for compatibility
                 # Transformer parameters (enabled by default)
                 transformer_start_layer=4,  # Start using Transformer from layer 4
                 transformer_heads=8, 
                 transformer_depth=1, 
                 transformer_dropout=0.1,
                 # Contrastive loss parameters
                 use_st_contrastive=False, 
                 contrastive_temperature=0.1, 
                 contrastive_scale_weights=[1.0, 1.0],
                 contrastive_confidence_threshold=0.8,  # Add high confidence threshold parameter
                 contrastive_learnable_temperature=True,  # Add learnable temperature parameter
                 # ESCL parameters
                 use_escl=False,
                 escl_gamma=0.4,
                 escl_va_dict=None,
                 # Multi-center ESCL parameters
                 escl_use_multi_center=False,
                 escl_num_centers=2,
                 escl_kl_weight=0.01,
                 # Sad expert module parameters
                 use_sad_expert=False,
                 sad_expert_feat_dim=256,
                 sad_expert_hidden=16,
                 sad_expert_dropout=0.3,
                 # Configuration object (for getting additional parameters)
                 config=None, **kwargs):
        super(Model, self).__init__()

        # Save contrastive loss parameters
        self.use_st_contrastive = use_st_contrastive
        
        # ESCL parameters
        self.use_escl = use_escl
        self.escl_gamma = escl_gamma
        self.escl_va_dict = escl_va_dict
        self.escl_use_multi_center = escl_use_multi_center
        self.escl_num_centers = escl_num_centers
        self.escl_kl_weight = escl_kl_weight
        
        # Multi-expert module parameters
        self.use_sad_expert = use_sad_expert
        
        # If there is a configuration object, get parameters from the configuration
        if config is not None and hasattr(config, 'model_args'):
            self.use_escl = getattr(config.model_args, 'use_escl', self.use_escl)
            self.escl_gamma = getattr(config.model_args, 'escl_gamma', self.escl_gamma)
            self.escl_va_dict = getattr(config.model_args, 'escl_va_dict', self.escl_va_dict)
            self.escl_use_multi_center = getattr(config.model_args, 'escl_use_multi_center', self.escl_use_multi_center)
            self.escl_num_centers = getattr(config.model_args, 'escl_num_centers', self.escl_num_centers)
            self.escl_kl_weight = getattr(config.model_args, 'escl_kl_weight', self.escl_kl_weight)
            self.escl_va_lr_factor = getattr(config.model_args, 'escl_va_lr_factor', 1.5)
            # Sad expert parameters
            self.use_sad_expert = getattr(config.model_args, 'use_sad_expert', self.use_sad_expert)
        else:
            # Get ESCL parameters from kwargs
            self.use_escl = kwargs.get('use_escl', self.use_escl)
            self.escl_gamma = kwargs.get('escl_gamma', self.escl_gamma)
            self.escl_va_dict = kwargs.get('escl_va_dict', self.escl_va_dict)
            self.escl_use_multi_center = kwargs.get('escl_use_multi_center', self.escl_use_multi_center)
            self.escl_num_centers = kwargs.get('escl_num_centers', self.escl_num_centers)
            self.escl_kl_weight = kwargs.get('escl_kl_weight', self.escl_kl_weight)
            self.escl_va_lr_factor = kwargs.get('escl_va_lr_factor', 1.5)
        
        # Initialize VA dictionary (if using ESCL but no VA dictionary is provided, use default values)
        if self.use_escl and self.escl_va_dict is None:
            # Default VA coordinate table
            self.escl_va_dict = {
                0: [0.81, 0.83],  # happy
                1: [0.27, 0.24],  # sad
                2: [0.17, 0.84],  # angry
                3: [0.50, 0.32],  # neutral
            }
        
        # Initialize VA parameters
        if self.use_escl:
            # Initialize VA parameters
            self.va_params = nn.ParameterDict()
            for class_id, va in self.escl_va_dict.items():
                if self.escl_use_multi_center:
                    # Multi-center mode: each class has multiple VA centers
                    for i in range(self.escl_num_centers):
                        # Add some random perturbation to make initial centers different
                        perturb = (torch.rand(2) - 0.5) * 0.1
                        center = torch.tensor(va) + perturb
                        self.va_params[f'class_{class_id}_center_{i}'] = nn.Parameter(center)
                else:
                    # Single-center mode
                    self.va_params[f'class_{class_id}'] = nn.Parameter(torch.tensor(va))
            
            # Register learning rate factor for VA parameters
            if hasattr(self, 'escl_va_lr_factor'):
                for param in self.va_params.parameters():
                    param.lr_factor = self.escl_va_lr_factor
        
        if use_st_contrastive:
            self.st_contrastive_loss = STContrastiveLoss(
                temperature=contrastive_temperature,
                scale_weights=contrastive_scale_weights,
                confidence_threshold=contrastive_confidence_threshold,  # Pass confidence threshold
                learnable_temperature=contrastive_learnable_temperature  # Pass learnable temperature parameter
            )
            self.center_loss = CenterLoss(num_classes=num_class, feat_dim=256, lambda_c=0.003)
        
        if use_escl:
            # Check if using multi-center ESCL
            if self.escl_use_multi_center:
                K = self.escl_num_centers
                kl_weight = self.escl_kl_weight
                self.escl_loss = MultiCenterESCL(
                    num_class=num_class,
                    K=K,
                    gamma=self.escl_gamma,
                    temperature=contrastive_temperature,
                    scale_weights=contrastive_scale_weights,
                    confidence_threshold=contrastive_confidence_threshold,
                    kl_weight=kl_weight
                )
            else:
                self.escl_loss = ESCL(
                    num_classes=num_class,
                    va_dict=self.escl_va_dict,
                    gamma=self.escl_gamma,
                    temperature=contrastive_temperature,
                    scale_weights=contrastive_scale_weights,
                    confidence_threshold=contrastive_confidence_threshold
                )
            self.center_loss = CenterLoss(num_classes=num_class, feat_dim=256, lambda_c=0.003)

        # Handle the number of channels for the dataset
        if dataset.startswith(('joint', 'bone', 'motion')):
            in_channels = 3
        self.dataset = dataset

        if graph is None:
            raise ValueError()
        else:
            # Directly import graph class
            if graph == 'src.dataset.graph.Graph':
                from src.dataset.graph import Graph
                self.graph = Graph(**graph_args)
            else:
                Graph = import_class(graph)
                self.graph = Graph(**graph_args)
        A = self.graph.A

        self.num_class = num_class
        self.num_point = num_point if num_point else 16
        self.in_channels = in_channels
        
        base_channel = 64
        
        # Data BN layer
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * self.num_point)
        bn_init(self.data_bn, 1)

        # ==================== Hybrid Architecture: Layered Design ====================
        # Define configuration for each layer: (in_channels, out_channels, stride, residual)
        layer_configs = [
            (in_channels, base_channel, 1, False),    # l1: TCN
            (base_channel, base_channel, 1, True),    # l2: TCN
            (base_channel, base_channel, 1, True),    # l3: TCN
            (base_channel, base_channel*2, 2, True),  # l4: Transformer + stride=2
            (base_channel*2, base_channel*2, 1, True), # l5: Transformer
            (base_channel*2, base_channel*2, 1, True), # l6: Transformer
            (base_channel*2, base_channel*4, 2, True), # l7: Transformer + stride=2
            (base_channel*4, base_channel*4, 1, True), # l8: Transformer
            (base_channel*4, base_channel*4, 1, True), # l9: Transformer
            (base_channel*4, base_channel*4, 1, True), # l10: Transformer
        ]

        self.layers = nn.ModuleList()
        for idx, (in_ch, out_ch, stride, residual) in enumerate(layer_configs):
            # First 3 layers use TCN (extract local features), from layer 4 use Transformer (model global dependencies)
            if idx < transformer_start_layer - 1:
                layer = TCN_GCN_unit(
                    in_ch, out_ch, A, stride=stride, residual=residual, adaptive=adaptive,
                    kernel_size=5, dilations=[1, 2]
                )
            else:
                layer = Transformer_GCN_unit(
                    in_ch, out_ch, A, stride=stride, residual=residual, adaptive=adaptive,
                    num_heads=transformer_heads, depth=transformer_depth, dropout=transformer_dropout
                )
            self.layers.append(layer)
        # =========================================================

        # Classification head
        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        
        # Dropout
        self.drop_out = nn.Dropout(drop_out) if drop_out else lambda x: x
        
        # Expert module - Sad expert only
        if self.use_sad_expert:
            # Get Sad expert configuration from config or parameters
            sad_expert_feat_dim = kwargs.get('sad_expert_feat_dim', 256)
            sad_expert_hidden = kwargs.get('sad_expert_hidden', 16)
            sad_expert_dropout = kwargs.get('sad_expert_dropout', 0.3)
            
            # If configuration exists, use values from configuration
            if config is not None and hasattr(config, 'model_args'):
                sad_expert_feat_dim = getattr(config.model_args, 'sad_expert_feat_dim', sad_expert_feat_dim)
                sad_expert_hidden = getattr(config.model_args, 'sad_expert_hidden', sad_expert_hidden)
                sad_expert_dropout = getattr(config.model_args, 'sad_expert_dropout', sad_expert_dropout)
            
            # Initialize Sad expert
            self.sad_expert = SadExpert(
                in_channels=sad_expert_feat_dim,
                hidden_channels=sad_expert_hidden,
                dropout_rate=sad_expert_dropout,
                num_classes=num_class
            )
        
        # Single expert mode (Sad expert only)

    def get_va_parameters(self):
        """Get VA coordinate parameters for setting different learning rates"""
        if self.use_escl:
            return self.escl_loss.get_va_parameters()
        return []

    def forward(self, x, labels=None):
        # Input processing
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()
        
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # Layer-by-layer forward propagation
        features_for_contrastive = None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Save features after layer 6 for contrastive loss (deeper semantic features)
            if (self.use_st_contrastive or self.use_escl) and i ==  6:
                features_for_contrastive = x.clone()

        # Save spatiotemporal features for expert module
        spatiotemporal_features = x.clone()  # (N*M, C, T, V)
        
        # Global pooling
        c_new = x.size(1)
        t_new = x.size(2)
        v_new = x.size(3)
        x = x.reshape(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        features = x.clone()  # For center loss
        x = self.drop_out(x)
        
        # Classification output
        logits = self.fc(x)

        # Expert module processing - single expert mode (Sad expert only)
        if self.use_sad_expert:
            if self.training:
                # During training: use true labels to determine Sad samples
                is_sad = (labels == 3)  # Boolean tensor (N,)
            else:
                # During inference: use main network prediction probability to determine Sad samples
                is_sad = (F.softmax(logits, dim=1)[:, 3] > 0.2)  # Boolean tensor (N,)
            
            # Reshape spatiotemporal features to match batch size
            spatiotemporal_features = spatiotemporal_features.view(N, M, c_new, t_new, v_new)
            # If multiple people, take average
            spatiotemporal_features = spatiotemporal_features.mean(1)  # (N, C, T, V)
            
            sad_residual = self.sad_expert(spatiotemporal_features, is_sad, train=self.training)

            if self.training:
                logits = logits + sad_residual  # Only sad_residual for Sad samples is non-zero
            else:
                # Inference: weight expert residual with main network Sad probability
                p_sad = F.softmax(logits, dim=1)[:, 3].unsqueeze(1)  # (N,1)
                logits = logits + p_sad * sad_residual               # Only effective for Sad dimension
        # -------------------------------------

        # Joint loss calculation
        if (self.use_st_contrastive or self.use_escl) and self.training and labels is not None:
            cls_loss = F.cross_entropy(logits, labels)
            
            if self.use_st_contrastive:
                # ST contrastive loss
                ctr_loss, t_loss, s_loss = self.st_contrastive_loss(features_for_contrastive, labels)
            else:
                # ESCL loss
                ctr_loss, t_loss, s_loss = self.escl_loss(features_for_contrastive, labels)
            
            # Center loss
            center_loss = self.center_loss(features, labels)
            
            # Weighted joint loss
            total_loss = cls_loss + 0.1 * ctr_loss + center_loss
            
            return logits, features, {
                'total_loss': total_loss,
                'cls_loss': cls_loss,
                'contrastive_loss': ctr_loss,
                'contrastive_time_loss': t_loss,
                'contrastive_space_loss': s_loss,
                'center_loss': center_loss
            }

        return logits, features













