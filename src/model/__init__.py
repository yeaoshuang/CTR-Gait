import math
from torch import nn

from .ctrgcn import Model as CTRGCN


def create(model_type, **kwargs):
    # Only support CTRGCN model
    if model_type == 'CTRGCN':
        # CTRGCN model parameters
        ctrgcn_args = {
            'num_class': kwargs.get('num_class', 60),
            'num_point': kwargs.get('num_point', 25),
            'num_person': kwargs.get('num_person', 2),
            'graph': kwargs.get('graph', 'graph.ntu_rgb_d.Graph'),
            'graph_args': kwargs.get('graph_args', {'labeling_mode': 'spatial'}),
            'in_channels': kwargs.get('in_channels', 3),
            'drop_out': kwargs.get('drop_out', 0),
            'adaptive': kwargs.get('adaptive', True),
            'data_shape': kwargs.get('data_shape'),
            'dataset': kwargs.get('dataset', ''),
            # Transformer parameters (enabled by default)
            'transformer_start_layer': kwargs.get('transformer_start_layer', 5),
            'transformer_heads': kwargs.get('transformer_heads', 8),
            'transformer_depth': kwargs.get('transformer_depth', 1),
            'transformer_dropout': kwargs.get('transformer_dropout', 0.1),
            # Contrastive loss parameters
            'use_st_contrastive': kwargs.get('use_st_contrastive', False),
            'contrastive_temperature': kwargs.get('contrastive_temperature', 0.1),
            'contrastive_scale_weights': kwargs.get('contrastive_scale_weights', [1.0, 1.0]),
            # ESCL parameters
            'use_escl': kwargs.get('use_escl', False),
            'escl_feat_dim': kwargs.get('escl_feat_dim', 256),
            'escl_temperature': kwargs.get('escl_temperature', 0.1),
            'escl_margin': kwargs.get('escl_margin', 0.5),
            'escl_gamma': kwargs.get('escl_gamma', 0.5),
            'escl_lambda': kwargs.get('escl_lambda', 0.1),
            'escl_center_lr': kwargs.get('escl_center_lr', 0.5),
            # Multi-center ESCL parameters
            'escl_use_multi_center': kwargs.get('escl_use_multi_center', False),
            'escl_num_centers': kwargs.get('escl_num_centers', 2),
            'escl_kl_weight': kwargs.get('escl_kl_weight', 0.01),
            # Sad expert module parameters
            'use_sad_expert': kwargs.get('use_sad_expert', False),
            'sad_expert_feat_dim': kwargs.get('sad_expert_feat_dim', 256),
            'sad_expert_hidden': kwargs.get('sad_expert_hidden', 16),
            'sad_expert_dropout': kwargs.get('sad_expert_dropout', 0.3),
        }
        return CTRGCN(**ctrgcn_args)
    else:
        raise ValueError('Unsupported model type: {}'.format(model_type))