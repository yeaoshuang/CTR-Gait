import numpy as np
import random
import torch


def random_move(data, move_config=None):
    """
    Apply random motion transformation to bone data, using global random seed consistent with configuration
    
    Args:
    data: Bone data with shape (C, T, V, M)
    move_config: Transformation configuration parameters
    
    Returns:
    Transformed data
    """
    if move_config is None:
        # Default configuration
        move_config = {
            'enable': True,
            'rotation_angles': [-10, -5, 0, 5, 10],  # Rotation angle options (degrees)
            'scale_factors': [0.9, 1.0, 1.1],        # Scale factor options
            'translation_x': [-0.2, -0.1, 0.0, 0.1, 0.2],  # X translation options
            'translation_y': [-0.2, -0.1, 0.0, 0.1, 0.2],  # Y translation options
        }
    
    # If transformation is not enabled, return original data directly
    if not move_config.get('enable', True):
        return data
    
    # Get data dimensions
    C, T, V, M = data.shape
    
    # Only transform 2D data (X, Y coordinates)
    if C < 2:
        return data
    
    # Ensure using global random seed
    # Note: No need to set random seed here, as global seed has been set in init_environment
    
    # Randomly select transformation parameters - using global random state
    # Use uniform instead of choice for smoother random distribution
    rotation_angles = move_config['rotation_angles']
    scale_factors = move_config['scale_factors']
    translation_x = move_config['translation_x']
    translation_y = move_config['translation_y']
    
    # Randomly select parameters from configuration range
    A_start = random.uniform(min(rotation_angles), max(rotation_angles))
    A_end = random.uniform(min(rotation_angles), max(rotation_angles))
    
    S_start = random.uniform(min(scale_factors), max(scale_factors))
    S_end = random.uniform(min(scale_factors), max(scale_factors))
    
    T_x_start = random.uniform(min(translation_x), max(translation_x))
    T_x_end = random.uniform(min(translation_x), max(translation_x))
    
    T_y_start = random.uniform(min(translation_y), max(translation_y))
    T_y_end = random.uniform(min(translation_y), max(translation_y))
    
    # Linear interpolation for 48 frames
    a = np.linspace(A_start, A_end, T) * np.pi / 180  # Convert to radians
    s = np.linspace(S_start, S_end, T)
    t_x = np.linspace(T_x_start, T_x_end, T)
    t_y = np.linspace(T_y_start, T_y_end, T)
    
    # Create transformed data copy
    data_transformed = np.copy(data)
    
    # Pre-calculate cos and sin values for efficiency
    cos_a = np.cos(a)
    sin_a = np.sin(a)
    
    # Vectorized operations for efficiency
    # Build transformation matrix: [[cos(a)*s, -sin(a)*s], [sin(a)*s, cos(a)*s]]
    theta_00 = cos_a * s
    theta_01 = -sin_a * s
    theta_10 = sin_a * s
    theta_11 = cos_a * s
    
    # Apply transformation to each frame (vectorized operations)
    for i_frame in range(T):
        # Apply transformation to all joints and persons
        # new_x = theta[0, 0] * x + theta[0, 1] * y + t_x
        # new_y = theta[1, 0] * x + theta[1, 1] * y + t_y
        x = data[0, i_frame, :, :]  # X coordinates of all joints
        y = data[1, i_frame, :, :]  # Y coordinates of all joints
        
        new_x = theta_00[i_frame] * x + theta_01[i_frame] * y + t_x[i_frame]
        new_y = theta_10[i_frame] * x + theta_11[i_frame] * y + t_y[i_frame]
        
        # Update data
        data_transformed[0, i_frame, :, :] = new_x
        data_transformed[1, i_frame, :, :] = new_y
    
    return data_transformed