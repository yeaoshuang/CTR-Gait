import logging
import numpy as np
import random

# Import data preprocessing module
from .random_move import random_move


def create(args):
    # Dataset uses default processing method
    dataset = args.dataset
    if dataset not in ['joint', 'bone', 'motion', 'cmu']:
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.format(dataset))
        raise ValueError()
    # Dataset now uses reader for data augmentation
    return Reader()


class Reader:
    def __init__(self):
        pass
    
    def start(self):
        # For datasets, reader is mainly used for data preprocessing
        logging.info('Using Reader for data preprocessing')
        pass
    
    def preprocess_data(self, data, phase='train', move_config=None):
        """
        Data preprocessing function
        
        Args:
        data: Input data
        phase: Training phase, 'train' or 'eval'
        move_config: Configuration parameters for random_move
        
        Returns:
        Preprocessed data
        """
        # Apply data augmentation only during training phase
        if phase == 'train':
            data = random_move(data, move_config)
        return data