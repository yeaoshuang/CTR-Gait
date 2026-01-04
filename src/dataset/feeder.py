import os
import logging
import numpy as np
import pickle
import torch
import random
from torch.utils.data import Dataset


class Feeder(Dataset):
    """Dataset loader"""
    
    def __init__(self, phase, dataset_path, inputs, num_frame, debug, seed=123, **kwargs):
        self.phase = phase  # Store phase information for data augmentation
        self.dataset_path = dataset_path
        self.inputs = inputs
        self.num_frame = num_frame
        self.seed = seed  # Store random seed
        # Store random_move configuration parameters
        self.move_config = kwargs.get('random_move_args', None)
            
        file_phase = 'train' if phase == 'train' else 'test'
        
        # Determine data file name based on inputs type
        if inputs == 'B':
            data_type = 'bone'
        elif inputs == 'J':
            data_type = 'joint'
        elif inputs == 'M':
            data_type = 'motion'
        else:
            data_type = 'bone'  # Default to bone data
            
        # Data file path - build path based on data type
        data_path = '{}/{}_{}.npy'.format(self.dataset_path, file_phase, data_type)
        label_path = '{}/{}_label.pkl'.format(self.dataset_path, file_phase)
        
        try:
            # Load data
            self.data = np.load(data_path, mmap_mode='r')
            print(f"Data shape: {self.data.shape}")
            
            # Load labels
            with open(label_path, 'rb') as f:
                labels_data = pickle.load(f)
                
            # Process label format
            if isinstance(labels_data, tuple):
                # Tuple format
                self.name, self.label = labels_data
            else:
                # List format
                self.label = labels_data
                self.name = [str(i) for i in range(len(self.label))]
                
        except Exception as e:
            logging.info('')
            logging.error('Error: Wrong in loading data files: {} or {}!'.format(data_path, label_path))
            logging.info('Please generate data first!')
            raise ValueError()
            
        if debug:
            debug_size = min(200, len(self.label))
            self.data = self.data[:debug_size]
            self.label = self.label[:debug_size]
            self.name = self.name[:debug_size]
            
        # Set random seed to ensure consistency of data loading
        np.random.seed(self.seed)
        random.seed(self.seed)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # Get label
        label = self.label[index]
        
        # Get data
        data = self.data[index]
        
        # Ensure data shape is (C, T, V, M)
        if len(data.shape) == 3:
            # If only (C, T, V), add M dimension
            data = np.expand_dims(data, axis=-1)
        elif len(data.shape) == 2:
            # If only (T, V), add C and M dimensions
            data = np.expand_dims(np.expand_dims(data, axis=0), axis=-1)
        
        # Ensure data shape is (C, T, V, M)
        # If needed, perform cropping or padding in time dimension
        if len(data.shape) == 4:
            C, T, V, M = data.shape
            if T > self.num_frame:
                # Crop time dimension
                data = data[:, :self.num_frame, :, :]
            elif T < self.num_frame:
                # Pad time dimension
                pad_width = ((0, 0), (0, self.num_frame - T), (0, 0), (0, 0))
                data = np.pad(data, pad_width, mode='constant', constant_values=0)
        else:
            # If shape is incorrect, create data with default shape
            data = np.zeros((3, self.num_frame, 16, 1), dtype=np.float32)
        
        # Apply data preprocessing (including data augmentation)
        # Now get preprocessing function from reader and pass configuration parameters
        if self.phase in ['train', 'eval']:
            from ..reader import create
            # Create reader instance
            reader_args = type('Args', (), {'dataset': 'bone'})()  # Create a simple args object
            reader = create(reader_args)
            # Apply preprocessing, pass configuration parameters
            data = reader.preprocess_data(data, self.phase, self.move_config)
        
        # Convert to PyTorch tensor
        data = torch.FloatTensor(data)
        label = torch.LongTensor([label]).squeeze()  # Ensure label is 1D tensor
        
        return data, label, index