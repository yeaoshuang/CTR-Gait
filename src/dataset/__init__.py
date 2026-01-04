import logging

from .graph import Graph
from .feeder import Feeder


__data_args = {
    'joint': {'class': 4, 'shape': [1, 3, 48, 16, 1], 'feeder': Feeder},
    'bone':  {'class': 4, 'shape': [1, 3, 48, 16, 1], 'feeder': Feeder},
    'motion':{'class': 4, 'shape': [1, 3, 48, 16, 1], 'feeder': Feeder},
}

def create(dataset, root_folder, num_frame, inputs, seed=123, **kwargs):
    # Select graph structure based on dataset type
    if dataset == 'bone':
        graph = Graph(data_type='bone')
    else:
        graph = Graph(data_type='joint')
    try:
        data_args = __data_args[dataset]
        data_args['shape'][2] = num_frame
    except:
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.format(dataset))
        raise ValueError()
    
    # Use correct path construction method for dataset
    dataset_path = root_folder
            
    kwargs.update({
        'dataset_path': dataset_path,
        'inputs': inputs,
        'num_frame': num_frame,
        'connect_joint': None,
        'seed': seed,  # Add random seed parameter
    })
    feeders = {
        'train': data_args['feeder']('train', **kwargs),
        'eval' : data_args['feeder']('eval', **kwargs),
    }
    return feeders, data_args['shape'], data_args['class'], graph.A, getattr(graph, 'parts', None)