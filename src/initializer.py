import os, yaml, thop, warnings, logging, torch, numpy as np
# Filter warnings before importing pynvml
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*", category=FutureWarning)
import pynvml
from copy import deepcopy
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from . import utils as U
from . import dataset
from . import model
from . import scheduler

# Add warning filtering at the beginning of the file
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*", category=FutureWarning)


class Initializer():
    def __init__(self, args):
        self.args = args
        self.init_save_dir()

        logging.info('')
        logging.info('Starting preparing ...')
        self.init_environment()
        self.init_device()
        self.init_dataloader()
        self.init_model()
        self.init_optimizer()
        self.init_lr_scheduler()
        self.init_loss_func()
        logging.info('Successful!')
        logging.info('')

    def init_save_dir(self):
        self.save_dir = U.set_logging(self.args)
        with open('{}/config.yaml'.format(self.save_dir), 'w') as f:
            yaml.dump(vars(self.args), f)
        logging.info('Saving folder path: {}'.format(self.save_dir))

    def init_environment(self):
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)  # Set random seed for all GPUs
        # Enable deterministic algorithms to ensure reproducible results
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Fix torch.backends access issue
        try:
            # Use getattr to safely access torch.backends.cudnn
            cudnn = getattr(torch, 'backends', None)
            if cudnn is not None:
                cudnn.benchmark = False  # Disable cudnn auto-optimization for determinism
                cudnn.enabled = True
                cudnn.deterministic = True  # Enable deterministic algorithms
        except:
            pass

        # Set seed for Python's built-in random number generator
        import random
        random.seed(self.args.seed)
        
        # Set environment variable to ensure determinism
        os.environ['PYTHONHASHSEED'] = str(self.args.seed)
        
        self.global_step = 0
        if self.args.debug:
            self.no_progress_bar = True
            self.model_name = 'debug'
            self.scalar_writer = None
        elif self.args.evaluate or self.args.extract:
            self.no_progress_bar = self.args.no_progress_bar
            self.model_name = '{}_{}_{}'.format(self.args.config, self.args.model_type, self.args.dataset)
            self.scalar_writer = None
            warnings.filterwarnings('ignore')
        else:
            self.no_progress_bar = self.args.no_progress_bar
            self.model_name = '{}_{}_{}'.format(self.args.config, self.args.model_type, self.args.dataset)
            self.scalar_writer = SummaryWriter(logdir=self.save_dir)
            warnings.filterwarnings('ignore')
        logging.info('Saving model name: {}'.format(self.model_name))

    def init_device(self):
        if type(self.args.gpus) is int:
            self.args.gpus = [self.args.gpus]
        if len(self.args.gpus) > 0 and torch.cuda.is_available():
            pynvml.nvmlInit()
            for i in self.args.gpus:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memused = float(meminfo.used) / 1024.0 / 1024.0
                logging.info('GPU-{} used: {}MB'.format(i, memused))
                # Remove GPU memory usage check, allow running when GPU memory usage exceeds 1000MB
            pynvml.nvmlShutdown()
            self.output_device = self.args.gpus[0]
            self.device =  torch.device('cuda:{}'.format(self.output_device))
            torch.cuda.set_device(self.output_device)
        else:
            logging.info('Using CPU!')
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            self.output_device = None
            self.device =  torch.device('cpu')

    def init_dataloader(self):
        dataset_name = self.args.dataset.split('-')[0]
        dataset_args = self.args.dataset_args[dataset_name]
        dataset_args['debug'] = self.args.debug
        self.train_batch_size = dataset_args['train_batch_size']
        self.eval_batch_size = dataset_args['eval_batch_size']
        # Fix dataset.create call parameter issue
        root_folder = dataset_args['root_folder']
        num_frame = dataset_args['num_frame']
        inputs = dataset_args['inputs']
        # Add random_move_args to dataset_kwargs
        dataset_kwargs = {k: v for k, v in dataset_args.items() if k not in ['root_folder', 'num_frame', 'inputs']}
        # Add random_move_args parameter
        dataset_kwargs['random_move_args'] = self.args.random_move_args
        self.feeders, self.data_shape, self.num_class, self.A, self.parts = dataset.create(
            self.args.dataset, root_folder, num_frame, inputs, **dataset_kwargs
        )
        
        # Create fixed seed random number generators
        import numpy as np
        train_generator = torch.Generator()
        train_generator.manual_seed(self.args.seed)
        
        eval_generator = torch.Generator()
        eval_generator.manual_seed(self.args.seed)
        
        # Set num_workers to 0, ensure data is stored on SSD for better IO performance
        self.train_loader = DataLoader(self.feeders['train'],
            batch_size=self.train_batch_size, num_workers=0,
            pin_memory=True, shuffle=True, drop_last=True,
            generator=train_generator  # Use fixed seed generator
        )
        self.eval_loader = DataLoader(self.feeders['eval'],
            batch_size=self.eval_batch_size, num_workers=0,
            pin_memory=True, shuffle=False, drop_last=False,
            generator=eval_generator  # Use fixed seed generator
        )
        self.location_loader = None
        logging.info('Dataset: {}'.format(self.args.dataset))
        logging.info('Batch size: train-{}, eval-{}'.format(self.train_batch_size, self.eval_batch_size))
        logging.info('Data shape (branch, channel, frame, joint, person): {}'.format(self.data_shape))
        logging.info('Number of action classes: {}'.format(self.num_class))

    def init_model(self):
        # Merge parameters to avoid conflicts
        model_kwargs = {}
        model_kwargs.update(self.args.model_args)  # First add parameters from model_args
        model_kwargs.update({
            'data_shape': self.data_shape,
            'num_class': self.num_class,
            'A': torch.Tensor(self.A),
            'parts': self.parts,
            'dataset': self.args.dataset,  # Add dataset information
            'config': self.args,  # Pass the complete configuration object
        })
        self.model = model.create(self.args.model_type, **model_kwargs)
        logging.info('Model: {} {}'.format(self.args.model_type, self.args.model_args))
        logging.info('Data shape passed to model: {}'.format(self.data_shape))
        with open('{}/model.txt'.format(self.save_dir), 'w') as f:
            print(self.model, file=f)
        # Fix thop.profile return value issue
        # Ensure input shape is 5D: [batch_size, channels, frames, vertices, persons]
        if len(self.data_shape) > 5:
            # If data_shape has extra dimensions, only take the last 5
            input_shape = [1] + list(self.data_shape[-5:])
        else:
            # Ensure it's 5D
            input_shape = [1] + list(self.data_shape)
        profile_result = thop.profile(deepcopy(self.model), inputs=torch.rand(input_shape), verbose=False)
        if isinstance(profile_result, tuple) and len(profile_result) >= 2:
            flops, params = profile_result[0], profile_result[1]
        else:
            flops, params = profile_result, 0
        logging.info('Model profile: {:.2f}G FLOPs and {:.2f}M Parameters'.format(flops / 1e9, params / 1e6))
        self.model = torch.nn.DataParallel(
            self.model.to(self.device), device_ids=self.args.gpus, output_device=self.output_device
        )
        pretrained_model = '{}/{}.pth.tar'.format(self.args.pretrained_path, self.model_name)
        if os.path.exists(pretrained_model):
            checkpoint = torch.load(pretrained_model, map_location=torch.device('cpu'))
            self.model.module.load_state_dict(checkpoint['model'])
            self.cm = checkpoint['best_state']['cm']
            logging.info('Pretrained model: {}'.format(pretrained_model))
        elif self.args.pretrained_path:
            logging.warning('Warning: Do NOT exist this pretrained model: {}!'.format(pretrained_model))
            logging.info('Create model randomly.')

    def init_optimizer(self):
        try:
            optimizer = U.import_class('torch.optim.{}'.format(self.args.optimizer))
        except:
            logging.warning('Warning: Do NOT exist this optimizer: {}!'.format(self.args.optimizer))
            logging.info('Try to use SGD optimizer.')
            self.args.optimizer = 'SGD'
            optimizer = U.import_class('torch.optim.SGD')
        optimizer_args = self.args.optimizer_args[self.args.optimizer]
        self.optimizer = optimizer(self.model.parameters(), **optimizer_args)
        logging.info('Optimizer: {} {}'.format(self.args.optimizer, optimizer_args))

    def init_lr_scheduler(self):
        scheduler_args = self.args.scheduler_args[self.args.lr_scheduler]
        self.max_epoch = scheduler_args['max_epoch']
        lr_scheduler = scheduler.create(self.args.lr_scheduler, len(self.train_loader), **scheduler_args)
        self.eval_interval, lr_lambda = lr_scheduler.get_lambda()
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        logging.info('LR_Scheduler: {} {}'.format(self.args.lr_scheduler, scheduler_args))

    def init_loss_func(self):
        self.loss_func = torch.nn.CrossEntropyLoss().to(self.device)
        logging.info('Loss function: {}'.format(self.loss_func.__class__.__name__))