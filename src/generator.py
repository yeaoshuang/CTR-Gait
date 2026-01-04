import logging

from . import utils as U
from . import reader


class Generator():
    def __init__(self, args):
        U.set_logging(args)
        self.dataset = args.dataset
        self.generator = reader.create(args)

    def start(self):
        logging.info('')
        logging.info('Starting generating ...')
        logging.info('Dataset: {}'.format(self.dataset))
        if self.generator is not None:
            self.generator.start()
        else:
            logging.info('No special generator needed for this dataset.')
        logging.info('Finish generating!')