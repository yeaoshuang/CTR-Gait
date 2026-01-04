import numpy as np


class Step_Scheduler():
    def __init__(self, num_sample, warm_up, step_lr, gamma=0.1, **kwargs):
        warm_up_num = warm_up * num_sample
        # Modified to validate every round
        self.eval_interval = lambda epoch: 1
        # Support gamma parameter for setting learning rate decay rate
        if isinstance(step_lr, list):
            self.lr_lambda = lambda num: num / warm_up_num \
                                         if num < warm_up_num else \
                                         gamma ** np.sum(np.array(step_lr) <= num // num_sample)
        else:
            # If step_lr is an integer, it indicates decay every how many rounds
            self.lr_lambda = lambda num: num / warm_up_num \
                                         if num < warm_up_num else \
                                         gamma ** (num // num_sample // step_lr)

    def get_lambda(self):
        return self.eval_interval, self.lr_lambda


class Cosine_Scheduler():
    def __init__(self, num_sample, max_epoch, warm_up, **kwargs):
        warm_up_num = warm_up * num_sample
        max_num = max_epoch * num_sample
        # Modified to validate every round
        self.eval_interval = lambda epoch: 1
        self.lr_lambda = lambda num: num / warm_up_num \
                                     if num < warm_up_num else \
                                     0.5 * (np.cos((num - warm_up_num) / (max_num - warm_up_num) * np.pi) + 1)

    def get_lambda(self):
        return self.eval_interval, self.lr_lambda