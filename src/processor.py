import logging, torch, numpy as np
from tqdm import tqdm
from time import time

# Add mixed precision training support
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler

from . import utils as U
from .initializer import Initializer


class Processor(Initializer):

    def __init__(self, args):
        super().__init__(args)
        # Initialize GradScaler for mixed precision training
        self.scaler = GradScaler()

    def train(self, epoch):
        self.model.train()
        start_train_time = time()
        num_top1, num_sample = 0, 0
        
        # For accumulating loss information for each epoch
        epoch_loss_info = {
            'cls_loss': 0.0,
            'total_loss': 0.0,
            'contrastive_loss': 0.0,
            'contrastive_time_loss': 0.0,
            'contrastive_space_loss': 0.0,
            'center_loss': 0.0,
            'batch_count': 0
        }
        
        train_iter = self.train_loader if self.no_progress_bar else tqdm(self.train_loader, dynamic_ncols=True)
        for num, (x, y, _) in enumerate(train_iter):
            self.optimizer.zero_grad()

            # Using GPU
            x = x.float().to(self.device)
            y = y.long().to(self.device)

            # Using mixed precision training
            with autocast():
                # Calculating Output
                # Check if model uses ST contrastive loss or ESCL
                if (hasattr(self.model, 'use_st_contrastive') and self.model.use_st_contrastive) or \
                   (hasattr(self.model, 'use_escl') and self.model.use_escl):
                    out, _, loss_details = self.model(x, y)  # Pass labels for contrastive loss calculation
                    
                    # Check if loss_details contains classification loss
                    if isinstance(loss_details, dict) and 'cls_loss' in loss_details:
                        # Use joint loss (classification loss + contrastive loss + center loss)
                        total_loss = loss_details['total_loss']
                        
                        # Record detailed loss information
                        if self.scalar_writer:
                            self.scalar_writer.add_scalar('cls_loss', loss_details['cls_loss'].item(), self.global_step)
                            self.scalar_writer.add_scalar('total_loss', total_loss.item(), self.global_step)
                            self.scalar_writer.add_scalar('contrastive_loss', loss_details['contrastive_loss'], self.global_step)
                            self.scalar_writer.add_scalar('center_loss', loss_details['center_loss'].item(), self.global_step)
                        
                        # Accumulate loss information for output at epoch end
                        epoch_loss_info['cls_loss'] += loss_details['cls_loss'].item()
                        epoch_loss_info['total_loss'] += total_loss.item()
                        epoch_loss_info['contrastive_loss'] += loss_details['contrastive_loss']
                        epoch_loss_info['contrastive_time_loss'] += loss_details['contrastive_time_loss']
                        epoch_loss_info['contrastive_space_loss'] += loss_details['contrastive_space_loss']
                        epoch_loss_info['center_loss'] += loss_details['center_loss'].item()
                        epoch_loss_info['batch_count'] += 1
                        
                        # Output detailed loss information in terminal (at certain intervals)
                        if self.global_step % 100 == 0:  # Output every 100 steps
                            logging.info('Step: {}, Detailed Losses:'.format(self.global_step))
                            logging.info('  Classification Loss: {:.6f}'.format(loss_details['cls_loss'].item()))
                            logging.info('  Total Loss: {:.6f}'.format(total_loss.item()))
                            logging.info('  Contrastive Loss: {:.6f}'.format(loss_details['contrastive_loss']))
                            logging.info('    Time Loss: {:.6f}'.format(loss_details['contrastive_time_loss']))
                            logging.info('    Space Loss: {:.6f}'.format(loss_details['contrastive_space_loss']))
                            logging.info('  Center Loss: {:.6f}'.format(loss_details['center_loss'].item()))
                    else:
                        # Compatible with old version, calculate cross-entropy loss
                        ce_loss = self.loss_func(out, y)
                        total_loss = ce_loss + loss_details
                        loss = total_loss  # For subsequent recording
                else:
                    out, _ = self.model(x)
                    # Calculating Loss
                    loss = self.loss_func(out, y)
                    total_loss = loss
            
            # Update weights, use GradScaler for backpropagation
            scaled_loss = self.scaler.scale(total_loss)
            scaled_loss.backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.global_step += 1

            # Calculating Recognition Accuracies
            num_sample += x.size(0)
            reco_top1 = out.max(1)[1]
            num_top1 += reco_top1.eq(y).sum().item()

            # Showing Progress
            lr = self.optimizer.param_groups[0]['lr']
            if self.scalar_writer:
                self.scalar_writer.add_scalar('learning_rate', lr, self.global_step)
                self.scalar_writer.add_scalar('train_loss', loss.item(), self.global_step)
            if self.no_progress_bar:
                logging.info('Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}, LR: {:.4f}'.format(
                    epoch+1, self.max_epoch, num+1, len(self.train_loader), loss.item(), lr
                ))
            else:
                try:
                    train_iter.set_description('Loss: {:.4f}, LR: {:.4f}'.format(loss.item(), lr))
                except:
                    pass

        # Showing Train Results
        train_acc = num_top1 / num_sample
        if self.scalar_writer:
            self.scalar_writer.add_scalar('train_acc', train_acc, self.global_step)
        logging.info('Epoch: {}/{}, Training accuracy: {:d}/{:d}({:.2%}), Training time: {:.2f}s'.format(
            epoch+1, self.max_epoch, num_top1, num_sample, train_acc, time()-start_train_time
        ))
        
        # Output average loss composition for this epoch
        if epoch_loss_info['batch_count'] > 0:
            logging.info('Epoch: {}/{}, Average Loss Composition:'.format(epoch+1, self.max_epoch))
            logging.info('  Classification Loss: {:.6f}'.format(epoch_loss_info['cls_loss'] / epoch_loss_info['batch_count']))
            logging.info('  Total Loss: {:.6f}'.format(epoch_loss_info['total_loss'] / epoch_loss_info['batch_count']))
            logging.info('  Contrastive Loss: {:.6f}'.format(epoch_loss_info['contrastive_loss'] / epoch_loss_info['batch_count']))
            logging.info('    Time Loss: {:.6f}'.format(epoch_loss_info['contrastive_time_loss'] / epoch_loss_info['batch_count']))
            logging.info('    Space Loss: {:.6f}'.format(epoch_loss_info['contrastive_space_loss'] / epoch_loss_info['batch_count']))
            logging.info('  Center Loss: {:.6f}'.format(epoch_loss_info['center_loss'] / epoch_loss_info['batch_count']))
        logging.info('')

    def eval(self):
        self.model.eval()
        start_eval_time = time()
        with torch.no_grad():
            num_top1, num_top5 = 0, 0
            num_sample, eval_loss = 0, []
            cm = np.zeros((self.num_class, self.num_class))
            eval_iter = self.eval_loader if self.no_progress_bar else tqdm(self.eval_loader, dynamic_ncols=True)
            for num, (x, y, _) in enumerate(eval_iter):

                # Using GPU
                x = x.float().to(self.device)
                y = y.long().to(self.device)

                # Use mixed precision for inference
                with autocast():
                    # Calculating Output
                    # Check if model uses ST contrastive loss or ESCL
                    if (hasattr(self.model, 'use_st_contrastive') and self.model.use_st_contrastive) or \
                       (hasattr(self.model, 'use_escl') and self.model.use_escl):
                        out, _, loss_details = self.model(x, y)  # Pass labels for contrastive loss calculation
                        # Calculate cross-entropy loss
                        ce_loss = self.loss_func(out, y)
                        # Total loss = CE loss + contrastive loss
                        if isinstance(loss_details, dict):
                            total_loss = ce_loss + loss_details['total_loss']
                            loss = total_loss  # For subsequent recording
                        else:
                            total_loss = ce_loss + loss_details
                            loss = total_loss  # For subsequent recording
                    else:
                        out, _ = self.model(x)
                        # Getting Loss
                        loss = self.loss_func(out, y)
                    eval_loss.append(loss.item())

                # Calculating Recognition Accuracies
                num_sample += x.size(0)
                reco_top1 = out.max(1)[1]
                num_top1 += reco_top1.eq(y).sum().item()
                # reco_top5 = torch.topk(out,5)[1]
                # num_top5 += sum([y[n] in reco_top5[n,:] for n in range(x.size(0))])

                # Calculating Confusion Matrix
                for i in range(x.size(0)):
                    cm[y[i], reco_top1[i]] += 1

                # Showing Progress
                if self.no_progress_bar and self.args.evaluate:
                    logging.info('Batch: {}/{}'.format(num+1, len(self.eval_loader)))

        # Showing Evaluating Results
        acc_top1 = num_top1 / num_sample
        # acc_top5 = num_top5 / num_sample
        eval_loss = sum(eval_loss) / len(eval_loss)
        eval_time = time() - start_eval_time
        eval_speed = len(self.eval_loader) * self.eval_batch_size / eval_time / len(self.args.gpus)
        logging.info('Top-1 accuracy: {:d}/{:d}({:.2%}), Mean loss:{:.4f}'.format(
            num_top1, num_sample, acc_top1, eval_loss
        ))
        logging.info('Evaluating time: {:.2f}s, Speed: {:.2f} sequnces/(second*GPU)'.format(
            eval_time, eval_speed
        ))
        logging.info('')
        if self.scalar_writer:
            self.scalar_writer.add_scalar('eval_acc', acc_top1, self.global_step)
            self.scalar_writer.add_scalar('eval_loss', eval_loss, self.global_step)

        torch.cuda.empty_cache()
        return acc_top1, None, cm

    def start(self):
        start_time = time()
        if self.args.evaluate:
            if self.args.debug:
                logging.warning('Warning: Using debug setting now!')
                logging.info('')

            # Loading Evaluating Model
            logging.info('Loading evaluating model ...')
            checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)
            if checkpoint and 'model' in checkpoint:
                try:
                    self.model.module.load_state_dict(checkpoint['model'])
                except:
                    self.model.load_state_dict(checkpoint['model'])
            logging.info('Successful!')
            logging.info('')

            # Evaluating
            logging.info('Starting evaluating ...')
            self.eval()
            logging.info('Finish evaluating!')

        else:
            # Resuming
            start_epoch = 0
            best_state = {'acc_top1': 0.0, 'cm': np.zeros((self.num_class, self.num_class))}
            if self.args.resume:
                logging.info('Loading checkpoint ...')
                checkpoint = U.load_checkpoint(self.args.work_dir)
                if checkpoint:
                    try:
                        self.model.module.load_state_dict(checkpoint['model'])
                    except:
                        self.model.load_state_dict(checkpoint['model'])
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                    start_epoch = checkpoint['epoch']
                    best_state.update({'acc_top1': checkpoint['best_state']['acc_top1'], 'cm': checkpoint['best_state']['cm']})
                    self.global_step = start_epoch * len(self.train_loader)
                logging.info('Start epoch: {}'.format(start_epoch+1))
                logging.info('Best accuracy: {:.2%}'.format(best_state['acc_top1']))
                logging.info('Successful!')
                logging.info('')

            # Training
            logging.info('Starting training ...')
            for epoch in range(start_epoch, self.max_epoch):

                # Training
                self.train(epoch)

                # Evaluating
                is_best = False
                # Check if evaluation needs to be skipped
                skip_evaluation = hasattr(self.args, 'skip_evaluation') and self.args.skip_evaluation
                if not skip_evaluation and (epoch+1) % self.eval_interval(epoch) == 0:
                    logging.info('Evaluating for epoch {}/{} ...'.format(epoch+1, self.max_epoch))
                    # Only use top1 accuracy
                    acc_top1, _, cm = self.eval()
                    if acc_top1 > best_state['acc_top1']:
                        is_best = True
                        # Only update top1 accuracy
                        best_state.update({'acc_top1': float(acc_top1), 'cm': cm})

                # Saving Model
                logging.info('Saving model for epoch {}/{} ...'.format(epoch+1, self.max_epoch))
                try:
                    model_state = self.model.module.state_dict()
                except:
                    model_state = self.model.state_dict()
                U.save_checkpoint(
                    model_state, self.optimizer.state_dict(), self.scheduler.state_dict(),
                    epoch+1, best_state, is_best, self.args.work_dir, self.save_dir, self.model_name
                )
                logging.info('Best top-1 accuracy: {:.2%}, Total time: {}'.format(
                    best_state['acc_top1'], U.get_time(time()-start_time)
                ))
                logging.info('')
            logging.info('Finish training!')
            logging.info('')

    def extract(self):
        logging.info('Starting extracting ...')
        if self.args.debug:
            logging.warning('Warning: Using debug setting now!')
            logging.info('')

        # Loading Model
        logging.info('Loading evaluating model ...')
        checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)
        if checkpoint and 'model' in checkpoint:
            self.cm = checkpoint['best_state']['cm']
            try:
                self.model.module.load_state_dict(checkpoint['model'])
            except:
                self.model.load_state_dict(checkpoint['model'])
        logging.info('Successful!')
        logging.info('')