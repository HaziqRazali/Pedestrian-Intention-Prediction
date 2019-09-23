"""Train a pifpaf net."""

import copy
import hashlib
import logging
import shutil
import time
import torch
import itertools
torch.set_printoptions(precision=10)

class Trainer(object):
    def __init__(self, model, loss, optimizer, out, *,
                 lr_scheduler=None,
                 log_interval=1,
                 device=None,
                 fix_batch_norm=False,
                 stride_apply=1,
                 ema_decay=None,
                 encoder_visualizer=None,
                 train_profile=None,
                 model_meta_data=None):
        self.log = logging.getLogger(self.__class__.__name__)

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.out = out
        self.lr_scheduler = lr_scheduler

        self.log_interval = log_interval
        self.device = device
        self.fix_batch_norm = fix_batch_norm
        self.stride_apply = stride_apply

        self.ema_decay = ema_decay
        self.ema = None
        self.ema_restore_params = None

        self.encoder_visualizer = encoder_visualizer
        self.model_meta_data = model_meta_data

        if train_profile:
            # monkey patch to profile self.train_batch()
            self.train_batch_without_profile = self.train_batch
            def train_batch_with_profile(*args, **kwargs):
                with torch.autograd.profiler.profile() as prof:
                    result = self.train_batch_without_profile(*args, **kwargs)
                print(prof.key_averages())
                print(prof.total_average())
                prof.export_chrome_trace(train_profile)
                return result
            self.train_batch = train_batch_with_profile

    def lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def step_ema(self):
        if self.ema is None:
            return

        for p, ema_p in zip(self.model.parameters(), self.ema):
            ema_p.mul_(1.0 - self.ema_decay).add_(self.ema_decay, p.data)

    def apply_ema(self):
        if self.ema is None:
            return

        self.log.info('applying ema')
        self.ema_restore_params = copy.deepcopy(
            [p.data for p in self.model.parameters()])
        for p, ema_p in zip(self.model.parameters(), self.ema):
            p.data.copy_(ema_p)

    def ema_restore(self):
        if self.ema_restore_params is None:
            return

        self.log.info('restoring params from before ema')
        for p, ema_p in zip(self.model.parameters(), self.ema_restore_params):
            p.data.copy_(ema_p)
        self.ema_restore_params = None

    def loop(self, train_scenes, val_scenes, epochs, start_epoch=0):
        for _ in range(start_epoch):
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        for epoch in range(start_epoch, epochs):
            self.train(train_scenes, epoch)

            self.write_model(epoch + 1, epoch == epochs - 1)
            self.val(val_scenes, epoch + 1)

    def train_batch(self, data1, targets1, meta1, data2, targets2, meta2, apply_gradients=True):  # pylint: disable=method-hidden
        
        if self.encoder_visualizer:
            self.encoder_visualizer(data1, targets1, meta1)

        if self.device:
            data1 = data1.to(self.device, non_blocking=True)
            targets1 = [[t.to(self.device, non_blocking=True) for t in head] for head in targets1]            
            data2 = data2.to(self.device, non_blocking=True)
            targets2 = [[t.to(self.device, non_blocking=True) for t in head] for head in targets2]

        if(1):
            outputs1 = self.model(data1, head="pifpaf")               
            loss1, head_losses1 = self.loss(outputs1, targets1, head="pifpaf")  
            if loss1 is not None:
                loss1.backward()
            if apply_gradients:
                self.optimizer.step()
                self.optimizer.zero_grad()  
                self.step_ema()   
            
            outputs2 = self.model(data2, head="crm")
            loss2, head_losses2 = self.loss(outputs2, targets2, head="crm")  
            if loss2 is not None:
                loss2.backward()
            if apply_gradients:
                self.optimizer.step()
                self.optimizer.zero_grad() 
                self.step_ema()

            outputs3 = self.model(data1, head="pifpaf")               
            loss3, head_losses3 = self.loss(outputs3, targets1, head="pifpaf")  
            if loss3 is not None:
                loss3.backward()
            if apply_gradients:
                self.optimizer.step()
                self.optimizer.zero_grad()  
                self.step_ema()
               
            loss = loss3 + loss2
            head_losses = head_losses3 + head_losses2      
        
        if(0):
            outputs1 = self.model(data1, head="pifpaf")               
            outputs2 = self.model(data2, head="crm")
            loss1, head_losses1 = self.loss(outputs1, targets1, head="pifpaf")
            loss2, head_losses2 = self.loss(outputs2, targets2, head="crm")  
            loss = loss1 + loss2
            head_losses = head_losses1 + head_losses2
            if loss is not None:
               loss.backward()
            if apply_gradients:
               self.optimizer.step()
               self.optimizer.zero_grad()
               self.step_ema()

        return (
            float(loss.item()) if loss is not None else None,
            [float(l.item()) if l is not None else None
             for l in head_losses],
        )

    def val_batch(self, data, targets, head):
        if self.device:
            data = data.to(self.device, non_blocking=True)
            targets = [[t.to(self.device, non_blocking=True) for t in head] for head in targets]

        with torch.no_grad():        
            outputs = self.model(data, head=head)
            loss, head_losses = self.loss(outputs, targets, head=head)              
            #outputs1 = self.model(data1, head="pifpaf")
            #loss1, head_losses1 = self.loss(outputs1, targets1, head="pifpaf")
            #outputs2 = self.model(data2, head="crm")
            #loss2, head_losses2 = self.loss(outputs2, targets2, head="crm")    
            #loss = loss1 + loss2
            #head_losses = head_losses1 + head_losses2
        
        return (
            float(loss.item()) if loss is not None else None,
            [float(l.item()) if l is not None else None
             for l in head_losses],
        )

    def train(self, scenes, epoch):
        
        #print(len(scenes[0]))
        #sys.exit(0)
        
        start_time = time.time()
        self.model.train()
        if self.fix_batch_norm:
            for m in self.model.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    # print('fixing parameters for {}. Min var = {}'.format(
                    #     m, torch.min(m.running_var)))
                    m.eval()
                    # m.weight.requires_grad = False
                    # m.bias.requires_grad = False

                    # avoid numerical instabilities
                    # (only seen sometimes when training with GPU)
                    # Variances in pretrained models can be as low as 1e-17.
                    # m.running_var.clamp_(min=1e-8)
                    m.eps = 1e-4
        self.ema_restore()
        self.ema = None

        epoch_loss = 0.0
        head_epoch_losses = None
        last_batch_end = time.time()
        self.optimizer.zero_grad()
                
        #jaad has a larger number of iterations per epoch so we cycle over coco (scenes[0])       
        #for batch_idx, ((data1, target1, meta1),(data2, target2, meta2)) in enumerate(zip(itertools.cycle(scenes[0]), scenes[1])):       
        for batch_idx, ((data1, target1, meta1),(data2, target2, meta2)) in enumerate(zip(scenes[0],scenes[1])):
                        
            preprocess_time = time.time() - last_batch_end

            batch_start = time.time()
            apply_gradients = batch_idx % self.stride_apply == 0
            loss, head_losses = self.train_batch(data1, target1, meta1, data2, target2, meta2, apply_gradients)

            # update epoch accumulates
            if loss is not None:
                epoch_loss += loss
            if head_epoch_losses is None:
                head_epoch_losses = [0.0 for _ in head_losses]
            for i, head_loss in enumerate(head_losses):
                if head_loss is None:
                    continue
                head_epoch_losses[i] += head_loss

            batch_time = time.time() - batch_start

            # write training loss
            if batch_idx % self.log_interval == 0:
                self.log.info({
                    'type': 'train',
                    'epoch': epoch, 'batch': batch_idx, 'n_batches': len(scenes[0]),
                    'time': round(batch_time, 3),
                    'data_time': round(preprocess_time, 3),
                    'lr': self.lr(),
                    'loss': round(loss, 3) if loss is not None else None,
                    'head_losses': [round(l, 3) if l is not None else None
                                    for l in head_losses],
                })

            # initialize ema
            if self.ema is None and self.ema_decay:
                self.ema = copy.deepcopy([p.data for p in self.model.parameters()])

            last_batch_end = time.time()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.apply_ema()
        self.log.info({
            'type': 'train-epoch',
            'epoch': epoch + 1,
            'loss': round(epoch_loss / len(scenes[0]), 5),
            'head_losses': [round(l / len(scenes[0]), 5) for l in head_epoch_losses],
            'time': round(time.time() - start_time, 1),
        })

    def val(self, scenes, epoch):
        
        #print(len(scenes[0]), len(scenes[1]))
        #sys.exit(0)
        
        start_time = time.time()
        self.model.eval()

        epoch_loss1 = 0.0
        epoch_loss2 = 0.0
        head_epoch_losses_pifpaf = None
        head_epoch_losses_crm    = None
        
        # cannot cycle when validating
        for data1, target1, _ in scenes[0]:
            loss, head_losses = self.val_batch(data1, target1, head="pifpaf")

            # update epoch accumulates
            if loss is not None:
                epoch_loss1 += loss
            if head_epoch_losses_pifpaf is None:
                head_epoch_losses_pifpaf = [0.0 for _ in head_losses]
            for i, head_loss in enumerate(head_losses):
                if head_loss is None:
                    continue
                head_epoch_losses_pifpaf[i] += head_loss  
                
        # cannot cycle when validating
        for data2, target2, _ in scenes[1]:
            loss, head_losses = self.val_batch(data2, target2, head="crm")

            # update epoch accumulates
            if loss is not None:
                epoch_loss2 += loss
            if head_epoch_losses_crm is None:
                head_epoch_losses_crm = [0.0 for _ in head_losses]
            for i, head_loss in enumerate(head_losses):
                if head_loss is None:
                    continue
                head_epoch_losses_crm[i] += head_loss           
         
        eval_time = time.time() - start_time

        self.log.info({
            'type': 'val-epoch',
            'epoch': epoch,
            'loss': round(epoch_loss1/len(scenes[0]) + epoch_loss2/len(scenes[1]), 5),
            'head_losses': [round(l / len(scenes[0]), 5) for l in head_epoch_losses_pifpaf] + [round(l / len(scenes[1]), 5) for l in head_epoch_losses_crm],
            'time': round(eval_time, 1),
        })

    def write_model(self, epoch, final=True):
        self.model.cpu()

        if isinstance(self.model, torch.nn.DataParallel):
            self.log.debug('Writing a dataparallel model.')
            model = self.model.module
        else:
            self.log.debug('Writing a single-thread model.')
            model = self.model

        filename = '{}.epoch{:03d}'.format(self.out, epoch)
        self.log.debug('about to write model')
        torch.save({
            'model': model,
            'epoch': epoch,
            'meta': self.model_meta_data,
        }, filename)
        self.log.debug('model written')

        if final:
            sha256_hash = hashlib.sha256()
            with open(filename, 'rb') as f:
                for byte_block in iter(lambda: f.read(8192), b''):
                    sha256_hash.update(byte_block)
            file_hash = sha256_hash.hexdigest()
            outname, _, outext = self.out.rpartition('.')
            final_filename = '{}-{}.{}'.format(outname, file_hash[:8], outext)
            shutil.copyfile(filename, final_filename)

        self.model.to(self.device)
