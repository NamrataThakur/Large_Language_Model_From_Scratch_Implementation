import os
import math
os.pardir

from math import ceil

import torch
from gpt_Pretraining.metrics import Metrics

class GPT2_ClassificationFineTune:
    def __init__(self, model, optimizer, device, train_dataLoader, test_dataLoader,
                 num_epochs, eval_batchSize, eval_freq, log_path, gradient_accumulation_steps, global_batch_size,
                 warmup_steps, initial_lr, min_lr, use_warmup, use_gradient_clip , checkpoint, config, pos_token = -1, avg_emb = False, focal= False, alpha = None, gamma = 0.0):

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_dataLoader
        self.val_loader = test_dataLoader
        self.num_epochs = num_epochs
        self.eval_batchSize = eval_batchSize
        self.eval_freq = eval_freq
        self.logger = log_path
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.use_warmup = use_warmup
        self.use_gradient_clip = use_gradient_clip
        self.focal = focal

        #NEW FEAT: Gradient Accumulation And Resume Training:
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.global_batch_size = global_batch_size
        self.checkpoint = checkpoint
        self.config = config

        self.metrics = Metrics(self.model, self.device, alpha=alpha, gamma=gamma, pos_token=pos_token, avg_emb=avg_emb)

    def evaluate_model(self):

        self.model.eval()

        with torch.no_grad():
            train_loss = self.metrics.loss_loader(self.train_loader, self.eval_batchSize, focal=self.focal, classify=True)
            test_loss = self.metrics.loss_loader(self.val_loader, self.eval_batchSize, focal=self.focal, classify=True)

        self.model.train()

        return train_loss, test_loss

    
    def train(self, model_save_path):

        #NEW FEAT: Gradient Accumulation And Resume Training:
        if self.checkpoint is not None:
            train_losses = self.checkpoint['train_losses']
            test_losses = self.checkpoint['test_losses']
            track_tokens_seen = self.checkpoint['track_tokens_seen']
            track_lr = self.checkpoint['track_lr']
            total_steps = self.checkpoint['total_steps']
            global_step = self.checkpoint['global_step']
            max_lr = self.checkpoint['max_lr']
            min_loss = self.checkpoint['validation_loss']
            last_lr = self.optimizer.param_groups[0]['lr']
            start_epoch = self.checkpoint.get('epoch', 0)
            start_batch = self.checkpoint.get('batch', 0)
            torch.set_rng_state(self.checkpoint.get('rng_state', torch.get_rng_state()))
            num_samples = self.checkpoint.get('track_tokens_seen', [0])[-1]

            self.logger.info(f"Model Training Resumed.")
            self.logger.info(f"Best Test Loss recorded : {min_loss}")
            self.logger.info(f"Resuming from Epoch {start_epoch}, Batch {start_batch}, Global Step {global_step}")
            self.logger.info(f'Last Learning Rate : {last_lr}.')
            
        
        else:
            self.logger.info(f"Model Fine-Tuning From SCRATCH..")

            train_losses, test_losses, train_accuracy, val_accuracy, track_tokens_seen, track_lr = [], [], [], [], [], []
            num_samples, global_step = 0, -1
            start_epoch, start_batch =0, 0

            #Get the maximum learning rate as given while defining the optimizer:
            max_lr = self.optimizer.param_groups[0]['lr']
            print('Maximum Learning Rate : ', max_lr)
            self.logger.info(f'Maximum Learning Rate : {max_lr}.')

            min_loss = 10
            print('Default Minimum Loss: ', min_loss)

        #Calculate the total training steps, that will be used for cosine annealing:
        #New Feat: Factor in the gradient accumulation steps
        num_batches_per_epoch = len(self.train_loader)
        updates_per_epoch = math.ceil(num_batches_per_epoch / self.gradient_accumulation_steps)
        total_training_steps = updates_per_epoch * self.num_epochs
        
        #total_training_steps = len(self.train_loader) * self.num_epochs
        print('Total steps to update optimizer : ', total_training_steps)
        self.logger.info(f'Total steps to update optimizer : {total_training_steps}.')
        self.logger.info(f"Total training steps acc. to train loader : {len(self.train_loader)}")

        if self.use_warmup:

            #Get the warmup steps as a ratio of the total steps:
            self.warmup_steps = ceil(total_training_steps * self.warmup_ratio)
            self.logger.info(f'Warmup Steps : {self.warmup_steps}')

            if self.checkpoint is not None:
            
                if self.checkpoint['global_step'] > self.warmup_steps:
                    self.logger.info(f"Global Steps Crossed Warmup Period. Disabling Warmup..!")
            
                else:
                    self.logger.info(f"Global Steps WITHIN Warmup Period. Continuing Warmup..!")
                    
                    #Calculate the learning rate increment during the warmup period:
                    lr_increment = (max_lr - self.initial_lr) / self.warmup_steps
                    print('Learning Rate Increment By : ', lr_increment)
                    self.logger.info(f'Learning Rate Increment By : {lr_increment}.')

            else:
                #Calculate the learning rate increment during the warmup period:
                lr_increment = (max_lr - self.initial_lr) / self.warmup_steps
                print('Learning Rate Increment By : ', lr_increment)
                self.logger.info(f'Learning Rate Increment By : {lr_increment}.')
        
        
        #NEW UPDATE: In case of resume training, keep a check on the overall train steps remaining.
        if self.checkpoint is not None:
            total_training_steps_rem = total_training_steps - self.checkpoint['global_step']
            self.logger.info(f"Global Steps Trained Already : {self.checkpoint['global_step']}")
            self.logger.info(f"Total training steps remaining : {total_training_steps_rem}")
        
        try:
            stop_training = False

            for ep in range(start_epoch, self.num_epochs):

                if stop_training:
                    break

                self.model.train()

                for batch_index, (train_x, train_y) in enumerate(self.train_loader):
                    
                    if stop_training:
                        break

                    # Skip batches we’ve already processed (only once, first resumed epoch)
                    if batch_index < start_batch:
                        continue

                    #Use focal loss if indicated:
                    if self.focal:
                        loss = self.metrics.focal_loss_batch(train_x, train_y)

                    #Else use cross entropy loss:
                    else:
                        loss = self.metrics.classification_loss_batch(train_x, train_y)

                    #New Feat: Gradient Accumulation Process Added
                    loss = loss / self.gradient_accumulation_steps #Scale the loss to account for gradient accumulation

                    loss.backward()

                    #New Feat:
                    gradient_update = ((batch_index + 1) % self.gradient_accumulation_steps == 0) or ((batch_index + 1) == len(self.train_loader))
                    self.optimizer.zero_grad()   # Reset loss gradients from previous batch iteration
                    
                    #New Feat:
                    #Update the gradients if the given accumulation steps have reached or is the last batch of the dataloader:
                    if gradient_update:
                        
                        #NEW UPDATE: Update the global step according to the gradient updates to make warmup and cosine annealing proportional to the weight updates and not dataloader batches
                        global_step += 1

                        #NEW UPDATE: In case of resume training:
                        if self.checkpoint is not None:
                            #NEW UPDATE:  With every update, remaning overall train step to decrease by 1.
                            total_training_steps_rem -= 1

                            #NEW UPDATE: Stop training if remaining overall train step falls to negative. 
                            #Meaning: We have covered all the train steps required, so stop training further.
                            if total_training_steps_rem < 0:
                                self.logger.info("Training complete (resume target reached).")
                                stop_training = True
                                break

                        if self.use_warmup:
                            #Check if the training is still within the warmup stage:
                            if global_step < self.warmup_steps:

                                #Apply linear warmup:
                                lr = self.initial_lr + global_step * lr_increment

                            else:

                                #Training has gone past the warmup period, so apply cosine annealing to bring the learning rate down:
                                total_steps_rem =((global_step - self.warmup_steps) / (total_training_steps - self.warmup_steps) )
                                #Clamping the total remaining steps to 1 to avoid cyclicity in LR updates:
                                total_steps_rem = min(1.0, total_steps_rem)
                                lr = self.min_lr + (max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * total_steps_rem))

                            

                        #If no warmup, then LR should remain same
                        else:
                            #Track the current learning rate:
                            lr = max_lr
                    
                        #Apply the updated learning rate to the optimizer:
                        for param in self.optimizer.param_groups:
                            param['lr'] = lr

                        #Track the current learning rate:
                        track_lr.append(lr)
                    
                        if self.use_gradient_clip:
                            #Apply gradient clipping after warmup period:
                            if global_step > self.warmup_steps:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm= 1.0)

                        #Calculate the weight updates with the modified learning rate and clipped gradient:
                        self.optimizer.step()
                        self.optimizer.zero_grad() # Reset loss gradients from previous batch iteration

                        self.logger.info(f"Gradient updated")
                        self.logger.info(f"Current Learning Rate : {lr}")
                        self.logger.info(f"Global Step : {global_step}")
                        self.logger.info(f"Batch Index : {batch_index+1}")
                    
                    num_samples += train_x.shape[0]

                    #Evaluate the model performance on train and validation datasets:
                    #if global_step % self.eval_freq == 0:
                    if (batch_index + 1) % self.gradient_accumulation_steps == 0:

                        train_loss, test_loss = self.evaluate_model()
                        train_losses.append(train_loss)
                        test_losses.append(test_loss)
                        total_steps.append(global_step)
                        print(f'Epoch No: {ep+1}, Global Step: {global_step:06d}, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}')
                        print(f'Total Samples seen till now: {num_samples}')

                        #Calculate avergae accuracy after each evaluation step:
                        train_accu = self.metrics.accuracy_loader(self.train_loader)
                        val_accu = self.metrics.accuracy_loader(self.val_loader)
                        train_accuracy.append(train_accu)
                        val_accuracy.append(val_accu)
                        
                        #Write the epoch wise metrics in the log file:
                        self.logger.info(f'Epoch No: {ep+1}, Global Step: {global_step:06d}, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}')
                        self.logger.info(f'Total Samples seen till now: {num_samples}\n')
                        track_tokens_seen.append(num_samples)
                        
                        #Check for the model performance improvement:
                        if test_loss < min_loss:
                            
                            min_loss = test_loss
                            torch.save({'model' : self.model.state_dict(),
                                        'optimizer': self.optimizer.state_dict(),
                                        'config' : self.config,
                                        'validation_loss' : test_loss,
                                        'global_step' : global_step,
                                        'learning_rate' : lr,
                                        'total_steps':total_steps,
                                        'train_losses':train_losses,
                                        'test_losses':test_losses,
                                        'track_tokens_seen':track_tokens_seen,
                                        'track_lr':track_lr,
                                        'max_lr':max_lr,
                                        'epoch': ep,
                                        'batch': batch_index + 1,  # resume from next batch
                                        'rng_state': torch.get_rng_state(),
                                        }, 
                                        model_save_path)
                            self.logger.info(f"BEST model SAVED on iteration {global_step:06d} to {model_save_path}..! ")
                        
                        
                        print(f"Training accuracy: {train_accu*100:.2f}%")
                        print(f"Validation accuracy: {val_accu*100:.2f}%")

                        #Write the epoch wise metrics in the log file:
                        self.logger.info(f"Training accuracy: {train_accu*100:.2f}%")
                        self.logger.info(f"Validation accuracy: {val_accu*100:.2f}%")
                        

                #Calculate avergae accuracy after each epoch:
                train_accu = self.metrics.accuracy_loader(self.train_loader)
                val_accu = self.metrics.accuracy_loader(self.val_loader)
                print(f"EP: {ep+1}, Training accuracy : {train_accu*100:.2f}%")
                print(f"EP: {ep+1}, Validation accuracy: {val_accu*100:.2f}%")

                #Write the epoch wise metrics in the log file:
                self.logger.info(f"EP: {ep+1}, Training accuracy: {train_accu*100:.2f}%")
                self.logger.info(f"EP: {ep+1}, Validation accuracy: {val_accu*100:.2f}%")

        
        except KeyboardInterrupt:

            #Save the model in case of an exception
            torch.save({'model' : self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'config' : self.config,
                        'validation_loss' : test_losses[-1] if len(test_losses) > 0 else 10,
                        'global_step' : global_step,
                        'learning_rate' : lr,
                        'total_steps':total_steps,
                        'train_losses':train_losses,
                        'test_losses':test_losses,
                        'track_tokens_seen':track_tokens_seen,
                        'track_lr':track_lr,
                        'max_lr':max_lr,
                        'epoch': ep,
                        'batch': batch_index + 1,  # resume from next batch
                        'rng_state': torch.get_rng_state(),
                        }, 
                        model_save_path)
            
            self.logger.info(f"BEST model SAVED on iteration {global_step:06d} to {model_save_path}..! ")
            
        return train_losses, test_losses, train_accuracy, val_accuracy, num_samples, track_lr

        
