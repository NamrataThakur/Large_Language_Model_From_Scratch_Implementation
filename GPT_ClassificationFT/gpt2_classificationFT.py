import os
import math
os.pardir

import torch
from gpt_Pretraining.metrics import Metrics

class GPT2_ClassificationFineTune:
    def __init__(self, model, optimizer, device, train_dataLoader, test_dataLoader,
                 num_epochs, eval_batchSize, eval_freq, log_path,
                 warmup_steps, initial_lr, min_lr, use_warmup, use_gradient_clip ):

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

        self.metrics = Metrics(self.model, self.device)

    def evaluate_model(self):

        self.model.eval()

        with torch.no_grad():
            train_loss = self.metrics.loss_loader(self.train_loader, self.eval_batchSize,classify=True)
            test_loss = self.metrics.loss_loader(self.val_loader, self.eval_batchSize,classify=True)

        self.model.train()

        return train_loss, test_loss

    
    def train(self, model_save_path):

        train_losses, test_losses, train_accuracy, val_accuracy, track_lr = [], [], [], [], []

        num_samples, global_step = 0, -1

        #Get the maximum learning rate as given while defining the optimizer:
        max_lr = self.optimizer.param_groups[0]['lr']
        print('Maximum Learning Rate : ', max_lr)
        self.logger.info(f'Maximum Learning Rate : {max_lr}.')

        #Calculate the total training steps, that will be used for cosine annealing:
        total_training_steps = len(self.train_loader) * self.num_epochs
        print('Total training steps : ', total_training_steps)
        self.logger.info(f'Total training steps : {total_training_steps}.')

        if self.use_warmup:
            #Calculate the learning rate increment during the warmup period:
            lr_increment = (max_lr - self.initial_lr) / self.warmup_steps
            print('Learning Rate Increment By : ', lr_increment)
            self.logger.info(f'Learning Rate Increment By : {lr_increment}.')

        min_loss = 10
        print('Initial Loss: ', min_loss)

        for ep in range(self.num_epochs):

            self.model.train()

            for train_x, train_y in self.train_loader:

                self.optimizer.zero_grad()   # Reset loss gradients from previous batch iteration
                global_step += 1

                if self.use_warmup:
                    #Check if the training is still within the warmup stage:
                    if global_step < self.warmup_steps:

                        #Apply linear warmup:
                        lr = self.initial_lr + global_step * lr_increment

                    else:

                        #Training has gone past the warmup period, so apply cosine annealing to bring the learning rate down:
                        total_steps_rem =((global_step - self.warmup_steps) / (total_training_steps - self.warmup_steps) )
                        lr = self.min_lr + (max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * total_steps_rem))

                    #Apply the updated learning rate to the optimizer:
                    for param in self.optimizer.param_groups:
                        param['lr'] = lr

                    #Track the current learning rate:
                    track_lr.append(lr)


                loss = self.metrics.classification_loss_batch(train_x, train_y)
                loss.backward()
                
                if self.use_gradient_clip:
                    #Apply gradient clipping after warmup period:
                    if global_step > self.warmup_steps:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm= 1.0)

                #Calculate the weight updates with the modified learning rate and clipped gradient:
                self.optimizer.step()
                
                num_samples += train_x.shape[0]

                #Evaluate the model performance on train and validation datasets:
                if global_step % self.eval_freq == 0:

                    train_loss, test_loss = self.evaluate_model()
                    train_losses.append(train_loss)
                    test_losses.append(test_loss)
                    print(f'Epoch No: {ep+1}, Step: {global_step:06d}, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}')

                    #Write the epoch wise metrics in the log file:
                    self.logger.info(f'Epoch No: {ep+1}, Step: {global_step:06d}, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}')

                    #Check for the model performance improvement:
                    if test_loss < min_loss:
                        
                        min_loss = test_loss
                        torch.save({'model' : self.model.state_dict(),
                                    'optimizer': self.optimizer.state_dict()
                                    }, 
                                    model_save_path)
                        self.logger.info(f"BEST model SAVED on iteration {global_step:06d} to {model_save_path}..! ")
                    
            #Calculate avergae accuracy for each epoch:
            train_accu = self.metrics.accuracy_loader(self.train_loader)
            val_accu = self.metrics.accuracy_loader(self.val_loader)
            print(f"Training accuracy: {train_accu*100:.2f}%")
            print(f"Validation accuracy: {val_accu*100:.2f}%")

            #Write the epoch wise metrics in the log file:
            self.logger.info(f"Training accuracy: {train_accu*100:.2f}%")
            self.logger.info(f"Validation accuracy: {val_accu*100:.2f}%")

            train_accuracy.append(train_accu)
            val_accuracy.append(val_accu)

        
        return train_losses, test_losses, train_accuracy, val_accuracy, num_samples, track_lr

        
