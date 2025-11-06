import torch
import math
from .text_generation import Text_Generation
from .metrics import Metrics
from math import ceil

class GPT2_PreTrain:
    def __init__(self, model, optimizer, device, train_dataLoader, test_dataLoader,num_epochs, gradient_accumulation_steps, global_batch_size, eval_batchSize, eval_freq, 
                 start_context,max_new_tokens, log_path, warmup_steps, initial_lr, min_lr, use_warmup, use_gradient_clip, kv_cache = False,
                 arch_type='original'):

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_dataLoader
        self.val_loader = test_dataLoader
        self.num_epochs = num_epochs
        self.eval_batchSize = eval_batchSize
        self.eval_freq = eval_freq
        self.start_context = start_context
        self.max_new_tokens = max_new_tokens
        self.logger = log_path
        self.warmup_ratio = warmup_steps
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.use_warmup = use_warmup
        self.use_gradient_clip = use_gradient_clip
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.global_batch_size = global_batch_size
        self.kv_cache = kv_cache
        self.arch_type = arch_type

        self.generation = Text_Generation(model=self.model, device=self.device, tokenizer_model='gpt2', 
                                          arch_type=self.arch_type)
        self.metrics = Metrics(self.model, self.device)

    def evaluate_model(self):

        self.model.eval()

        with torch.no_grad():
            train_loss = self.metrics.loss_loader(self.train_loader, self.eval_batchSize,classify=False)
            test_loss = self.metrics.loss_loader(self.val_loader, self.eval_batchSize,classify=False)

        self.model.train()

        return train_loss, test_loss

    
    def train(self, model_save_path, temp=0.0, top_k= None, eos_id = None ):

        train_losses, test_losses, track_tokens_seen, track_lr, total_steps  = [], [], [], [], []

        tokens_seen, global_step = 0, -1
        
        #Get the maximum learning rate as given while defining the optimizer:
        max_lr = self.optimizer.param_groups[0]['lr']
        print('Maximum Learning Rate : ', max_lr)
        self.logger.info(f'Maximum Learning Rate : {max_lr}.')

        #Calculate the total training steps, that will be used for cosine annealing: 
        #New Feat: Factor in the gradient accumulation steps
        num_batches_per_epoch = len(self.train_loader)
        updates_per_epoch = math.ceil(num_batches_per_epoch / self.gradient_accumulation_steps)
        total_training_steps = updates_per_epoch * self.num_epochs
        
        #total_training_steps = ceil((len(self.train_loader) * self.num_epochs) / self.gradient_accumulation_steps)
        print('Total steps to update optimizer : ', total_training_steps)
        self.logger.info(f'Total steps to update optimizer : {total_training_steps}.')
        self.logger.info(f"Total training steps acc. to train loader : {len(self.train_loader)}")

        if self.use_warmup:
            
            #Get the warmup steps as a ratio of the total steps:
            self.warmup_steps = ceil(total_training_steps * self.warmup_ratio)
            self.logger.info(f'Warmup Steps : {self.warmup_steps}')

            #Calculate the learning rate increment during the warmup period:
            lr_increment = (max_lr - self.initial_lr) / self.warmup_steps
            print('Learning Rate Increment By : ', lr_increment)
            self.logger.info(f'Learning Rate Increment By : {lr_increment}.')

        min_loss = 10
        print('Default Minimum Loss: ', min_loss)

        try:

            for ep in range(self.num_epochs):

                self.model.train()

                for batch_index, (train_x, train_y) in enumerate(self.train_loader):

                    # global_step += 1

                    loss = self.metrics.loss_batch(train_x, train_y)

                    #New Feat: Gradient Accumulation Process Added
                    loss = loss / self.gradient_accumulation_steps #Scale the loss to account for gradient accumulation

                    loss.backward()

                    #New Feat:
                    gradient_update = ((batch_index + 1) % self.gradient_accumulation_steps == 0) or ((batch_index + 1) == len(self.train_loader))

                    #New Feat:
                    #Update the gradients if the given accumulation steps have reached or is the last batch of the dataloader:
                    if gradient_update:
                        
                        #NEW UPDATE: Update the global step according to the gradient updates to make warmup and cosine annealing proportional to the weight updates and not dataloader batches
                        global_step += 1

                        if self.use_warmup:
                            #Check if the training is still within the warmup stage:
                            if global_step <= self.warmup_steps:

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
                            if global_step >= self.warmup_steps:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm= 1.0)

                       
                        #Calculate the weight updates with the modified learning rate and clipped gradient:
                        self.optimizer.step()
                        self.optimizer.zero_grad() # Reset loss gradients from previous batch iteration

                        self.logger.info(f"Gradient updated")
                        self.logger.info(f"Current Learning Rate : {lr}")
                        self.logger.info(f"Global Step : {global_step}")
                        self.logger.info(f"Batch Index : {batch_index+1}")
                    
                    tokens_seen += train_x.numel()
                    
                    #Evaluate the model performance on train and validation datasets:
                    #if (global_step > 0) and (global_step % self.eval_freq == 0):
                    if (batch_index + 1) % self.gradient_accumulation_steps == 0:

                        train_loss, test_loss = self.evaluate_model()
                        train_losses.append(train_loss)
                        test_losses.append(test_loss)
                        total_steps.append(global_step)
                        print(f'Epoch No: {ep+1}, Step: {global_step:06d}, Train Loss: {train_loss:.3f}, Val Loss: {test_loss:.3f}')
                        print(f'Total Tokens seen till now: {tokens_seen}')

                        #Write the epoch wise metrics in the log file:
                        self.logger.info(f'Epoch No: {ep+1}, Global Step: {global_step:06d}, Train Loss: {train_loss:.3f}, Val Loss: {test_loss:.3f}\n')
                        self.logger.info(f'Total Tokens seen till now: {tokens_seen}\n')
                        track_tokens_seen.append(tokens_seen)

                        #Check for the model performance improvement:
                        if test_loss < min_loss:

                            min_loss = test_loss
                            torch.save({'model' : self.model.state_dict(),
                                        'optimizer': self.optimizer.state_dict()
                                        }, 
                                        model_save_path)
                            self.logger.info(f"BEST model SAVED on iteration {global_step:06d} to {model_save_path}..! ")

                        
                        #Write the model generated response after each evaluation step:
                        gen_output = self.generation.text_generation(self.start_context, self.max_new_tokens, temp, top_k, eos_id, self.kv_cache)
                        print(gen_output)
                        self.logger.info(gen_output)


                #Write the model generated response after each epoch:
                gen_output = self.generation.text_generation(self.start_context, self.max_new_tokens, temp, top_k, eos_id, self.kv_cache)
                print(gen_output)
                self.logger.info(gen_output)

        except KeyboardInterrupt:

            #Save the model in case of an exception
            torch.save({'model' : self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                        }, 
                        model_save_path)
            
            self.logger.info(f"BEST model SAVED on iteration {global_step:06d} to {model_save_path}..! ")
            

        return train_losses, test_losses, track_tokens_seen, track_lr, total_steps

        
