import os
import math
os.pardir

import torch
from gpt_Pretraining.metrics import Metrics
from gpt_Pretraining.text_generation import Text_Generation

class GPT2_PreferenceFineTune:
    def __init__(self, policy_model, reference_model, optimizer, device, train_dataLoader, test_dataLoader,
                 num_epochs, eval_batchSize, eval_freq, log_path, start_context, max_new_tokens, beta, 
                 warmup_steps, initial_lr, min_lr, use_warmup, use_gradient_clip ):
        
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_dataLoader
        self.val_loader = test_dataLoader
        self.num_epochs = num_epochs
        self.eval_batchSize = eval_batchSize
        self.eval_freq = eval_freq
        self.beta = beta
        self.start_context = start_context
        self.max_new_tokens = max_new_tokens
        self.logger = log_path
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.use_warmup = use_warmup
        self.use_gradient_clip = use_gradient_clip

        self.metrics = Metrics(model= self.policy_model, device= self.device, reference_model= self.reference_model)
        self.generation = Text_Generation(self.policy_model, self.device, 'gpt2')

    def evaluate_model(self):

        self.policy_model.eval()
        result = {}

        with torch.no_grad():
            train_loss, train_rewards_correct, train_rewards_wrong = self.metrics.preference_loss_loader(dataloader = self.train_loader, 
                                                                                                        beta= self.beta, 
                                                                                                        num_batches = self.eval_batchSize)
            
            val_loss, val_rewards_correct, val_rewards_wrong = self.metrics.preference_loss_loader(dataloader = self.val_loader,
                                                                                                    beta = self.beta, 
                                                                                                    num_batches = self.eval_batchSize)

        result = {
            'train_loss' : train_loss,
            'train_rewards_correct' : train_rewards_correct,
            'train_rewards_wrong' : train_rewards_wrong,
            'val_loss' : val_loss,
            'val_rewards_correct' : val_rewards_correct,
            'val_rewards_wrong' : val_rewards_wrong,
        }
        
        self.policy_model.train()

        return result 
        

    def train(self, model_save_path, temp=0.0, top_k= None, eos_id = None):

        tracking = {
            'train_loss' : [],
            'train_rewards_correct' : [],
            'train_rewards_wrong' : [],
            'val_loss' : [],
            'val_rewards_correct' : [],
            'val_rewards_wrong' : [],
            'tokens_seen' : []
        }

        tokens_tracked, global_step = 0, -1
        track_lr = []

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
        print('Default Minimum Loss: ', min_loss)

        for epoch in range(self.num_epochs):
            
            self.policy_model.train()

            for i, batch in enumerate(self.train_loader):

                self.optimizer.zero_grad() # Reset loss gradients from previous batch iteration
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

                loss, rewards_correct, rewards_wrong = self.metrics.preference_loss_batch(batch = batch, beta = self.beta)
                
                loss.backward()

                if self.use_gradient_clip:
                    #Apply gradient clipping after warmup period:
                    if global_step > self.warmup_steps:
                        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm= 1.0)
                        #torch.nn.utils.clip_grad_norm_(self.reference_model.parameters(), max_norm= 1.0)

                
                #Calculate the weight updates with the modified learning rate and clipped gradient:
                self.optimizer.step()

                tokens_tracked += batch['correct_response'].numel()
                
                #Evaluate the model performance on train and validation datasets:
                if (global_step % self.eval_freq) == 0:

                    result = self.evaluate_model()
                    

                    tracking['train_loss'].append(result['train_loss'])
                    tracking['train_rewards_correct'].append(result['train_rewards_correct'])
                    tracking['train_rewards_wrong'].append(result['train_rewards_wrong'])
                    tracking['val_loss'].append(result['val_loss'])
                    tracking['val_rewards_correct'].append(result['val_rewards_correct'])
                    tracking['val_rewards_wrong'].append(result['val_rewards_wrong'])
                    tracking['tokens_seen'].append(tokens_tracked)

                    train_reward_margin = result['train_rewards_correct'] - result['train_rewards_wrong']
                    val_reward_margin = result['val_rewards_correct'] - result['val_rewards_wrong']

                    print(
                        f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {result['train_loss']:.3f}, Val loss {result['val_loss']:.3f}, "
                        f"Train reward margins {train_reward_margin:.3f}, "
                        f"Val reward margins {val_reward_margin:.3f}"
                    )
                    #Write the epoch wise metrics in the log file:
                    self.logger.info(f"Epoch No: {epoch+1}, Step: {global_step:06d}, Train Loss: {result['train_loss']:.3f}, Val Loss: {result['val_loss']:.3f} "
                                        f"Train reward margins {train_reward_margin:.3f}, Val reward margins {val_reward_margin:.3f}")
                    

                    #Check for the model performance improvement:
                    if result['val_loss'] < min_loss:
                        
                        min_loss = result['val_loss']
                        torch.save({'model' : self.policy_model.state_dict(),
                                    'optimizer': self.optimizer.state_dict()
                                    }, 
                                    model_save_path)
                        self.logger.info(f"BEST model SAVED on iteration {global_step:06d} to {model_save_path}..! ")

            #Write the model generated response after each epoch:
            output_text = self.generation.text_generation(input_text = self.start_context, max_new_tokens = self.max_new_tokens, 
                                                            temp = temp, top_k = top_k, eos_id = eos_id)
            print(output_text)
            self.logger.info(output_text)


        return tracking, track_lr