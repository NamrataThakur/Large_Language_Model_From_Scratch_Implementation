import torch
import torch.nn.functional as F

class Metrics:
    def __init__(self, model, device, alpha = None, gamma = 0.0, pos_token=-1, avg_emb=False, reference_model = None ):
        
        self.model = model
        self.device = device
        self.reference_model = reference_model
        
        #Following parameters are used in Focal Loss computation in imbalanced supervised classification fine-tuning task:
        self.alpha = alpha
        self.gamma = gamma

        #NEW FEAT: Average Embedding and POS Token Inclusion: USED FOR CLASSIFICATION SFT
        self.pos_token = pos_token
        self.avg_emb = avg_emb

    def loss_loader(self, dataloader, num_batches=None, classify=False, focal=False):

        batch_loss = 0.0
        if len(dataloader) == 0:
            print('No batches/data in the dataloader..!')
        elif num_batches is None:
            num_batches = len(dataloader)
        else:
            num_batches = min(num_batches,len(dataloader))
            
        for i, (input_batch, target_batch) in enumerate(dataloader):

            if i < num_batches:
                if classify:
                    if focal:
                        ce_loss = self.focal_loss_batch(input_batch=input_batch, target_batch=target_batch)
                    else:
                        ce_loss = self.classification_loss_batch(input_batch, target_batch)
                    
                else:
                    ce_loss = self.loss_batch(input_batch, target_batch)
                batch_loss += ce_loss.item()
            else:
                break

        avg_batch_loss = batch_loss / num_batches
        return avg_batch_loss
    
    def loss_batch(self,input_batch, target_batch ):
        input_batch, target_batch = input_batch.to(self.device), target_batch.to(self.device)
        output_logits = self.model(input_batch)
        ce_loss = torch.nn.functional.cross_entropy(output_logits.flatten(0,1), target_batch.flatten())
        return ce_loss

    @torch.no_grad()  # Disable gradient tracking for efficiency
    def accuracy_loader(self, dataloader, num_batches=None):

        correct_pred, num_samples = 0, 0
        self.model.eval()

        if num_batches is None:
            num_batches = len(dataloader)
        else:
            num_batches = min(num_batches, len(dataloader))
        
        for i, (X_batch, Y_batch) in enumerate(dataloader):

            if i < num_batches:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)

                with torch.no_grad():
                    output_logits = self.model(X_batch)
                
                #last_idx_logits = output_logits[:, -1, :]

                #Selecting the row corresponding to the token position or taking average of all tokens for all batch:
                #NEW FEAT: Average Embedding and POS Token Inclusion:
                if self.avg_emb:
                    last_idx_logits = output_logits.mean(dim=1)
                else:
                    last_idx_logits = output_logits[:, self.pos_token, :]

                predictions = torch.argmax(last_idx_logits, dim=-1)

                num_samples += predictions.shape[0]

                correct_pred += (predictions == Y_batch).sum().item()

            else:
                break
        avg_accuracy = correct_pred / num_samples
        return avg_accuracy
    
    #Function used in classification supervised fine-tuning task:
    def classification_loss_batch(self, input_batch, target_batch):

        input_batch, target_batch = input_batch.to(self.device), target_batch.to(self.device)
        output_logits = self.model(input_batch)

        #NEW FEAT: Average Embedding and POS Token Inclusion:
        if self.avg_emb:
            last_idx_logits = output_logits.mean(dim=1)
        else:
            #last_idx_logits = output_logits[:, -1, :]
            last_idx_logits = output_logits[:, self.pos_token, :]
            
        classify_loss = torch.nn.functional.cross_entropy(last_idx_logits, target_batch)
        return classify_loss
    

    #Function used in imbalanced classification supervised fine-tuning task:
    def focal_loss_batch(self, input_batch, target_batch):

        input_batch, target_batch = input_batch.to(self.device), target_batch.to(self.device)
        output_logits = self.model(input_batch)

        #Selecting the row corresponding to the last token for all batch:
        #Shape: (B, num_tokens, logits) --> (B, logits)
        #last_idx_logits = output_logits[:, -1, :]

        #Selecting the row corresponding to the token position or taking average of all tokens for all batch:
        #NEW FEAT:  Average Embedding and POS Token Inclusion:
        if self.avg_emb:
            last_idx_logits = output_logits.mean(dim=1)
        else:
            last_idx_logits = output_logits[:, self.pos_token, :]

        #Computed the weighted cross entropy term: - alpha * log(pt)
        ce_loss = torch.nn.functional.cross_entropy(input=last_idx_logits, target=target_batch, weight=self.alpha, reduction='none')

        #Computing the focal part: (1 - p) ^ gamma
        #Shape: (B, logits) --> (B, class_probs)
        log_probs = F.log_softmax(last_idx_logits, dim=-1)

        all_rows = torch.arange(last_idx_logits.size(0))

        #Shape: (B, class_probs) --> (B)
        log_probs_target = log_probs[all_rows, target_batch]

        probs_target = log_probs_target.exp()
        focal_term = (1 - probs_target) ** self.gamma

        #Focal loss : - alpha * [(1 - pt) ^ gamma] * log(pt)
        focal_loss =  focal_term * ce_loss

        loss = focal_loss.mean()

        return loss


    #Function used in preference fine-tuning task:
    def preference_loss_batch(self, batch, beta):
        
        policy_correct_response_probs = self.get_log_probability(inputs = batch['correct_response'], mask = batch['correct_response_mask'], model_type = None)

        policy_wrong_response_probs = self.get_log_probability(inputs = batch['wrong_response'], mask = batch['wrong_response_mask'], model_type = None)
        
        reference_correct_response_probs = self.get_log_probability(inputs = batch['correct_response'], mask = batch['correct_response_mask'], model_type = 'reference')

        reference_wrong_response_probs = self.get_log_probability(inputs = batch['wrong_response'], mask = batch['wrong_response_mask'], model_type = 'reference')

        dpo_loss, rewards_correct_response, rewards_wrong_response = self.dpo_loss_batch(policy_correct_probs = policy_correct_response_probs, 
                                                                                         policy_wrong_probs = policy_wrong_response_probs, 
                                                                                         reference_correct_probs = reference_correct_response_probs, 
                                                                                         reference_wrong_probs = reference_wrong_response_probs, 
                                                                                         beta = beta)
        
        return dpo_loss, rewards_correct_response, rewards_wrong_response
    

    #Loss loader function for DPO Loss:
    def preference_loss_loader(self, dataloader, beta, num_batches = None):

        losses, rewards_correct, rewards_wrong = 0.0, 0.0, 0.0

        if len(dataloader) == 0:
            print('No batches/data in the dataloader..!')
        elif num_batches is None:
            num_batches = len(dataloader)
        else:
            num_batches = min(len(dataloader), num_batches)

        for i, batch in enumerate(dataloader):

            if i < num_batches:
                loss, rewards_correct_response, rewards_wrong_response = self.preference_loss_batch(batch, beta)

                losses += loss.item()
                rewards_correct += rewards_correct_response.item()
                rewards_wrong += rewards_wrong_response.item()

            else:

                break

        #Average loss and rewards for the loader
        losses /= num_batches
        rewards_correct /= num_batches
        rewards_wrong /= num_batches

        return losses, rewards_correct, rewards_wrong
    

    #Function used in preference fine-tuning task:
    def get_log_probability(self, inputs, mask = None, model_type = None):

        if model_type == 'reference':

            #FROZEN MODEL, Hence no backprop:
            with torch.no_grad():

                logits = self.reference_model(inputs)
        else:

            #Policy model, Hence we need to backprop:
            logits = self.model(inputs)
            
        logits = logits[ :, :-1, :]
        
        labels = inputs[ :, 1:].clone()

        log_probs = F.log_softmax(logits, dim = 1)

        #"labels" has a shape of <batch, num_tokens> and "log_probs" has a shape of <batch, num_tokens, vocab_size>, so we need to add an extra dimension at the very last 
        # in "labels" to perform the selection process. This additional dimension is thus added at the end by specifying "-1". We also need to remove this additional dimension
        # from the output log_probability tensor. So we use "squeeze(-1)" there.
        selected_log_probs = torch.gather(input= log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        if mask is not None:

            selected_mask = mask[:, 1:].clone()

            selected_log_probs = selected_log_probs * selected_mask

            avg_log_probs = selected_log_probs.sum(-1) / selected_mask.sum(-1)

            return avg_log_probs
        
        else:
            
            return selected_log_probs.mean(-1)
        
    #Function used in preference fine-tuning task:
    def dpo_loss_batch(self, policy_correct_probs, policy_wrong_probs, 
                       reference_correct_probs, reference_wrong_probs, 
                       beta= 0.1):
    

        policy_logits = policy_correct_probs - policy_wrong_probs
        reference_logits = reference_correct_probs - reference_wrong_probs

        logits = policy_logits - reference_logits

        scaled_logits = beta * logits

        dpo_loss = -F.logsigmoid(scaled_logits)

        #Calculate the reward gains for correct:
        rewards_correct_response = (policy_correct_probs - reference_correct_probs).detach()

        #Calculate the reward gains for wrong:
        rewards_wrong_response = (policy_wrong_probs - reference_wrong_probs).detach()

        return dpo_loss.mean(), rewards_correct_response.mean(), rewards_wrong_response.mean()
