import tiktoken
import torch

import os
os.pardir

from transformer_blocks.kvCache import KVCache

class Text_Generation:
    def __init__(self, model, device='cuda', tokenizer_model="gpt2", arch_type="original"):

        self.tokenizer = tiktoken.get_encoding(tokenizer_model)
        self.device = device
        self.model = model
        self.arch_type = arch_type

        self.model.eval()

        if self.arch_type == 'original':
            self.context_size = self.model.pos_embedding.weight.shape[0]         
        else:
            self.context_size = self.model.rope_angles.m.shape[0]


    #Convert input text into a torch tensor of token list with batch dimension added
    def text_to_tokenID(self, text):
        encoded_ids = []
        if type(text) == str:
            encoded_ids = (self.tokenizer.encode(text, allowed_special = {'<|endoftext|>'}))
            batch_tok_ids = torch.tensor(encoded_ids).unsqueeze(0)
        else:
            for t in text:
                encoded_ids.append(self.tokenizer.encode(t, allowed_special = {'<|endoftext|>'}))
                batch_tok_ids = torch.tensor(encoded_ids)
        
        return batch_tok_ids

    #Convert input torch tensor of token list with batch dimension to text (without batch dimension)
    def tokenID_to_text(self,batch_tokID):
        token_list = batch_tokID.squeeze(0)
        text = self.tokenizer.decode(token_list.tolist())
        return text
    
    def text_generation(self, input_text, max_new_tokens, temp=1.0, top_k= None, eos_id = None, kv_cache = False):

        idx = self.text_to_tokenID(input_text).to(self.device)

        
        if self.arch_type == 'GQA' or self.arch_type == 'MOE':
            if kv_cache:
                cache = KVCache(n_layers= self.model.config['num_layers'])
                cache.reset()
                self.model.clear_cache()
            else:
                cache = None

        else:
            cache = kv_cache

            if kv_cache:
                self.model.clear_cache()
            

        idx_cond = idx[:, -self.context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = self.model(idx_cond , cache = cache)


        with torch.no_grad():
            # idx is (B, T) array of indices in the current context
            for _ in range(max_new_tokens):

                # Focus only on the last time step
                # (batch, n_token, vocab_size) becomes (batch, vocab_size)
                logits = logits[:, -1, :]

                #Get the top_k tokens:
                if top_k is not None and top_k > 0:

                    # Get the top k tokens from the logits vector
                    top_logits, _ = torch.topk(logits, top_k)
                    
                    # Mask out values of all other location (apart from those of top k) with -infinity value so that softmax is not applied on them later.
                    logits = torch.where(condition= logits < top_logits[:, -1],
                                        input= torch.tensor(float("-inf")).to(logits.device),
                                        other= logits)
                    
                #Get the temperature scaled logits:
                if temp > 0.0:

                    #Scale the logits according to the temperature value:
                    scaled_logits = logits / temp

                    #Get the softmax probabilities on the scaled logits
                    probas = torch.softmax(scaled_logits, dim=-1)

                    #Get the next token prediction sampled on a multinomial distribution
                    idx_next = torch.multinomial(probas, num_samples=1)
                
                else:

                    # Example: idx_next : tensor([[48121]], device='cuda:0')
                    # Get the idx of the vocab entry with the highest logits value
                    idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

                #If end of seuence token id is provided, stop generating if that token id is predicted
                if eos_id is not None and idx_next.item() == eos_id:
                    print(f"Token predicted : {idx_next.item()}")
                    break

                # Append sampled index to the running sequence
                idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

                if kv_cache:

                    # Get the predictions
                    with torch.no_grad():
                        logits = self.model(idx_next, cache = cache) #cache is object of KVCache class or True

                else:

                    # Crop current context if it exceeds the supported context size
                    # E.g., if LLM supports only 5 tokens, and the context size is 10
                    # then only the last 5 tokens are used as context
                    idx_cond = idx[:, -self.context_size:]

                    # Get the predictions
                    with torch.no_grad():

                        #Shape: torch.Size([1, 1, 50257])
                        logits = self.model(idx_cond , cache = cache) #cache is either None or False


        gen_output = self.tokenID_to_text(idx)
        gen_output = gen_output.replace('\n', '')
        return gen_output
    

    def classify_text(self, input_text, max_length, pad_token_id=50256):
        '''
        max_length: Can be the max sequence length of the training dataset or a custom number too for inference.
        '''
        
        self.model.eval()

        input_encoded = self.tokenizer.encode(input_text, allowed_special='all')

        input_encoded = input_encoded[:min(self.context_size, max_length)]
        input_encoded += [pad_token_id] * (max_length - len(input_encoded))

        input_tensor = torch.tensor(input_encoded,dtype=torch.long, device=self.device).unsqueeze(0)

        with torch.no_grad():
            output_logit = self.model(input_tensor)[:, -1, :]

        class_prediction = torch.argmax(output_logit, dim = -1).item()

        return class_prediction
    

    
