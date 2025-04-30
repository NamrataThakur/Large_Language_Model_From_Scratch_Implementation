import torch
from text_generation import Text_Generation
from metrics import Metrics

class GPT2_PreTrain:
    def __init__(self, model, optimizer, device, train_dataLoader, test_dataLoader,num_epochs, eval_batchSize, eval_freq, 
                 start_context,max_new_tokens):

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

        self.generation = Text_Generation(self.model, self.device, 'gpt2')
        self.metrics = Metrics(self.model, self.device)

    def evaluate_model(self):

        self.model.eval()

        with torch.no_grad():
            train_loss = self.metrics.loss_loader(self.train_loader, self.eval_batchSize)
            test_loss = self.metrics.loss_loader(self.val_loader, self.eval_batchSize)

        self.model.train()

        return train_loss, test_loss

    
    def train(self, temp=0.0, top_k= None, eos_id = None):

        train_losses, test_losses, track_tokens_seen = [], [], []

        tokens_seen, global_step = 0, -1

        for ep in range(self.num_epochs):

            self.model.train()

            for train_x, train_y in self.train_loader:

                self.optimizer.zero_grad()
                loss = self.metrics.loss_batch(train_x, train_y)
                loss.backward()
                self.optimizer.step()
                global_step += 1
                tokens_seen += train_x.numel()

                if global_step % self.eval_freq == 0:

                    train_loss, test_loss = self.evaluate_model(self)
                    train_losses.append(train_loss)
                    test_losses.append(test_loss)
                    print(f'Epoch No: {ep+1}, Step: {global_step:06d}, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}')
                    print(f'Total Tokens seen till now: {tokens_seen}')
                    track_tokens_seen.append(tokens_seen)

            self.generation.text_generation(self.start_context, self.max_new_tokens, temp, top_k, eos_id)

        return train_losses, test_losses, track_tokens_seen

        
