import os
os.pardir

import torch
from gpt_Pretraining.metrics import Metrics

class GPT2_ClassificationFineTune:
    def __init__(self, model, optimizer, device, train_dataLoader, test_dataLoader,
                 num_epochs, eval_batchSize, eval_freq, log_path):

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_dataLoader
        self.val_loader = test_dataLoader
        self.num_epochs = num_epochs
        self.eval_batchSize = eval_batchSize
        self.eval_freq = eval_freq

        self.log_path = log_path
        self.metrics = Metrics(self.model, self.device)

    def evaluate_model(self):

        self.model.eval()

        with torch.no_grad():
            train_loss = self.metrics.loss_loader(self.train_loader, self.eval_batchSize,classify=True)
            test_loss = self.metrics.loss_loader(self.val_loader, self.eval_batchSize,classify=True)

        self.model.train()

        return train_loss, test_loss

    
    def train(self):

        train_losses, test_losses, train_accuracy, val_accuracy = [], [], [], []

        num_samples, global_step = 0, -1

        #Open the log file present in the log_path:
        log_file = open(self.log_path, "a")

        for ep in range(self.num_epochs):

            self.model.train()

            for train_x, train_y in self.train_loader:

                self.optimizer.zero_grad()
                loss = self.metrics.classification_loss_batch(train_x, train_y)
                loss.backward()
                self.optimizer.step()
                global_step += 1
                num_samples += train_x.shape[0]

                if global_step % self.eval_freq == 0:

                    train_loss, test_loss = self.evaluate_model()
                    train_losses.append(train_loss)
                    test_losses.append(test_loss)
                    print(f'Epoch No: {ep+1}, Step: {global_step:06d}, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}')

                    #Write the epoch wise metrics in the log file:
                    log_file.write(f'Epoch No: {ep+1}, Step: {global_step:06d}, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}\n')
                    
            #Calculate avergae accuracy for each epoch:
            train_accu = self.metrics.accuracy_loader(self.train_loader)
            val_accu = self.metrics.accuracy_loader(self.val_loader)
            print(f"Training accuracy: {train_accu*100:.2f}%")
            print(f"Validation accuracy: {val_accu*100:.2f}%")

            #Write the epoch wise metrics in the log file:
            log_file.write(f"Training accuracy: {train_accu*100:.2f}%")
            log_file.write(f"Validation accuracy: {val_accu*100:.2f}%")

            train_accuracy.append(train_accu)
            val_accuracy.append(val_accu)

        #Close the log file:
        log_file.close()
            
        return train_losses, test_losses, train_accuracy, val_accuracy, num_samples

        
