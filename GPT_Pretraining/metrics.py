import torch

class Metrics:
    def __init__(self, model, device ):
        
        self.model = model
        self.device = device

    def loss_loader(self, dataloader, num_batches=None, classify=False):

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
                
                last_idx_logits = output_logits[:, -1, :]
                predictions = torch.argmax(last_idx_logits, dim=-1)

                num_samples += predictions.shape[0]

                correct_pred += (predictions == Y_batch).sum().item()

            else:
                break
        avg_accuracy = correct_pred / num_samples
        return avg_accuracy
    
    
    def classification_loss_batch(self, input_batch, target_batch):

        input_batch, target_batch = input_batch.to(self.device), target_batch.to(self.device)
        output_logits = self.model(input_batch)
        last_idx_logits = output_logits[:, -1, :]
        classify_loss = torch.nn.functional.cross_entropy(last_idx_logits, target_batch)
        return classify_loss