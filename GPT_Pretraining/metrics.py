import torch

class Metrics:
    def __init__(self, model, device ):
        
        self.model = model
        self.device = device

    def loss_loader(self, dataloader, num_batches=None):

        batch_loss = 0.0
        if len(dataloader) == 0:
            print('No batches/data in the dataloader..!')
        elif num_batches is None:
            num_batches = len(dataloader)
        else:
            num_batches = min(num_batches,len(dataloader))
            
        for i, (input_batch, target_batch) in enumerate(dataloader):

            if i < num_batches:
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
