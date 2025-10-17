import torch
import torch.nn as nn

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):

        #sigmoid = 1 / (1 + torch.exp(-input))

        #Using torch.sigmoid is significantly faster, hence using like this:
        self.activation_output = input * torch.sigmoid(input)

        return self.activation_output
    

if __name__ == '__main__':
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    silu = SiLU()
    output = silu(x)
    print(output)