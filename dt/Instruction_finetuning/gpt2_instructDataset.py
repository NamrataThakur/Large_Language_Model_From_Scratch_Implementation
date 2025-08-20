import torch
from torch.utils.data import Dataset, DataLoader
from .gpt2_instructDataFormat import format_input_response

class GPTCustomInstructionDataset(Dataset):
    def __init__(self, input_data, tokenizer, prompt_style):
        self.encoded_data = []
        self.instruction_length = []
        self.data = input_data

        for row in input_data:

            instruct_length, formatted_row = format_input_response(row, prompt_style, inference=False)
            instruction_input = formatted_row[:instruct_length]

            tokenized_row = tokenizer.encode(formatted_row, allowed_special = 'all')
            tokenized_instruction_length = len(tokenizer.encode(instruction_input, allowed_special = 'all'))

            self.encoded_data.append(tokenized_row)
            self.instruction_length.append(tokenized_instruction_length)

    def __getitem__(self, index):
        return self.instruction_length[index], self.encoded_data[index]
    
    def __len__(self):
        return len(self.data)