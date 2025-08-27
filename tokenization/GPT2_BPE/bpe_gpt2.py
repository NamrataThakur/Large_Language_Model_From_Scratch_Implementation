from utils import *
from bpe_base import BPETokenizer

class GPT2Tokenizer(BPETokenizer):
    def __init__(self, vocab_size):
        super().__init__(vocab_size)
    def train(self, text, merge_times, verbose=False):

        #Getting the raw bytes as tokens from the text:
        byte_tokens = text.encode('utf-8')

        # Converting the tokens to int:
        token_list = list(map(int, byte_tokens))

        merges = dict()
        merged_tokens = token_list
        sorted_merged_frequency = get_most_common(merged_tokens,None, True)

        new_token = self.vocab_size
        for i in range(merge_times):
            merged_tokens = tokens_merge(merged_tokens,list(sorted_merged_frequency.keys())[0],new_token )
            merges[list(sorted_merged_frequency.keys())[0]] = new_token
            if verbose:
                print(f'Merging {list(sorted_merged_frequency.keys())[0]} with {new_token}. ')
            sorted_merged_frequency = get_most_common(merged_tokens,None, True)

            new_token = new_token + 1
        
        print(f'{merge_times} times merged complete.')
        self.merges = merges
        self.vocab = self.build_vocab() 

    def encode(self, text):
        '''
        Encoding: Given the raw text return the merged byte token list:
        input: text
        output: list
        '''
        tokens = list(map(int, text.encode('utf-8')))

        while len(tokens) >= 2:
            sorted_frequency_dict = get_most_common(tokens, None, False)

            pair = min(sorted_frequency_dict, key = lambda x: self.merges.get(x, float("inf")))

            if pair not in self.merges:
                break
            else:
                new_token = self.merges[pair]
                tokens = tokens_merge(tokens,pair,new_token)
        return tokens
    
    def decode(self, token_list):
        '''
        Decoding: Given a list of token_list (byte tokens) return the string:
        input: list
        output: string
        '''

        output_tok = b''.join(self.vocab[tok] for tok in token_list)
        output_string = output_tok.decode('utf-8', errors="replace")
        return output_string   


if __name__ == "__main__" :
        
    tokenizer = GPT2Tokenizer(256)

    with open('taylorswift.txt', "r", encoding = 'utf-8') as file:
        raw_text = file.read()

    tokenizer.train(raw_text, 64, True)
    input = 'supercalifragilistic'
    tokens = tokenizer.encode(input)
    print(tokens)
    output = tokenizer.decode(tokens)
    print(output)
    print(input == output)

    #[115, 117, 112, 278, 99, 274, 105, 102, 114, 97, 103, 105, 108, 105, 115, 298, 99]
    #[115, 117, 112, 278, 99, 274, 105, 102, 114, 97, 103, 105, 108, 105, 115, 298, 99]




    