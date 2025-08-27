from utils import *
from bpe_base import BPETokenizer
import regex as re

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class GPT2RegexTokenizer(BPETokenizer):
    def __init__(self, vocab_size,pattern):
        super().__init__(vocab_size)

        self.pattern = pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.inverse_special_tokens = {} # { int : str }

    def train(self, text,  merge_times, verbose=False):

        assert self.vocab_size >= 256

        text_chunks = re.findall(self.compiled_pattern,text)
        #print(text_chunks)

        token_list = [list(map(int,ch.encode("utf-8"))) for ch in text_chunks]
        #print(len(token_list))

        merges = dict()
        merged_tokens = token_list
        new_token = self.vocab_size

        for i in range(merge_times):

            freq_dict = {}
            for chunk_tokens in merged_tokens:
                get_most_common(chunk_tokens,freq_dict,False)
                
            common_pair = list({key : value for key, value in sorted(freq_dict.items(), 
                                                              key= lambda freq_dict : freq_dict[1], 
                                                              reverse= True )}.keys())[0]

            #print(common_pair)
            merged_tokens = [tokens_merge(chunk_tokens,common_pair, new_token) for chunk_tokens in merged_tokens]
            merges[common_pair] = new_token
            
            if verbose:
                print(f'Merging {common_pair} with {new_token}. ')

            new_token = new_token + 1

        print(f'{merge_times} times merged complete.')
        self.merges = merges
        self.vocab = self.build_vocab()
        # print(self.vocab)

    
    def encode(self, text, allowed_special = "none_raise"):
        '''
        Encoding: Given the raw text return the merged byte token list:
        input: text
        output: list

        Karpathy's comment: 
        " This function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun "

        '''

        special = None

        if allowed_special == "none":
            special = {}
        elif allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none_raise":
            special = {}
            assert all(tok not in text for tok in self.special_tokens)
        elif isinstance(allowed_special,set):
            special = {tok : tok_id for tok, tok_id in self.special_tokens.items() if tok in allowed_special}
        else:
            raise ValueError(f"Invalid special tokens : {allowed_special}")
        
        if not special:
            print('No special tokens detected. Encoding normally!')
            token_list = self.general_encode(text)
        else:
            print('Encoding with special tokens::  ',allowed_special )
            custom_pattern = '(' + '|'.join(tok for tok in allowed_special) + ')'
            text_split = re.split(custom_pattern, text)

            token_list = []
            for text in text_split:
                if text in special:
                    token_list.append(special[text])
                else:
                    token_list.extend(self.general_encode(text))

        return token_list
    
    def general_encode(self, text):
        
        text_chunks = re.findall(self.compiled_pattern, text)
        print(text_chunks)

        token_list = [list(map(int,ch.encode('utf-8'))) for ch in text_chunks]

        tokens = []

        for tok in token_list:

            while len(tok) >= 2:

                freq_dict = get_most_common(tok, None, False)
                
                common_pair = min(freq_dict, key = lambda k: self.merges.get(k, float("inf")))

                if common_pair not in self.merges:
                    #print("Pair not in merges!")
                    break
                
                new_token = self.merges[common_pair]
                tok = tokens_merge(tok, common_pair, new_token)
            tokens.extend(tok)

        return tokens

    def decode(self, token_list):
        '''
        Decoding: Given a list of token_list (byte tokens) return the string:
        input: list
        output: string       
        '''

        tokens = []
        for tok in token_list:

            if tok in self.vocab:
                tokens.append(self.vocab[tok])
            elif tok in self.inverse_special_tokens:
                tokens.append(self.inverse_special_tokens[tok].encode('utf-8'))
            else:
                raise ValueError(f"Invalid token id: {tok}")
        
        byte_text = b''.join(tokens)
        output_text = byte_text.decode('utf-8',errors="replace")

        return  output_text     






        
        









if __name__ == "__main__" :
        
    tokenizer = GPT2RegexTokenizer(256,GPT4_SPLIT_PATTERN)

    with open('taylorswift.txt', "r", encoding = 'utf-8') as file:
        raw_text = file.read()

    tokenizer.train(raw_text, 64, True)
    input = 'supercalifragilistic'
    tokens = tokenizer.encode(input,allowed_special=set(("<|endoftext|>", "<|unk|>")))
    print(tokens)
    output = tokenizer.decode(tokens)
    print(output)
    print(input == output)

        