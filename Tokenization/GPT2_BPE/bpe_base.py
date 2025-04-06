from utils import *

class BPETokenizer:

    def __init__(self,vocab_size):
        
        self.merges = {} #{ (int, int) : int }
        self.special_tokens = {} # {str : int}
        self.vocab_size = vocab_size
        self.vocab = {} # { int : str }

    def train(self, text, merge_times, verbose=False):
        return NotImplementedError
    
    def encode(self, text):
        return NotImplementedError
    
    def decode(self,token_list):
        return NotImplementedError
    
    def build_vocab(self):
        vocab = dict()
        vocab = {token_id : bytes([token_id]) for token_id in range(self.vocab_size)}
        for k,v in self.merges.items():
            vocab[v] = vocab[k[0]] + vocab[k[1]]
        
        if len(list(self.special_tokens.keys())) > 0:
            for spc_tok, tok_id in self.special_tokens.items():
                vocab[tok_id] = spc_tok.encode('utf-8')
        
        print('Vocabulary built of size :', len(vocab) )
        return vocab

    
    def save_model(self, filename):
        return
    
    def load_model(self,filename):
        return
    
    def token_visualization(self,byte_list):
        return 
