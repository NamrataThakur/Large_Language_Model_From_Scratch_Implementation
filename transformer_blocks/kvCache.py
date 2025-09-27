# This code is taken as is from the notebook called 'standalone-qwen3-plus-kvcache.ipynb' of the Sebastian Raschka's repo for 'Build LLM from Scratch' book

import torch
import torch.nn as nn

class KVCache:
    def __init__(self, n_layers):
        self.cache = [None] * n_layers

    def get(self, layer_idx):
        return self.cache[layer_idx]

    def update(self, layer_idx, value):
        self.cache[layer_idx] = value

    def get_all(self):
        return self.cache

    def reset(self):
        for i in range(len(self.cache)):
            self.cache[i] = None