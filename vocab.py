import pandas as pd
import numpy as np
import torch
from utils import normalizeString, padding

class Vocab:
    def __init__(self, dataset, max_length):
        self.word2idx = {}
        self.idx2word = {}
        self.n_words = 0

        # dataset: list of sentences
        self.dataset = dataset
        self.max_length = max_length

        self.build_vocab()

    def build_vocab(self):
        word_count = {}

        for sentence in self.dataset:
            sentence = normalizeString(sentence)
            for word in sentence.split():
                if word not in word_count:
                    word_count[word] = 1
                else:
                    word_count[word] += 1

        sorted_word_count = dict(sorted(word_count.items(), key= lambda x: x[1], reverse= True))

        self.word2idx = {'<sos>': 0, '<eos>': 1, '<unk>': 2, '<pad>': 3}
        self.n_words = 4

        for word in sorted_word_count.keys():
            self.word2idx[word] = self.n_words
            self.n_words += 1

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def convert_tokens_to_ids(self, tokens: str):
        ids = [self.sos_idx()]
        ids += [self.word2idx.get(token, self.word2idx['<unk>']) for token in normalizeString(tokens).split()]
        ids += [self.eos_idx()]
        return padding(ids, self.max_length, self.pad_idx())
    
    def convert_ids_to_tokens(self, ids: list):
        tokens = [self.idx2word[idx] for idx in ids]
        return [token for token in tokens if token not in ['<eos>', '<sos>']]
    
    def vocab_size(self):
        return self.n_words

    def pad_idx(self):
        return self.word2idx['<pad>']
    
    def sos_idx(self):
        return self.word2idx['<sos>']

    def eos_idx(self):
        return self.word2idx['<eos>']