import random
import numpy as np
import pandas as pd
import torch
from nltk.tokenize import word_tokenize
import json

from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)

def load_data(filepath='YOUR/CSV/FILE/PATH', train=True):
    with open(filepath) as f:
        data = json.load(f)

    token_list, tag_list = [], []
    if train:
        for v in data.values():
            token_list.append(v['tokens'])
            tag_list.append(v['ud_tags'])
            
        return token_list, tag_list
    else:
        for v in data.values():
            token_list.append(v['tokens'])
            
        return token_list


def load_txt(filepath='YOUR/CSV/FILE/PATH'):
    data = {}
    idx = 0
    with open(filepath) as f:
        while True:
            line = f.readline()
            if not line: break
            data[line.strip()] = idx
            idx += 1 
    
    return data


def vectorization(tokens, word2idx, sent_len):
    sent_list = []
    for sent in tokens:
        word_list = []
        for w_idx, word in enumerate(sent):
            # pre-sequence truncation
            if w_idx >= sent_len:
                break

            if word2idx.get(word)==None:
                index = 1
            else:
                index = word2idx[word]
            word_list.append(index)
        # pre-padding
        while w_idx < (sent_len-1):
            w_idx += 1
            word_list.append(0)
    
        sent_list.append(word_list)
    
    return sent_list


def empty_vectorization(data_len, sent_len):
    sent_list = []
    word_list = []
    for i in range(sent_len):
        word_list.append(0)
        
    for i in range(data_len):
        sent_list.append(word_list)

    return sent_list