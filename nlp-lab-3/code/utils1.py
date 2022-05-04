import random
import numpy as np
import pandas as pd
import torch
from nltk.tokenize import word_tokenize
import json

from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)


def set_seed(seed_value=42):
    """Set seed for reproducibility."""

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def load_data(filepath='YOUR/CSV/FILE/PATH', train=True):
    csv = pd.read_csv(filepath, header=0)
    data = csv['sentence']
    
    if train:
        targets = csv['label']
        return data, targets
    
    return data
    

def tokenization(texts):
	tokenized_texts = []
	
	for sent in texts:
		tokenized_sent = word_tokenize(sent.lower())
		tokenized_texts.append(tokenized_sent)

	return tokenized_texts


def dictionary(tokens):
    word2idx = {}

    word2idx['[pad]'] = 0
    word2idx['[unk]'] = 1

    idx = 2
    for sent in tokens:
        for word in sent:
            if word.lower() not in word2idx:
                word2idx[word.lower()] = idx
                idx += 1
    
    return word2idx


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
            word_list.insert(0,0)
    
        sent_list.append(word_list)
    
    return sent_list

def load_pretrained(filepath='YOUR/CSV/FILE/PATH'):
    with open(filepath) as f:
        glove = json.load(f)
    
    glove_lower =  {k.lower(): v for k, v in glove.items()}
    return glove_lower


def embedding(vectors, glove, idx2word, device):
    sent_list = []
    for sent in vectors:
        word_list = []
        for idx in sent:
            word = idx2word[idx]
            if glove.get(word)==None:
                embedding = glove['[unk]']
            else:
                embedding = glove[word]
            word_list.append(embedding)
        sent_list.append(word_list)

    var = torch.tensor(sent_list, device=device)
    return var


def data_loader(train_inputs, test_inputs, train_labels, test_labels, device, batch_size=32):
    """Convert train and test sets to torch.Tensors and load them to
    DataLoader.
    """

    # Convert label data type to torch.Tensor
    train_labels, test_labels = \
	tuple(torch.tensor(data, device=device) for data in [train_labels, test_labels])

    # Create DataLoader for training data
    train_data = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create DataLoader for test data
    test_data = TensorDataset(test_inputs, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return train_dataloader, test_dataloader