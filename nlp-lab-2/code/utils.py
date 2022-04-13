import enum
from tkinter.tix import Tree
from typing import Union, List, Dict
import random
import numpy as np
import pandas as pd
import torch

from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)

def set_seed(seed_value=42):
    """Set seed for reproducibility."""

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def load_data(filepath: str = 'YOUR/CSV/FILE/PATH') -> Union[List[str], List[int]]:
	csv = pd.read_csv(filepath, header=0)
	data = csv['sentence']
	targets = csv['label']

	return data, targets


def tokenization(texts: List[str]) -> List[List[str]]:
	tokenized_texts = []
	
	for sent in texts:
		tokenized_sent = word_tokenize(sent.lower())
		tokenized_texts.append(tokenized_sent)

	return tokenized_texts


def get_wordnet_pos(word):
	tag = pos_tag([word])[0][1][0].upper()
	tag_dict = {"J": wordnet.ADJ,
				"N": wordnet.NOUN,
				"V": wordnet.VERB,
				"R": wordnet.ADV}
	
	return tag_dict.get(tag, wordnet.NOUN) # 예외 값은 다 명사로 처리


def lemmatization(tokens, train=True):
	lemmatizer = WordNetLemmatizer()
	lemmas = []
	char2idx = {}

	char2idx['<pad>'] = 0
	char2idx['<unk>'] = 1

	idx = 2
	for token in tokens:
		lemma = []
		for word in token:
			lemma.append(lemmatizer.lemmatize(word, get_wordnet_pos(word)))
			if train:
				for char in word:
					if char not in char2idx:
						char2idx[char] = idx
						idx += 1
		lemmas.append(lemma)

	return lemmas, char2idx


def one_hot_encoding(char, char2idx):
	one_hot_vector = [0]*len(char2idx)

	if char2idx.get(char)==None:
		return one_hot_vector
		
	index = char2idx[char]
	one_hot_vector[index] = 1

	return one_hot_vector


def encode(lemmas, char2idx, sent_len, word_len, device):
	sent_list = []
	for sent in lemmas:
		word_list = []
		for w_idx, word in enumerate(sent):
			if w_idx >= sent_len:
				break
			char_list = []
			for c_idx, char in enumerate(word):
				if c_idx >= word_len:
					break
				char_list.append(one_hot_encoding(char, char2idx))
			# word padding
			while c_idx < (word_len-1):
				c_idx += 1
				char_list.append([0]*len(char2idx))
			
			word_list.append(char_list)
		# sentence padding
		while w_idx < (sent_len-1):
			w_idx += 1
			word_list.append([[0]*len(char2idx)]*word_len)
			
		sent_list.append(word_list)
	
	var = torch.tensor(sent_list, dtype=torch.long, device=device)

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