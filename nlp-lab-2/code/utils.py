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


def char_onehot(lemmas: List[List[str]]) -> torch.Tensor:
	
	return v
