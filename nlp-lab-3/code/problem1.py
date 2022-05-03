# Problem 1 Code
import utils
import model

import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

utils.set_seed(42)

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# parameter setting
lr = 1e-5
epoch = 100
batch_size = 256
weight_decay = 1e-4

sent_len = 20
hidden_dim = 512
embed_dim = 300
n_classes = 6

# %%
print("Preprocessing...")
# load data
tr_sents, tr_labels = utils.load_data(filepath='nlp-lab-3/dataset/classification/train_set.csv')
ts_sents = utils.load_data(filepath='nlp-lab-3/dataset/classification/test_set.csv', train=False)
ts_labels = pd.DataFrame(np.zeros(len(ts_sents)).astype(int))[0]

# tokenization
tr_tokens = utils.tokenization(tr_sents)
ts_tokens = utils.tokenization(ts_sents)

# dictionary generation
word2idx = utils.dictionary(tr_tokens)
idx2word = {v:k for k, v in word2idx.items()}

# vectorization using pre-padding, pre-sequence truncation
tr_vec = utils.vectorization(tr_tokens, word2idx, sent_len)
ts_vec = utils.vectorization(ts_tokens, word2idx, sent_len)

# load GloVe
glove = utils.load_glove(filepath='nlp-lab-3/dataset/classification/glove_word.json')

# embedding
tr_inputs = utils.embedding(tr_vec, glove, idx2word, device)
ts_inputs = utils.embedding(ts_vec, glove, idx2word, device)

# Load data to PyTorch DataLoader
train_dataloader, test_dataloader = \
utils.data_loader(tr_inputs, ts_inputs, tr_labels, ts_labels, device, batch_size=batch_size)

# %%
loss_fn = nn.CrossEntropyLoss()

rnn_model = model.CNN_NLP(vocab_size=len(word2idx),
                          word_len=10,
                          embed_dim=embed_dim,
                          filter_sizes=[2, 3, 4],
                          num_filters=[100, 100, 100],
                          num_classes=n_classes,
                          dropout=0.5)
rnn_model.to(device)
optimizer = torch.optim.Adam(rnn_model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)