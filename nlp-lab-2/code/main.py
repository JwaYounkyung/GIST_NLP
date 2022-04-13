import utils

import numpy as np
import torch
from nltk.tokenize import word_tokenize
from tqdm import tqdm_notebook

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time

utils.set_seed(42)

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# parameter setting
lr = 0.001
epoch = 20
batch_size = 32

sent_len = 20
word_len = 10
embed_dim = 10

# %%
# load data
tr_sents, tr_labels = utils.load_data(filepath='nlp-lab-2/data/sent_class.train.csv')
ts_sents, ts_labels = utils.load_data(filepath='nlp-lab-2/data/sent_class.test.csv')

# tokenization
tr_tokens = utils.tokenization(tr_sents)
ts_tokens = utils.tokenization(ts_sents)

# lemmatization
tr_lemmas, char2idx = utils.lemmatization(tr_tokens, train=True)
ts_lemmas, _ = utils.lemmatization(ts_tokens, train=False)

# encode for dataloader
tr_inputs = utils.encode(tr_lemmas, char2idx, sent_len, word_len, device)
ts_inputs = utils.encode(ts_lemmas, char2idx, sent_len, word_len, device)

# Load data to PyTorch DataLoader
train_dataloader, test_dataloader = \
utils.data_loader(tr_inputs, ts_inputs, tr_labels, ts_labels, device, batch_size=batch_size)

# %%
class CNN_NLP(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""
    def __init__(self,
                 vocab_size=None,
                 embed_dim=300,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 num_classes=2,
                 dropout=0.5):
        """
        The constructor for CNN_NLP class.

        Args:
            vocab_size (int): Need to be specified when not pretrained word
                embeddings are not used.
            embed_dim (int): Dimension of word vectors. Need to be specified
                when pretrained word embeddings are not used. Default: 300
            filter_sizes (List[int]): List of filter sizes. Default: [3, 4, 5]
            num_filters (List[int]): List of number of filters, has the same
                length as `filter_sizes`. Default: [100, 100, 100]
            n_classes (int): Number of classes. Default: 2
            dropout (float): Dropout rate. Default: 0.5
        """

        super(CNN_NLP, self).__init__()

        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim=self.embed_dim,
                                        padding_idx=0,
                                        max_norm=5.0)
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):

        x_embed = self.embedding(input_ids).float()
        x_reshaped = x_embed.permute(0, 2, 1)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
            for x_conv in x_conv_list]
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)
        
        logits = self.fc(self.dropout(x_fc))

        return logits

loss_fn = nn.CrossEntropyLoss()
cnn_model = CNN_NLP(vocab_size=len(char2idx),
                    embed_dim=embed_dim,
                    filter_sizes=[3, 4, 5],
                    num_filters=[100, 100, 100],
                    num_classes=2,
                    dropout=0.5)
cnn_model.to(device)
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=lr)

# %%
def train(model, optimizer, train_dataloader, epochs=20):
    """Train the CNN model."""

    print("Start training...\n")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^9} | {'Elapsed':^9}")
    print("-"*50)

    best_accuracy = 0
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================

        t0_epoch = time.time()
        total_loss = 0
        train_accuracy = []

        model.train()

        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_labels = tuple(t.to(device) for t in batch)

            model.zero_grad()

            logits = model(b_input_ids)

            loss = loss_fn(logits, b_labels)
            preds = torch.argmax(logits, dim=1).flatten()

            total_loss += loss.item()
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            train_accuracy.append(accuracy)

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        train_accuracy = np.mean(train_accuracy)

        if train_accuracy > best_accuracy:
            best_accuracy = train_accuracy
            torch.save(model.state_dict(), 'nlp-lab-2/result/lab2.pt')

        time_elapsed = time.time() - t0_epoch
        print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {train_accuracy:^9.2f} | {time_elapsed:^9.2f}")
        
    print("\n")
    print(f"Training complete! Best train accuracy: {best_accuracy:.2f}%.")

train(cnn_model, optimizer, train_dataloader, epochs=epoch)

# %%
def evaluate(model, test_dataloader):
    model.eval()

    test_accuracy = []
    test_loss = []

    for batch in test_dataloader:
        b_input_ids, b_labels = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            logits = model(b_input_ids)

        loss = loss_fn(logits, b_labels)
        test_loss.append(loss.item())

        preds = torch.argmax(logits, dim=1).flatten()

        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        test_accuracy.append(accuracy)

    test_loss = np.mean(test_loss)
    test_accuracy = np.mean(test_accuracy)

    return test_loss, test_accuracy