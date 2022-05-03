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

import matplotlib.pyplot as plt

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
epoch = 150
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

rnn_model = model.RNN_NLP(embed_dim, hidden_dim, n_classes)
rnn_model.to(device)
optimizer = torch.optim.Adam(rnn_model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

# %%
def train(model, optimizer, train_dataloader, model_root, epochs=20):
    """Train the CNN model."""

    print("Start training...")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^9} | {'Elapsed':^9}")
    print("-"*50)

    train_loss_list, train_acc_list = [], []
    best_accuracy = None
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
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        train_accuracy = np.mean(train_accuracy)

        if best_accuracy is not None and train_accuracy < best_accuracy:
            scheduler.step()
        else:
            torch.save(model.state_dict(), model_root)
            best_accuracy = train_accuracy

        time_elapsed = time.time() - t0_epoch
        print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {train_accuracy:^9.2f} | {time_elapsed:^9.2f}")
        
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(train_accuracy)   

    print("\n")
    print(f"Training complete! Best train accuracy: {best_accuracy:.2f}%.")

    return train_loss_list, train_acc_list

train_loss_list, train_acc_list = \
    train(rnn_model, optimizer, train_dataloader, 'nlp-lab-3/result/lab3_problem1.pt', epochs=epoch)

# %% 
# Graph
def graph(train_list, fgname):
    plt.plot(train_list, label='train')
    plt.legend()
    plt.savefig('nlp-lab-3/result/'+ fgname + '.png', dpi=300)
    plt.clf()

graph(train_loss_list, 'problem1_train_loss')
graph(train_acc_list, 'problem1_train_acc')

# %%
def test(model, test_dataloader):
    model.eval()

    test_labels = []
    for batch in test_dataloader:
        b_input_ids, _ = tuple(t.to(device) for t in batch)
        
        with torch.no_grad():
            logits = model(b_input_ids)

        preds = torch.argmax(logits, dim=1).flatten()
        test_labels.append(preds)

    test_labels = torch.cat(test_labels)
    return test_labels.cpu()

rnn_model.load_state_dict(torch.load('nlp-lab-3/result/lab3_problem1.pt', map_location=device))
test_id = pd.read_csv('nlp-lab-3/dataset/classification/classification_class.pred.csv')['ID']
test_labels = test(rnn_model, test_dataloader)

result_df = pd.DataFrame(
    {'ID': test_id,
     'label': test_labels
    })

result_df.to_csv("nlp-lab-3/result/lab3_problem1.csv", index=False)
