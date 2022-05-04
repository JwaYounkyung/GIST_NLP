# Problem 2 Code
import utils1, utils2
import model

import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

utils1.set_seed(42)

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# parameter setting
lr = 1e-3
epoch = 150
batch_size = 256
weight_decay = 1e-4

sent_len = 65
hidden_dim = 512
embed_dim = 300
n_classes = 18

# %%
print("Preprocessing...")
# load data
tr_tokens, tr_labels = utils2.load_data(filepath='nlp-lab-3/dataset/pos/train_set.json')
ts_tokens = utils2.load_data(filepath='nlp-lab-3/dataset/pos/test_set.json', train=False)
ts_sent_len = [len(sent) for sent in ts_tokens]

# dictionary generation
word2idx = utils1.dictionary(tr_tokens)
idx2word = {v:k for k, v in word2idx.items()}
dict_label = utils2.load_txt(filepath='nlp-lab-3/dataset/pos/tgt.txt')

# vectorization using post-padding, pre-sequence truncation
tr_vec = utils2.vectorization(tr_tokens, word2idx, sent_len)
ts_vec = utils2.vectorization(ts_tokens, word2idx, sent_len)
tr_vec_label = utils2.vectorization(tr_labels, dict_label, sent_len)
ts_vec_label = utils2.empty_vectorization(ts_sent_len, sent_len)

# load Fasttext
fasttext = utils1.load_pretrained(filepath='nlp-lab-3/dataset/pos/fasttext_word.json')

# embedding
tr_inputs = utils1.embedding(tr_vec, fasttext, idx2word, device)
ts_inputs = utils1.embedding(ts_vec, fasttext, idx2word, device)

# Load data to PyTorch DataLoader
train_dataloader, test_dataloader = \
utils2.data_loader(tr_inputs, ts_inputs, tr_vec_label, ts_vec_label, device, batch_size=batch_size)

# %%
loss_fn = nn.CrossEntropyLoss(ignore_index=0)

lstm_model = model.LSTM_NLP(embed_dim, hidden_dim, n_classes)
lstm_model.to(device)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

# %%
def categorical_accuracy(preds, y, tag_pad_idx=0):
    """
    미니 배치에 대한 정확도 출력, pad가 아닌 것들에 대해서만 정확도 추출
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])

    return correct.sum() / y[non_pad_elements].shape[0]

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
        total_loss, total_acc = 0, 0 

        model.train()

        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_labels = tuple(t.to(device) for t in batch)

            model.zero_grad()

            logits = model(b_input_ids)

            # loss = loss_fn(logits.permute(0,2,1), b_labels)
            preds = logits.view(-1, logits.shape[-1])
            tags = b_labels.view(-1)
            loss = loss_fn(preds, tags)
            
            accuracy = categorical_accuracy(preds, tags)
            total_loss += loss.item()
            total_acc += accuracy.item()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        train_accuracy = (total_acc / len(train_dataloader))*100

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
    train(lstm_model, optimizer, train_dataloader, 'nlp-lab-3/result/lab3_problem2.pt', epochs=epoch)

# %% 
# Graph
def graph(train_list, fgname):
    plt.plot(train_list, label='train')
    plt.legend()
    plt.savefig('nlp-lab-3/result/'+ fgname + '.png', dpi=300)
    plt.clf()

graph(train_loss_list, 'problem2_train_loss')
graph(train_acc_list, 'problem2_train_acc')

# %%
def test(model, test_dataloader, ts_sent_len, sent_len):
    model.eval()

    test_labels = []
    for step, batch in enumerate(test_dataloader):
        b_input_ids, b_labels = tuple(t.to(device) for t in batch)
        
        with torch.no_grad():
            logits = model(b_input_ids)

        preds = logits.view(-1, logits.shape[-1])
        max_preds = preds.argmax(dim = 1, keepdim = True)
        non_pad_elements = (b_labels.flatten() != 0).nonzero()
        non_pad_preds = max_preds[non_pad_elements].flatten()

        # padded sentence
        if ts_sent_len[step] > sent_len:
            for i in range(sent_len, ts_sent_len[step]):
                non_pad_preds = torch.cat((non_pad_preds,torch.tensor([0], device=device)))

        test_labels.append(non_pad_preds)

    test_labels = torch.cat(test_labels)
    return test_labels.cpu()

lstm_model.load_state_dict(torch.load('nlp-lab-3/result/lab3_problem2.pt', map_location=device))
test_id = pd.read_csv('nlp-lab-3/dataset/pos/pos_class.pred.csv')['ID']
test_labels = test(lstm_model, test_dataloader, ts_sent_len, sent_len)

result_df = pd.DataFrame(
    {'ID': test_id,
     'label': test_labels
    })

result_df.to_csv("nlp-lab-3/result/lab3_problem2.csv", index=False)
