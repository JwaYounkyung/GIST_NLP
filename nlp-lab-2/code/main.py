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
lr = 0.001
epoch = 40
batch_size = 32

sent_len = 20
word_len = 10
embed_dim = 10
n_classes = 6

# %%
print("Preprocessing...")
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
loss_fn = nn.CrossEntropyLoss()
cnn_model = model.CNN_NLP(vocab_size=len(char2idx),
                          word_len=word_len,
                          embed_dim=embed_dim,
                          filter_sizes=[2, 3, 4],
                          num_filters=[100, 100, 100],
                          num_classes=n_classes,
                          dropout=0.5)
cnn_model.to(device)
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
# %%
def train(model, optimizer, train_dataloader, model_root, epochs=20):
    """Train the CNN model."""

    print("Start training...")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^9} | {'Elapsed':^9}")
    print("-"*50)

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
        
    print("\n")
    print(f"Training complete! Best train accuracy: {best_accuracy:.2f}%.")

train(cnn_model, optimizer, train_dataloader, 'nlp-lab-2/result/lab2.pt', epochs=epoch)

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
    return test_labels

cnn_model.load_state_dict(torch.load('nlp-lab-2/result/lab2.pt',  map_location=device))
test_id = pd.read_csv('nlp-lab-2/data/sent_class.pred.csv')['id']
test_labels = test(cnn_model, test_dataloader)

result_df = pd.DataFrame(
    {'id': test_id,
     'pred': test_labels
    })

result_df.to_csv("nlp-lab-2/result/lab2.csv", index=False)
