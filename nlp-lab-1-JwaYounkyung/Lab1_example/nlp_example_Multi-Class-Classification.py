# Multi Class Text Classification wiht Torchtext
# https://tutorials.pytorch.kr/beginner/text_sentiment_ngrams_tutorial.html
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# %%
# AG_NEWS 불러오기
train_iter = iter(AG_NEWS(split='train'))

print(next(train_iter))
print(next(train_iter))

# %% 

tokenizer = get_tokenizer('basic_english') # token 형성기
train_iter = AG_NEWS(split='train')

# 한문장씩 tokenize
# generator 생성 : 데이터를 미리 만들어 놓지 않고 필요할 때마다 즉석에서 하나씩 만들어 낼 수 있는 객체 
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text) 

vocab = build_vocab_from_iterator(yield_tokens(train_iter))
# print(vocab.stoi)
print(vocab.stoi['here'])

text_pipeline = lambda x: [vocab.stoi[word] for word in tokenizer(x)]
label_pipeline = lambda x: int(x) - 1

print(text_pipeline('here is the an example'))
print(label_pipeline('10'))

# %%
# collate batch setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0)) # text 길이
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

train_iter = AG_NEWS(split='train')
dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)
# %%
# Model
class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

num_class = len(set([label for (label, text) in train_iter])) # set : 중복을 허용하지 않는 list
vocab_size = len(vocab)
emsize = 64 # embedding size
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

# %%
# train & evaluate

def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()
    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets) # offsets : text 길이 누적 [0,40,86] 첫번째 길이 40, 두번째 길이 46
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1) # gradient exploding 방지
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)

            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    
    return total_acc/total_count


# Hyperparameters
EPOCHS = 10 # epoch
LR = 5  # learning rate
BATCH_SIZE = 64 # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
train_iter, test_iter = AG_NEWS()

def to_map_style_dataset(iter_data):
    """Convert iterable-style dataset to map-style dataset.

    args:
        iter_data: An iterator type object. Examples include Iterable datasets, string list, text io, generators etc.
    """

    # Inner class to convert iterable-style to map-style dataset
    class _MapStyleDataset(torch.utils.data.Dataset):

        def __init__(self, iter_data):
            # TODO Avoid list issue #1296
            self._data = list(iter_data)

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            return self._data[idx]

    return _MapStyleDataset(iter_data)

train_dataset = to_map_style_dataset(train_iter) # 120000개(114000 + 6000)
test_dataset = to_map_style_dataset(test_iter) # 7600개
num_train = int(len(train_dataset) * 0.95) 
split_train_, split_valid_ = \
    random_split(train_dataset, [num_train, len(train_dataset) - num_train]) # train, validate split : testdataset수에 맞게 validation 수 맞춤

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_batch)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)

    if total_accu is not None and total_accu > accu_val: # accuracy가 업데이트가 안되었을 때
      scheduler.step()
    else:
       total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)

# %% 

ag_news_label = {1: "World",
                 2: "Sports",
                 3: "Business",
                 4: "Sci/Tec"}

def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."

model = model.to("cpu")

print("This is a %s news" %ag_news_label[predict(ex_text_str, text_pipeline)])
