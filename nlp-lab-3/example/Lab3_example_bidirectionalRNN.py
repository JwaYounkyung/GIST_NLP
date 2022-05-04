# 양방향 RNN을 이용한 품사 태깅
# https://wikidocs.net/66747

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import legacy
from torchtext.legacy import datasets
import time
import random
import urllib.request

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3개의 필드 정의
TEXT = legacy.data.Field(lower = True)
UD_TAGS = legacy.data.Field(unk_token = None)
PTB_TAGS = legacy.data.Field(unk_token = None)

fields = (("text", TEXT), ("udtags", UD_TAGS), ("ptbtags", PTB_TAGS))

train_data, valid_data, test_data = datasets.UDPOS.splits(fields)
print(f"훈련 샘플의 개수 : {len(train_data)}")
print(f"검증 샘플의 개수 : {len(valid_data)}")
print(f"테스트 샘플의 개수 : {len(test_data)}")

# 훈련 데이터의 3개의 필드 확인
print(train_data.fields)
print(vars(train_data.examples[0])['text'])
print(vars(train_data.examples[0])['udtags'])
print(vars(train_data.examples[0])['ptbtags'])

# 최소 허용 빈도
MIN_FREQ = 5

# 사전 훈련된 워드 임베딩 GloVe 다운로드
TEXT.build_vocab(train_data, min_freq = MIN_FREQ, vectors = "glove.6B.100d")
UD_TAGS.build_vocab(train_data)
PTB_TAGS.build_vocab(train_data)
# 상위 빈도수 20개 단어
print(TEXT.vocab.freqs.most_common(20))
# 상위 정수 인덱스 단어 10개 출력
print(TEXT.vocab.itos[:10])
# 상위 빈도순으로 udtags 출력
print(UD_TAGS.vocab.freqs.most_common())
# 상위 정수 인덱스 순으로 출력
print(UD_TAGS.vocab.itos)

def tag_percentage(tag_counts): # 태그 레이블의 분포를 확인하는 함수
    total_count = sum([count for tag, count in tag_counts])
    tag_counts_percentages = [(tag, count, count/total_count) for tag, count in tag_counts]

    return tag_counts_percentages

print("Tag  Occurences Percentage\n")
for tag, count, percent in tag_percentage(UD_TAGS.vocab.freqs.most_common()):
    print(f"{tag}\t{count}\t{percent*100:4.1f}%")

BATCH_SIZE = 64

train_iterator, valid_iterator, test_iterator = legacy.data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    device = device)

batch = next(iter(train_iterator))

# 이번 모델에서는 batch_first=True를 사용하지 않으므로 배치 차원이 맨 앞이 아님.
class RNNPOSTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout): 
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional = bidirectional)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)        
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [sent len, batch size]
        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(embedded)

        # output = [sent len, batch size, hid dim * n directions]
        # hidden/cell = [n layers * n directions, batch size, hid dim]
        predictions = self.fc(self.dropout(outputs))

        # predictions = [sent len, batch size, output dim]
        return predictions

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = len(UD_TAGS.vocab)
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = RNNPOSTagger(INPUT_DIM, 
                     EMBEDDING_DIM, 
                     HIDDEN_DIM, 
                     OUTPUT_DIM, 
                     N_LAYERS, 
                     BIDIRECTIONAL, 
                     DROPOUT)

# 파라미터 개수 출력
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

pretrained_embeddings = TEXT.vocab.vectors
print(pretrained_embeddings.shape)

model.embedding.weight.data.copy_(pretrained_embeddings) # 임베딩 벡터값 copy
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
print(UNK_IDX)
print(PAD_IDX)

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM) # 0번 임베딩 벡터에는 0값을 채운다.
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM) # 1번 임베딩 벡터에는 1값을 채운다.
print(model.embedding.weight.data)

TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]
print(TAG_PAD_IDX)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)
model = model.to(device)
criterion = criterion.to(device)
prediction = model(batch.text)
prediction = prediction.view(-1, prediction.shape[-1])

def categorical_accuracy(preds, y, tag_pad_idx):
    """
    미니 배치에 대한 정확도 출력
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])

def train(model, iterator, optimizer, criterion, tag_pad_idx):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:

        text = batch.text
        tags = batch.udtags

        optimizer.zero_grad()

        #text = [sent len, batch size]     
        predictions = model(text)

        #predictions = [sent len, batch size, output dim]
        #tags = [sent len, batch size]
        predictions = predictions.view(-1, predictions.shape[-1]) # #predictions = [sent len * batch size, output dim]
        tags = tags.view(-1) # tags = [sent len * batch_size]

        #predictions = [sent len * batch size, output dim]
        #tags = [sent len * batch size]
        loss = criterion(predictions, tags)

        acc = categorical_accuracy(predictions, tags, tag_pad_idx)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, tag_pad_idx):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for batch in iterator:

            text = batch.text
            tags = batch.udtags

            predictions = model(text)

            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)

            loss = criterion(predictions, tags)

            acc = categorical_accuracy(predictions, tags, tag_pad_idx)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


N_EPOCHS = 10

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, TAG_PAD_IDX)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, TAG_PAD_IDX)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')

    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

test_loss, test_acc = evaluate(model, test_iterator, criterion, TAG_PAD_IDX)
print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')
