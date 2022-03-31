import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import legacy #data
import random
import pandas as pd

# random seed 고정
SEED = 5
random.seed(SEED)
torch.manual_seed(SEED)

# hyper-parameter
BATCH_SIZE = 16
lr = 0.001
EPOCHS = 100

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("cpu와 cuda 중 다음 기기로 학습함:", DEVICE)

TEXT = legacy.data.Field(sequential=True, batch_first=True, fix_length=20)
LABEL = legacy.data.Field(sequential=False, batch_first=True)

# 전체 데이터를 훈련 데이터와 테스트 데이터를 5:5 비율로 나누기
trainset, testset = legacy.data.TabularDataset.splits(
        path='data', train='train_lab1.csv', test='test_lab1.csv', format='csv',
        fields=[('text', TEXT), ('label', LABEL)], skip_header=True)

print('trainset의 구성 요소 출력 : ', trainset.fields)
print('testset의 구성 요소 출력 : ', testset.fields)

print(vars(trainset[0]))

# Dictionary 생성
TEXT.build_vocab(trainset, min_freq=5)
LABEL.build_vocab(trainset)

vocab_size = len(TEXT.vocab)
n_classes = len(LABEL.vocab)

print('단어 집합의 크기 : {}'.format(vocab_size))
print('클래스의 개수 : {}'.format(n_classes))
print(LABEL.vocab.stoi)

#데이터 로더 형성
trainset, valset = trainset.split(split_ratio=0.8)

train_iter, val_iter, test_iter = legacy.data.BucketIterator.splits(
        (trainset, valset, testset), batch_size=BATCH_SIZE,
        shuffle=True, sort=False, repeat=False)

print('훈련 데이터의 미니 배치의 개수 : {}'.format(len(train_iter)))
print('테스트 데이터의 미니 배치의 개수 : {}'.format(len(test_iter)))
print('검증 데이터의 미니 배치의 개수 : {}'.format(len(val_iter)))

# %%
class GRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
        super(GRU, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_dim, self.hidden_dim,
                          num_layers=self.n_layers,
                          batch_first=True)
        self.bnorm = nn.BatchNorm1d(n_classes)
        self.relu = nn.ReLU(inplace=True)
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):
        x = self.embed(x)
        h_0 = self._init_state(batch_size=x.size(0)) # 첫번째 히든 스테이트를 0벡터로 초기화
        x, _ = self.gru(x, h_0)  # GRU의 리턴값은 (배치 크기, 시퀀스 길이, 은닉 상태의 크기)
        x = self.relu(self.bnorm(x))
        h_t = x[:,-1,:] # (배치 크기, 은닉 상태의 크기)의 텐서로 크기가 변경됨. 즉, 마지막 time-step의 은닉 상태만 가져온다.
        self.dropout(h_t)
        logit = self.out(h_t)  # (배치 크기, 은닉 상태의 크기) -> (배치 크기, 출력층의 크기)
        return logit

    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()


model = GRU(1, 256, vocab_size, 128, n_classes, 0.5).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train(model, optimizer, train_iter):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1)  # 레이블 값을 0과 1로 변환
        optimizer.zero_grad()

        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()

def evaluate(model, val_iter):
    """evaluate model"""
    model.eval()
    corrects, total_loss = 0, 0
    for batch in val_iter:
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1) # 레이블 값을 0과 1로 변환
        logit = model(x)
        loss = F.cross_entropy(logit, y, reduction='sum')
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(val_iter.dataset) # validation data 전체 개수
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy
    
# train, evaluate
best_val_acc = None
for e in range(1, EPOCHS+1):
    train(model, optimizer, train_iter)
    val_loss, val_accuracy = evaluate(model, val_iter)

    print("[Epoch: %d] val loss : %5.2f | val accuracy : %5.2f" % (e, val_loss, val_accuracy))

    # 검증 오차가 가장 적은 최적의 모델을 저장
    if not best_val_acc or val_accuracy < best_val_acc:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(model.state_dict(), './snapshot/GRU_lab1.pt')
        best_val_acc = val_accuracy

# %%
reverse_dict = dict(map(reversed, LABEL.vocab.stoi.items()))
def generator(model, val_iter):
    """testset output generator"""
    model.eval()
    output = []
    for batch in val_iter:
        x = batch.text.to(DEVICE)
        logit = model(x)
        output.extend(logit.max(1)[1])

    return output

model.load_state_dict(torch.load('./snapshot/GRU_lab1.pt',  map_location=DEVICE))
output = generator(model, test_iter)

for i in range(len(output)):
    output[i] = reverse_dict[int(output[i])] 

submission = pd.read_csv('./data/submission_example.csv')

result_df = pd.DataFrame(
    {'id': submission['id'],
     'pred': output
    })

result_df.to_csv("data/result_lab1.csv", index=False)