import os
import math
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import legacy #data
import random
import pandas as pd

# random seed 고정
SEED = 32
random.seed(SEED)
torch.manual_seed(SEED)

# hyper-parameter
BATCH_SIZE = 16
lr = 0.1
EPOCHS = 400

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("cpu와 cuda 중 다음 기기로 학습함:", DEVICE)

TEXT = legacy.data.Field(sequential=True, batch_first=True, fix_length=20)
LABEL = legacy.data.Field(sequential=False, batch_first=True)

trainset= legacy.data.TabularDataset(
        path='data/train_lab1.csv', format='csv',
        fields=[('text', TEXT), ('label', LABEL)], skip_header=True)

testset= legacy.data.TabularDataset(
        path='data/test_lab1.csv', format='csv',
        fields=[('text', TEXT)], skip_header=True)

print('trainset의 구성 요소 출력 : ', trainset.fields)
print('testset의 구성 요소 출력 : ', testset.fields)

print(vars(trainset[0]))

# Dictionary 생성
TEXT.build_vocab(trainset, min_freq=0)
LABEL.build_vocab_exceptunk(trainset)

vocab_size = len(TEXT.vocab)
n_classes = len(LABEL.vocab)

print('단어 집합의 크기 : {}'.format(vocab_size))
print('클래스의 개수 : {}'.format(n_classes))
print(LABEL.vocab.stoi)

#데이터 로더 형성
# trainset, valset = trainset.split(split_ratio=1)
# print(vars(valset[0]))

train_iter = legacy.data.BucketIterator(
        dataset=trainset, batch_size=BATCH_SIZE,
        shuffle=True)

test_iter = legacy.data.BucketIterator(
        dataset=testset, batch_size=BATCH_SIZE,
        shuffle=False)


print('train 데이터의 미니 배치의 개수 : {}'.format(len(train_iter))) 
# print('validate 데이터의 미니 배치의 개수 : {}'.format(len(val_iter)))
print('test 데이터의 미니 배치의 개수 : {}'.format(len(test_iter))) 

# %%
# Linear
class myLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, input):
        x, y = input.shape
        if y != self.in_features:
            print(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
            return 0
        output = input.matmul(self.weight.t())
        if self.bias is not None:
            output += self.bias
        ret = output
        return ret
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
# %%
# Model
class TextClassificationModel2(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = myLinear(20*embed_dim, 1024)
        self.fc2 = myLinear(1024, 128)
        self.fc3 = myLinear(128, num_class)
        
        self.embedding.weight = nn.Parameter(torch.zeros(vocab_size, embed_dim))
        
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.4)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(128)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.flatten(x,1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


emsize = 256
model = TextClassificationModel2(vocab_size, emsize, n_classes).to(DEVICE)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.999)

# %%
# train & evaluate
def train(train_iter):
    model.train()
    total_loss, total_acc, total_count = 0, 0, 0
    log_interval = 20
    start_time = time.time()

    for idx, batch in enumerate(train_iter):
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        optimizer.zero_grad()

        logit = model(x)
        loss = criterion(logit, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # gradient exploding 방지
        optimizer.step()

        total_loss += int(loss.data)
        total_acc += (logit.argmax(1) == y).sum().item()
        total_count += y.size(0)

        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            # 누적 accuracy
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(train_iter),
                                              total_acc/total_count))
            start_time = time.time()

    return total_loss/total_count, total_acc/total_count

def evaluate(val_iter):
    """evaluate model"""
    model.eval()
    criterion.eval()
    total_loss, total_acc, total_count = 0, 0, 0

    with torch.no_grad():
        for batch in val_iter:
            x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)

            logit = model(x)
            loss = criterion(logit, y)

            total_loss += int(loss.data)
            total_acc += (logit.argmax(1) == y).sum().item()
            total_count += y.size(0)

    return total_loss/total_count, total_acc/total_count
    

best_train_acc = None
train_loss_list, train_acc_list = [], []
# val_loss_list, val_acc_list = [], []
for epoch in range(1, EPOCHS+1):
    epoch_start_time = time.time()
    train_loss, train_acc = train(train_iter)
    # val_loss, val_acc = evaluate(val_iter)
    
    if best_train_acc is not None and best_train_acc > train_acc: # accuracy가 업데이트가 안되었을 때
        scheduler.step()
    else: # best_train_acc를 가진 최적의 모델을 저장(같은 값일 때도 update)
        if not os.path.isdir("results"):
            os.makedirs("results")
        torch.save(model.state_dict(), './results/lab1_Problem2.pt')
        best_train_acc = train_acc
    
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'train accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           train_acc))
    print('-' * 59)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    # val_loss_list.append(val_loss)
    # val_acc_list.append(val_acc)

# torch.save(model.state_dict(), './snapshot/GRU_lab1.pt')
print('best validation accuracy',  best_train_acc)

# %% 
# Graph
def graph(train_list, fgname):
    plt.plot(train_list, label='train')
    #plt.plot(val_list, label='validation')
    plt.legend()
    plt.savefig('./results/'+ fgname + '.png', dpi=300)
    plt.clf()


graph(train_loss_list, 'loss_Problem2')
graph(train_acc_list, 'acc_Problem2')

# %%
def generator(test_iter):
    """testset output generator"""
    model.eval()
    criterion.eval()
    output = []

    with torch.no_grad():
        for idx, batch in enumerate(test_iter):
            x = batch.text.to(DEVICE) # test label은 없어서 y 안불러옴
            logit = model(x) # Text.vocab update도 이루어짐
            output.extend(logit.argmax(1))

    return output

model.load_state_dict(torch.load('./results/lab1_Problem2.pt',  map_location=DEVICE))

reverse_dict = dict(map(reversed, LABEL.vocab.stoi.items()))
output = generator(test_iter)

for i in range(len(output)):
    output[i] = reverse_dict[int(output[i])] 

submission = pd.read_csv('./data/submission_example.csv')
result_df = pd.DataFrame(
    {'id': submission['id'],
     'pred': output
    })
result_df.to_csv("results/result_lab1_Problem2.csv", index=False)