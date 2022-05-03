# CSV Maker
import random
import pandas as pd

# random seed 고정
SEED = 5
random.seed(SEED)

# 1. Load input symbol sequence
train = pd.read_csv('./data/simple_seq.train.csv', sep='\n', header=None)
test = pd.read_csv('./data/simple_seq.test.csv', sep='\n', header=None)

train_input = []
train_output = []
test_input = []

# 행 마지막에 , 있는 예외 처리
for i in range(len(train)):
    last1 = train.iloc[i,0].split(',')[-1]
    last2 = train.iloc[i,0].split(',')[-2]
    if (last1==''):
        train_input.append((train.iloc[i,0][:-(len(last2)+2)]).replace(',', ' '))
        train_output.append(last2)
    else:
        train_input.append((train.iloc[i,0][:-(len(last1)+1)]).replace(',', ' '))
        train_output.append(last1)

    # 행 마지막에 ,, 있는 예외 처리
for i in range(len(test)):
    last2 = test.iloc[i,0].split(',')[-2]
    if (last2==''):
        test_input.append((test.iloc[i,0][:-2]).replace(',', ' '))
    else:
        test_input.append((test.iloc[i,0][:-1]).replace(',', ' '))

# %%
train_df = pd.DataFrame(
    {'Text': train_input,
     'Label': train_output
    })
test_df = pd.DataFrame(
    {'Text': test_input
    })

train_df.to_csv("data/train_lab1.csv", index=False)
test_df.to_csv("data/test_lab1.csv", index=False)

# %%
from torchtext import legacy #data
from imblearn.over_sampling import SMOTE
import numpy as np

TEXT = legacy.data.Field(sequential=True, batch_first=True, fix_length=20)
LABEL = legacy.data.Field(sequential=False, batch_first=True)
trainset= legacy.data.TabularDataset(
        path='data/train_lab1.csv', format='csv',
        fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
TEXT.build_vocab(trainset, min_freq=0)
LABEL.build_vocab_exceptunk(trainset)

train_iter = legacy.data.BucketIterator(
        dataset=trainset, batch_size=len(trainset),
        shuffle=True)

for idx, batch in enumerate(train_iter): # batch는 한번에 끝냄
    x, y = batch.text, batch.label # [900,20], [900]
    x, y = x.cpu().detach().numpy(), y.cpu().detach().numpy()
    
    remove_index = []
    for i, label in enumerate(y):
        if label == 15 or label == 16 or label == 17 or label == 18:
            remove_index.append(i)
    remove_index.reverse()
    for i in remove_index:
        x = np.delete(x, i, 0)
        y = np.delete(y, i)  
    
    X_samp, y_samp = SMOTE(random_state=4, k_neighbors=2).fit_resample(x, y)

# %%
# 다시 integer -> str 형식으로 저장
text_vocab = dict(map(reversed, TEXT.vocab.stoi.items()))
label_vocab = dict(map(reversed, LABEL.vocab.stoi.items()))

X_str, y_str = [], []

for text in X_samp:
    text_str = []
    for word in text:
        text_str.append(text_vocab[word])
    X_str.append(' '.join(text_str))

for i in y_samp:
    y_str.append(label_vocab[i])

# %%
train_df = pd.DataFrame(
    {'Text': X_str[:-1], #짝수 되게 하려고
     'Label': y_str[:-1]
    })

train_df.to_csv("data/train_lab1_Sampled.csv", index=False)
