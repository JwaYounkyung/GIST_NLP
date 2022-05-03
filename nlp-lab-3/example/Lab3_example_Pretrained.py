# Pretrained Word Embedding
# https://wikidocs.net/64904

from torchtext import legacy, datasets
from torchtext.vocab import GloVe

from torchtext.legacy.data import TabularDataset, Iterator, Field

import torch
from torch import nn 

import pandas as pd
import urllib.request

# urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")
df = pd.read_csv('IMDb_Reviews.csv', encoding='latin1')
print('전체 샘플의 개수 : {}'.format(len(df)))

train_df = df[:25000]
test_df = df[25000:]

train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

TEXT = legacy.data.Field(sequential=True, batch_first=True, lower=True)
LABEL = legacy.data.Field(sequential=False, batch_first=True)

train_data, test_data = TabularDataset.splits(
        path='', train='train_data.csv', test='test_data.csv', format='csv',
        fields=[('text', TEXT), ('label', LABEL)], skip_header=True)

TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300), max_size=10000, min_freq=10)
LABEL.build_vocab(train_data)

print(TEXT.vocab.stoi)
print('임베딩 벡터의 개수와 차원 : {} '.format(TEXT.vocab.vectors.shape))
print(TEXT.vocab.vectors[0]) # <unk>의 임베딩 벡터값
print(TEXT.vocab.vectors[1]) # <pad>의 임베딩 벡터값
print(TEXT.vocab.vectors[10]) # this의 임베딩 벡터값
print(TEXT.vocab.vectors[9999]) # seeing의 임베딩 벡터값

embedding_layer = nn.Embedding.from_pretrained(TEXT.vocab.vectors, freeze=False)
embedding_layer(torch.LongTensor([10])) # 단어 this의 임베딩 벡터값
