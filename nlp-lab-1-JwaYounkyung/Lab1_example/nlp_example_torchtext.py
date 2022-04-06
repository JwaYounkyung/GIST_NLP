# Pytorch로 시작하는 딥러닝 입문/ 08. 자연어처리의 전처리/ 02. 토치텍스트 튜토리얼 - 영어 
# https://wikidocs.net/60314

from torchtext.legacy.data import TabularDataset, Iterator, Field

import urllib.request
import pandas as pd

# IMDB 리뷰 데이터 : 대표적인 영어 데이터
# 영화 리뷰가 긍정인지(1) 부정인지(0) 분류하는 데이터 
# urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")

df = pd.read_csv('data/IMDb_Reviews.csv', encoding='latin1')

print('전체 샘플의 개수 : {}'.format(len(df)))

train_df = df[:25000]
test_df = df[25000:]

train_df.to_csv("data/train_data.csv", index=False)
test_df.to_csv("data/test_data.csv", index=False)

# 필드 정의
TEXT = Field(sequential=True, # 순차적 데이터인지 여부(text는 단어들의 순차적인 조합)
                  use_vocab=True, # Dictionary 생성 가능 여부
                  tokenize=str.split, # Tokenization 방법
                  lower=True, # 영어 데이터 소문자화
                  batch_first=True, # 미니 배치 차원을 맨앞으로
                  fix_length=150) # 최대 허용 길이 (Padding)

LABEL = Field(sequential=False,
                   use_vocab=False, # label로는 Dictionary를 만들지 않아
                   batch_first=False,
                   is_target=True)

# 데이터를 필드 형태로 불러와 전처리
train_data, test_data = TabularDataset.splits(
        path='data', train='train_data.csv', test='test_data.csv', format='csv',
        fields=[('text', TEXT), ('label', LABEL)], skip_header=True)

print('훈련 샘플의 개수 : {}'.format(len(train_data)))
print('테스트 샘플의 개수 : {}'.format(len(test_data)))

print(vars(train_data[0]))
print(train_data.fields.items()) # 필드 구성 확인

TEXT.build_vocab(train_data, 
                    min_freq=10, # Dictionary 추가 할 단어 최소 등장 빈도
                    max_size=10000) # Dictionary 최대 크기

print('단어 집합의 크기 : {}'.format(len(TEXT.vocab))) # 특별 토큰 2개 까지 : '<unk>': 0 <- Dictionary에 없는 단어, '<pad>': 1 <- padding용
print(TEXT.vocab.stoi)

# 데이터 로더 
batch_size = 5
train_loader = Iterator(dataset=train_data, batch_size = batch_size)
test_loader = Iterator(dataset=test_data, batch_size = batch_size)

print('훈련 데이터의 미니 배치 수 : {}'.format(len(train_loader)))
print('테스트 데이터의 미니 배치 수 : {}'.format(len(test_loader)))

batch = next(iter(train_loader)) # 첫번째 미니배치
print(type(batch))
print(batch.text)
print(batch.text.shape) # torch.Size([5,150]) batch_size, fix_length