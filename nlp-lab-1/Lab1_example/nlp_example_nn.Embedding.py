# Pytorch로 시작하는 딥러닝 입문/ 09. 단어의 표현 방법/ 07. PyTorch의 nn.Embedding()
# https://wikidocs.net/64779

import torch
import torch.nn as nn

# %%
# 1. nn.Embedding 사용 안함 

train_data = 'you need to know how to code'

# 중복을 제거한 단어들의 집합인 단어 집합 생성.
word_set = set(train_data.split()) # type 'set'

# 단어 집합의 각 단어에 고유한 정수 맵핑.
vocab = {word: i+2 for i, word in enumerate(word_set)}
vocab['<unk>'] = 0
vocab['<pad>'] = 1
print(vocab)

# 단어 집합의 크기(8)만큼의 행을 가지는 테이블 생성.
# embedding vector size = 3
embedding_table = torch.FloatTensor([
                               [ 0.0,  0.0,  0.0],
                               [ 0.0,  0.0,  0.0],
                               [ 0.2,  0.9,  0.3],
                               [ 0.1,  0.5,  0.7],
                               [ 0.2,  0.1,  0.8],
                               [ 0.4,  0.1,  0.1],
                               [ 0.1,  0.8,  0.9],
                               [ 0.6,  0.1,  0.1]])

sample = 'you need to run'.split()
idxes = []

# 각 단어를 정수로 변환
for word in sample:
  try:
    idxes.append(vocab[word])
  # 단어 집합에 없는 단어일 경우 <unk>로 대체된다.
  except KeyError:
    idxes.append(vocab['<unk>'])
idxes = torch.LongTensor(idxes)

# 각 정수를 인덱스로 임베딩 테이블에서 값을 가져온다.
lookup_result = embedding_table[idxes, :]
print(lookup_result)

# %%
# 2. nn.Embedding 사용

# 위에 있는 embedding_table을 만드는격
embedding_layer = nn.Embedding(num_embeddings=len(vocab), # dictionary 크기
                               embedding_dim=3, # embedding vector dimension(임의로 정할 수 있음 128등)
                               padding_idx=1) # <pad> index

print(embedding_layer.weight)
