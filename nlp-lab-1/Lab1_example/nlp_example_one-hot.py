# Pytorch로 시작하는 딥러닝 입문/ 09. 단어의 표현 방법/ 01. NLP에서의 원-핫 인코딩
# https://wikidocs.net/60853

from konlpy.tag import Okt  

okt = Okt()  
token = okt.morphs("나는 자연어 처리를 배운다") # list
print(token)

word2index = {}
for voca in token:
     if voca not in word2index.keys():
       word2index[voca] = len(word2index)
print(word2index)

def one_hot_encoding(word, word2index):
       one_hot_vector = [0]*(len(word2index))
       index = word2index[word]
       one_hot_vector[index] = 1
       return one_hot_vector

vector = one_hot_encoding("자연어",word2index)
print(vector)