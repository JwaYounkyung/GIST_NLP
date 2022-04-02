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
# data augmentation

def random_deletion(sentence, p=0.1):
    words = sentence.split()
    n = len(words)
    
    if n == 1: # return if single word
        return words

    remaining = list(filter(lambda x: random.uniform(0,1) > p,words))
        
    if len(remaining) == 0: # if not left, choice one word
        return ' '.join([random.choice(words)])
    else:
        return ' '.join(remaining)

aug_input = [random_deletion(text) for text in train_input[:600]]

train_df = pd.DataFrame(
    {'Text': train_input[:600] + aug_input + train_input[600:], # 맨 뒤가 validation 
     'Label': train_output[:600] + train_output[:600] + train_output[600:]
    })
test_df = pd.DataFrame(
    {'Text': test_input,
     'Label': ['D20']*len(test_input) # 전처리 하기 위해 그냥 허수를 넣은것
    })

train_df.to_csv("data/train_lab1_EDA.csv", index=False)
test_df.to_csv("data/test_lab1.csv", index=False)

