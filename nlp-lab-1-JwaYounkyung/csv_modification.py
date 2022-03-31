# CSV Maker

import pandas as pd

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
    
train_df = pd.DataFrame(
    {'Text': train_input,
     'Label': train_output
    })
test_df = pd.DataFrame(
    {'Text': test_input,
     'Label': [0]*len(test_input)
    })

train_df.to_csv("data/train_lab1.csv", index=False)
test_df.to_csv("data/test_lab1.csv", index=False)

