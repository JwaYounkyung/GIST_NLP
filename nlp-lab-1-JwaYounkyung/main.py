# One-hot representation

import pandas as pd


def dic_maker(train, test):
    # squeeze 

    word_dict = 0
    return word_dict


if __name__ == '__main__':
    # 1. Load input symbol sequence
    train = pd.read_csv('./data/simple_seq.train.csv', sep='\n', header=None)
    test = pd.read_csv('./data/simple_seq.test.csv', sep='\n', header=None)

    train_input = []
    train_output = []
    test_input = []

    # 행 마지막에 , 있는 예외 처리
    for i in range(len(train)):
        if (train.iloc[i,0].split(',')[-1]==''):
            train_input.append(train.iloc[i,0].split(',')[:-2])
            train_output.append(train.iloc[i,0].split(',')[-2])
        else:
            train_input.append(train.iloc[i,0].split(',')[:-1])
            train_output.append(train.iloc[i,0].split(',')[-1])

    for i in range(len(test)):
        # 행 마지막에 ,, | , 예외 처리
        test_input.append(list(filter(None,test.iloc[i,0].split(',')))[:-1])

    # 2. Generate dictionary
    w_dict = dic_maker(train_input, test_input)

    print('hi')
