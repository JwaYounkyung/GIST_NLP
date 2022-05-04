# NLP_Lab3 - RNN
[![PYTORCH](https://img.shields.io/badge/Pytorch-1.8.1-8118AB)](https://www.python.org/downloads/release/python-360/)
[![Python 3.7](https://img.shields.io/badge/Python-3.7-3776AB)](https://www.python.org/downloads/release/python-360/)

A repository for the assignment Lab3 in NLP2022.


## Run
command.txt 참조

## Result

### Problem 1 
|Model|train acc|test acc| epoch|
|:---:|:---:|:---:|:---:|
|my RNN|0.886|0.788|150|


### Problem 2
|Model|train acc|test acc| epoch|
|:---:|:---:|:---:|:---:|
|my RNN|0.875|0.539|150|
|my RNN|0.919|0.544|250|
|nn.RNN|0.985|0.649|150|
|nn.RNN|0.997|0.656|250|
|nn.RNN + sent_len 65|0.996|0.666|250|
