# NLP_Lab2 - NLP data representation
[![PYTORCH](https://img.shields.io/badge/Pytorch-1.8.1-8118AB)](https://www.python.org/downloads/release/python-360/)
[![Python 3.7](https://img.shields.io/badge/Python-3.7-3776AB)](https://www.python.org/downloads/release/python-360/)

A repository for the assignment Lab2 in NLP2022.


## Run
command.txt 참조

## Result

|data|train acc|test acc| epoch|
|:---:|:---:|:---:|:---:|
|기본|0.970|0.728|20|
|+ BatchNorm, Dropout|0.975|0.758|20|
|+ lr_scheduler|0.990|0.772 |40|


## 주의사항
cpu에서 사용을 권장함
(추후에 필요시 gpu version으로 업데이트 예정)