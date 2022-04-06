
# NLP_Lab1
[![TORCHTEXT](https://img.shields.io/badge/Torchtext-0.8.1-811zAB)](https://www.python.org/downloads/release/python-360/)
[![PYTORCH](https://img.shields.io/badge/Pytorch-1.8.1-8118AB)](https://www.python.org/downloads/release/python-360/)
[![Python 3.7](https://img.shields.io/badge/Python-3.7-3776AB)](https://www.python.org/downloads/release/python-360/)

A repository for the assignment Lab1 in NLP2022.


## 모듈 수정
`torchtext.legacy.data.OneHotField`, `torchtext.legacy.data.Field.buil_vocab_exceptunk`

를 추가하기 위한 torchtext 모듈 수정이 필요함

1. `torchtext/legacy/data/field.py`에 `Field.py` 전체 복사 붙여넣기

2. `torchtext/legacy/data/__init__.py`에 `Field__init__.py` 전체 복사 붙여넣기

## 시행 하기
1. `torchtext.legacy.data.TabularDatase`에 맞는 csv형식을 맞추기 위한 전처리 코드
    + `csv_modification.py` : 기본 전처리
    + `csv_modification_EDA.py` : random deletion data augmentation 사용한 전처리
    + `csv_modification_imbalance.py`: over sampling을 사용한 전처리
    + `csv_modification_mix.py` : data augmentation과 over sampling을 같이 사용한 전처리
2. `lab1_Problem1` 코드 시행 (Sampled data 사용)
3. `lab1_Problem2` 코드 시행 (Sampled data 사용)

## 결과
모든 실험의 epoch은 400으로 고정
### Problem1
|data|train acc|test acc|
|:---:|:---:|:---:|
|기본|0.908|0.44|
|EDA|0.946|0.48|
|Sampled|0.936|0.51|
### Problem2
|data|train acc|test acc|
|:---:|:---:|:---:|
|기본|1.0|0.55|
|Sampled|1.0|0.57|
