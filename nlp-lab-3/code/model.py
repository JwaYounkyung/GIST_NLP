import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class CharEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, _weight=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        if _weight is None:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = nn.Parameter(_weight)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, input):
        batch_weight = [self.weight for i in range(input.shape[1])]
        batch_weight = torch.stack(batch_weight)

        bmm_list = []
        for word in input:
            bmm = torch.bmm(word.float(), batch_weight)
            bmm = torch.flatten(bmm, 1)
            bmm_list.append(bmm.unsqueeze(1))
        
        ret = torch.cat(bmm_list, dim=1)

        return ret

    def extra_repr(self):
        s = '{num_embeddings}, {embedding_dim}'
        return s.format(**self.__dict__)


class CNN_NLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, word_len,
                 filter_sizes=[2, 3, 4],
                 num_filters=[100, 100, 100],
                 num_classes=2,
                 dropout=0.5):

        super(CNN_NLP, self).__init__()
        self.charembedding = CharEmbedding(vocab_size, embed_dim)

        self.conv1d_list = nn.ModuleList([nn.Conv1d(in_channels=embed_dim*word_len,
                                                    out_channels=num_filters[i],
                                                    kernel_size=filter_sizes[i])
                                                    for i in range(len(filter_sizes))
        ])
        self.norm_list = nn.ModuleList([nn.BatchNorm1d(num_features=num_filters[i]) 
                                                       for i in range(len(num_filters))
        ])

        self.fc1 = nn.Linear(np.sum(num_filters), 100)
        self.fc2 = nn.Linear(100, num_classes)
        self.bn1 = nn.BatchNorm1d(100)
        self.dropout1 = nn.Dropout(p=dropout)
    
    def forward(self, x):
        # character embedding
        x = x.permute(1, 0, 2, 3)
        x = self.charembedding(x)

        # CNN
        x = x.permute(0, 2, 1)
        x = [conv1d(x) for conv1d in self.conv1d_list]
        x = [F.relu(self.norm_list[i](x[i])) for i in range(len(x))]
        x = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x]
        x = torch.cat([x_pool.squeeze(dim=2) for x_pool in x], dim=1)

        # fc
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.softmax(self.fc2(x), dim=1)

        return x