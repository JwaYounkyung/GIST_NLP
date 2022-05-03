import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=3, batch_first=True, dropout=0)
        self.fc = nn.Linear(hidden_dim*3, output_dim)
        
    def forward(self, x):
        output, hidden = self.rnn(x)
        hidden = hidden.permute(1, 0, 2)
        hidden = torch.flatten(hidden, 1)
        out = self.fc(hidden)
        return out