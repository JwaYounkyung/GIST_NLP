import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import pad_mask, masked_attn_mask
from model.sub_layers import Encoder, Decoder

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class Transformer(nn.Module):
    def __init__(self, num_token_src, num_token_tgt, max_seq_len, dim_model, n_head=8, dim_hidden=2048, d_prob=0.1, n_enc_layer=6, n_dec_layer=6):
        super(Transformer, self).__init__()

        """
        each variable is one of example, so you can change it, it's up to your coding style.
        """
        self.max_seq_len = max_seq_len
        self.dim_model = dim_model
        
        self.src_pad_idx = None
        self.tgt_pad_idx = None

        self.emb_scale = dim_model ** 0.5

        # self.embed = Embedding_Layer(num_token=num_token, dim_model=dim_model, max_seq_len=max_seq_len, d_prob=d_prob)
        # self.encoder = Encoder(dim_model, d_k, d_v, n_head, dim_hidden, d_prob, n_enc_layer)
        # self.decoder = Decoder(dim_model, d_k, d_v, n_head, dim_hidden, d_prob, n_dec_layer)
        
        self.s_vocab_embedding = nn.Embedding(num_token_src, dim_model)
        self.t_vocab_embedding = nn.Embedding(num_token_tgt, dim_model)

        self.s_emb_dropout = nn.Dropout(d_prob)
        self.t_emb_dropout = nn.Dropout(d_prob)

        # For positional encoding
        num_timescales = dim_model // 2
        max_timescale = 10000.0
        min_timescale = 1.0
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) / max(num_timescales - 1, 1))
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales) # optimizer의 영향을 받지 않기 위해

        self.encoder = Encoder(dim_model, n_head, dim_hidden, d_prob, n_enc_layer)
        self.decoder = Decoder(dim_model, n_head, dim_hidden, d_prob, n_dec_layer)
        self.classifier = nn.Sequential(
            nn.Linear(dim_model, num_token_tgt),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, src, tgt):
        src_mask = None # pad_mask(src, self.src_pad_idx)
        enc_output = self.encode(src, src_mask)

        t_mask = None # pad_mask(tgt, self.tgt_pad_idx)
        t_size = tgt.size()[1]
        t_self_mask = None # masked_attn_mask(t_size, tgt.device)
        output = self.decode(tgt[:,:-1], enc_output, src_mask, t_self_mask, t_mask)
        output = self.classifier(output)

        return output
    
    def encode(self, src, src_mask):
        # Input embedding
        src_embedded = self.s_vocab_embedding(src)
        # src_embedded.masked_fill_(i_mask.squeeze(1).unsqueeze(-1), 0) # mask 된곳을 0으로
        src_embedded = src_embedded*self.emb_scale + self.get_position_encoding(src)
        src_embedded = self.s_emb_dropout(src_embedded)

        return self.encoder(src_embedded, src_mask)
    
    def decode(self, targets, enc_output, src_mask, t_self_mask, t_mask):
        # target embedding
        target_embedded = self.t_vocab_embedding(targets)
        # target_embedded.masked_fill_(t_mask.squeeze(1).unsqueeze(-1), 0)

        # Shifting
        # target_embedded = target_embedded[:, :-1] # 맨 뒤를 잘라
        # target_embedded = F.pad(target_embedded, (0, 0, 1, 0))
        target_embedded = target_embedded*self.emb_scale + self.get_position_encoding(targets)
        target_embedded = self.t_emb_dropout(target_embedded)

        # decoder
        decoder_output = self.decoder(target_embedded, enc_output, src_mask, t_self_mask)

        return decoder_output

    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32,
                                device=x.device)
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1)
        signal = F.pad(signal, (0, 0, 0, self.dim_model % 2))
        signal = signal.view(1, max_length, self.dim_model)
        return signal

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)