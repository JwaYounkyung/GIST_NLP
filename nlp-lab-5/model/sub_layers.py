"""
Todo: Code Transformer sub-Layers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model, n_head, d_prob):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        assert dim_model % n_head == 0

        self.att_size = dim_model // n_head
        self.scale = self.att_size ** -0.5

        self.linear_q = nn.Linear(dim_model, dim_model, bias=False)
        self.linear_k = nn.Linear(dim_model, dim_model, bias=False)
        self.linear_v = nn.Linear(dim_model, dim_model, bias=False)

        self.att_dropout = nn.Dropout(d_prob)

        self.output_layer = nn.Linear(n_head * self.att_size, dim_model,
                                      bias=False)

    def forward(self, q, k, v, mask):
        orig_q_size = q.size()
        d_k = d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        # Multi-Head
        q = q.view(batch_size, -1, self.n_head, d_k).permute(0, 2, 1, 3) 
        k = k.view(batch_size, -1, self.n_head, d_k).permute(0, 2, 1, 3) 
        v = v.view(batch_size, -1, self.n_head, d_v).permute(0, 2, 1, 3)

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        x = torch.matmul(q, k.transpose(2, 3))  / self.scale # [b, h, q_len, k_len]
        x.masked_fill_(mask, -1e9) # 매우 작은 음수
        x = torch.softmax(x, dim=3)
        x = torch.matmul(self.att_dropout(x), v) # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.n_head * d_v) # concat

        x = self.output_layer(x)

        assert x.size() == orig_q_size

        return x

class FeedForwardNetwork(nn.Module):
    def __init__(self, dim_model, dim_hidden, d_prob):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(dim_model, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(d_prob)
        self.layer2 = nn.Linear(dim_hidden, dim_model)


    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, dim_model, n_head, dim_hidden, d_prob):
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(dim_model, n_head, d_prob)
        self.self_attention_dropout = nn.Dropout(d_prob)
        self.self_attention_norm = nn.LayerNorm(dim_model)

        self.ffn = FeedForwardNetwork(dim_model, dim_hidden, d_prob)
        self.ffn_dropout = nn.Dropout(d_prob)
        self.ffn_norm = nn.LayerNorm(dim_model)

    def forward(self, x, mask):  
        # Encoder Self-Attention
        y = self.self_attention(x, x, x, mask)
        y = self.self_attention_dropout(y)
        x = self.self_attention_norm(x + y) # Residual connection

        # Feed Forward
        y = self.ffn(x)
        y = self.ffn_dropout(y)
        x = self.ffn_norm(x + y)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, dim_model, n_head, dim_hidden, d_prob):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(dim_model, n_head, d_prob)
        self.self_attention_dropout = nn.Dropout(d_prob)
        self.self_attention_norm = nn.LayerNorm(dim_model)

        self.enc_dec_attention = MultiHeadAttention(dim_model, n_head, d_prob)
        self.enc_dec_attention_dropout = nn.Dropout(d_prob)
        self.enc_dec_attention_norm = nn.LayerNorm(dim_model)
        
        self.ffn = FeedForwardNetwork(dim_model, dim_hidden, d_prob)
        self.ffn_dropout = nn.Dropout(d_prob)
        self.ffn_norm = nn.LayerNorm(dim_model)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Masked Decoder Self-Attention
        y = self.self_attention(x, x, x, tgt_mask)
        y = self.self_attention_dropout(y)
        x = self.self_attention_norm(x + y)

        # Encoder-Decoder Attention
        y = self.enc_dec_attention(x, enc_output, enc_output, src_mask) 
        y = self.enc_dec_attention_dropout(y)
        x = self.enc_dec_attention_norm(x + y)
        
        # Feed Forward
        y = self.ffn(x)
        y = self.ffn_dropout(y)
        x = self.ffn_norm(x + y)

        return x


class Encoder(nn.Module):
    def __init__(self, dim_model, n_head, dim_hidden, d_prob, n_enc_layer):
        super(Encoder, self).__init__()

        encoders = [EncoderLayer(dim_model, n_head, dim_hidden, d_prob)
                    for _ in range(n_enc_layer)]
        self.layers = nn.ModuleList(encoders)

    def forward(self, src, mask):
        encoder_output = src
        for enc_layer in self.layers:
            encoder_output = enc_layer(encoder_output, mask)
        return encoder_output


class Decoder(nn.Module):
    def __init__(self, dim_model, n_head, dim_hidden, d_prob, n_dec_layer):
        super(Decoder, self).__init__()

        decoders = [DecoderLayer(dim_model, n_head, dim_hidden, d_prob)
                    for _ in range(n_dec_layer)]
        self.layers = nn.ModuleList(decoders)

    def forward(self, targets, enc_output, src_mask, tgt_mask):
        decoder_output = targets
        for dec_layer in self.layers:
            decoder_output = dec_layer(decoder_output, enc_output,
                                       src_mask, tgt_mask)
        return decoder_output