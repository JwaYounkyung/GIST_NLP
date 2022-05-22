from typing import List, Tuple, Optional, overload, Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from torch.autograd import Variable

def apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
    return tensor.index_select(dim, permutation)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias=True):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias

        self.rnn_cell_list = nn.ModuleList()
        self.rnn_cell_list.append(nn.LSTMCell(self.input_size,
                                            self.hidden_size,
                                            self.bias))
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(nn.LSTMCell(self.hidden_size,
                                                self.hidden_size,
                                                self.bias))

    def permute_hidden(self, hx: Tuple[Tensor, Tensor], permutation: Optional[Tensor]) -> Tuple[Tensor, Tensor]:  # type: ignore
        if permutation is None:
            return hx
        return apply_permutation(hx[0], permutation), apply_permutation(hx[1], permutation)

    def forward(self, input, hx=None):
        orig_input = input
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            h0 = torch.zeros(self.num_layers, max_batch_size, self.hidden_size, dtype=input.dtype, device=input.device)
            c0 = torch.zeros(self.num_layers, max_batch_size, self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (h0, c0)
        else:
            hx = self.permute_hidden(hx, sorted_indices)

        outs = []
        hs, cs = [], []
        hidden = []
        
        if batch_sizes != None: # batch_sizes 사용
            hss, css = [], [] 
            h_t, c_t = [], []

            cur_batch = 0
            for layer in range(self.num_layers):
                hs.append(hx[0][layer, :, :])
                cs.append(hx[1][layer, :, :])

            for t in range(len(batch_sizes)): # sentence length
                for layer in range(self.num_layers):
                    if layer == 0:
                        h, c = self.rnn_cell_list[layer](
                            input[cur_batch:cur_batch+batch_sizes[t]],
                            (hs[layer][:batch_sizes[t]],cs[layer][:batch_sizes[t]])
                            )
                    else:
                        h, c = self.rnn_cell_list[layer](
                            hs[layer - 1], # input : 전 layer의 h
                            (hs[layer][:batch_sizes[t]],cs[layer][:batch_sizes[t]])
                            )
                
                    hs[layer] = h
                    cs[layer] = c

                cur_batch += batch_sizes[t]
                hss.append(hs[:])
                css.append(cs[:])
                outs.append(h) # 마지막 layer의 h

            output = torch.cat(outs)

            # PackedSequence Hidden state 추출
            for t in range(len(batch_sizes)):
                for layer in range(self.num_layers):
                    if t==0:
                        hs[layer] = hss[-t-1][layer]
                        cs[layer] = css[-t-1][layer]
                    else:
                        hs[layer] = hss[-t-1][layer][batch_sizes[-t]:batch_sizes[-t-1]]
                        cs[layer] = css[-t-1][layer][batch_sizes[-t]:batch_sizes[-t-1]]
                
                h_t.append(torch.stack(hs))
                c_t.append(torch.stack(cs))

            hs = torch.cat(h_t, dim=1)
            cs = torch.cat(c_t, dim=1)
            hidden = (hs, cs)                
        else:
            # 모든 layer에 같은 h, c가 들어감
            for layer in range(self.num_layers):
                hs.append(hx[0][layer, :, :])
                cs.append(hx[1][layer, :, :])

            for t in range(input.size(1)): # sentence length
                for layer in range(self.num_layers):
                    if layer == 0:
                        h, c = self.rnn_cell_list[layer](
                            input[:, t, :],
                            (hs[layer],cs[layer])
                            )
                    else:
                        h, c = self.rnn_cell_list[layer](
                            hs[layer - 1], # input : 전 layer의 h
                            (hs[layer],cs[layer])
                            )
                    hs[layer] = h
                    cs[layer] = c

                outs.append(h) # 마지막 layer의 h

            output = torch.stack(outs, dim=1)
            hs = torch.stack(hs)
            cs = torch.stack(cs)
            hidden = (hs, cs)

        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            return output, self.permute_hidden(hidden, unsorted_indices)


class Encoder(nn.Module):
    def __init__(self, vocab_size, hid_dim, n_layers, **kwargs):
        super(Encoder, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, hid_dim)
        self.dropout = nn.Dropout(0.5)
        self.rnn = nn.LSTM(hid_dim, hid_dim, n_layers, batch_first=True)
        # self.rnn = LSTM(hid_dim, hid_dim, n_layers)

    def forward(self, x):
        """ TO DO: feed the unpacked input x to Encoder """
        
        inputs_length = torch.sum(torch.where(x > 0, True, False), dim=1)
        # inputs_length, sorted_idx = inputs_length.sort(0, descending=True)
        # x = x[sorted_idx]
        x = self.dropout(self.embedding(x))
        packed = pack(x, inputs_length.tolist(), batch_first=True, enforce_sorted=False)
        output, state = self.rnn(packed)
        output, outputs_length = unpack(output, batch_first=True, total_length=x.shape[1])
        '''
        x = self.dropout(self.embedding(x))
        output, state = self.rnn(x)
        '''
        return output, state
	

class Decoder(nn.Module):
    def __init__(self, vocab_size, hid_dim, n_layers=4, **kwargs):
        super(Decoder, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.output_dim = vocab_size

        self.embedding = nn.Embedding(vocab_size, hid_dim)
        self.dropout = nn.Dropout(0.5)
        # self.rnn = nn.LSTM(hid_dim, hid_dim, n_layers, batch_first=True)
        self.rnn = LSTM(hid_dim, hid_dim, n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hid_dim, vocab_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, enc_outputs, x, state):
        """ TO DO: feed the input x to Decoder """
        x = x.unsqueeze(1)
        x = self.dropout(self.embedding(x))
        output, state = self.rnn(x, state)
        output = self.classifier(output.squeeze())

        return output, state


class AttnDecoder(nn.Module):
    def __init__(self, vocab_size, hid_dim, max_len, n_layers=4, **kwargs):
        super(AttnDecoder, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.output_dim = vocab_size

        self.embedding = nn.Embedding(vocab_size, hid_dim)
        self.dropout = nn.Dropout(0.5)
        # self.rnn = nn.LSTM(hid_dim, hid_dim, n_layers, batch_first=True)
        self.rnn = LSTM(hid_dim, hid_dim, n_layers)
        self.softmax = nn.Softmax(dim=1)
        self.classifier = nn.Sequential(
            nn.Linear(hid_dim*2, vocab_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, enc_outputs, x, state):
        """ TO DO: feed the input x to Decoder """
        x = x.unsqueeze(1)
        x = self.dropout(self.embedding(x))
        output, state = self.rnn(x, state)
        
        # Attension
        score = torch.bmm(enc_outputs, torch.transpose(output, 1, 2))
        dist = self.softmax(score)
        value = torch.bmm(torch.transpose(dist, 1, 2), enc_outputs)
        concat = torch.cat([value, output], dim=2)

        output = self.classifier(concat.squeeze())

        return output, state


# Seq2Seq
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, Auto=True):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.Auto = Auto

        assert encoder.hid_dim == decoder.hid_dim, \
            'Hidden dimensions of encoder decoder must be equal'
        assert encoder.n_layers == decoder.n_layers, \
            'Encoder and decoder must have equal number of layers'

    def forward(self, src, trg, teacher_force=True): 
        batch_size = trg.shape[0]
        trg_len = trg.shape[1] # 타겟 토큰 길이 얻기
        trg_vocab_size = self.decoder.output_dim # context vector의 차원

        # h0 = torch.zeros(self.encoder.n_layers, batch_size, self.encoder.hid_dim, requires_grad=True)
        # c0 = torch.zeros(self.encoder.n_layers, batch_size, self.encoder.hid_dim, requires_grad=True)

        # decoder의 output을 저장하기 위한 tensor
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # Encode
        enc_outputs, (h, c) = self.encoder(src)

        # 첫 번째 입력값 <eos> 토큰
        input = trg[:,0]

        # Decode
        for t in range(1,trg_len): # <eos> 제외하고 trg_len-1 만큼 반복
            output, (h, c) = self.decoder(enc_outputs, input, (h, c))
            # prediction 저장
            outputs[:,t] = output
            # 가장 높은 확률을 갖은 값 얻기
            top1 = output.argmax(1)
            # teacher forcing의 경우에 다음 lstm에 target token 입력
            if self.Auto:
                input = trg[:,t] if teacher_force else top1

        return outputs
        	


