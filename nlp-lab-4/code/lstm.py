import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class Encoder(nn.Module):
    def __init__(self, vocab_size, hid_dim, n_layers=4, **kwargs):
        super(Encoder, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, hid_dim)
        """ TO DO: Implement your LSTM """
        self.rnn = nn.LSTM(hid_dim, hid_dim, n_layers, batch_first=True)

    def forward(self, x):
        """ TO DO: feed the unpacked input x to Encoder """
        inputs_length = torch.sum(torch.where(x > 0, True, False), dim=1)
        x = self.embedding(x)
        packed = pack(x, inputs_length.tolist(), batch_first=True, enforce_sorted=False)
        output, state = self.rnn(packed)
        output, outputs_length = unpack(output, batch_first=True, total_length=x.shape[1])

        return output, state
	

class Decoder(nn.Module):
    def __init__(self, vocab_size, hid_dim, n_layers=4, **kwargs):
        super(Decoder, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.output_dim = vocab_size

        self.embedding = nn.Embedding(vocab_size, hid_dim)
        """ TO DO: Implement your LSTM """
        self.rnn = nn.LSTM(hid_dim, hid_dim, n_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hid_dim, vocab_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, enc_outputs, x, state):
        """ TO DO: feed the input x to Decoder """
        x = x.unsqueeze(1)
        x = self.embedding(x)
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
        """ TO DO: Implement your LSTM """
        self.rnn = nn.LSTM(hid_dim, hid_dim, n_layers, batch_first=True)

        self.softmax = nn.Softmax(dim=1)

        self.classifier = nn.Sequential(
            nn.Linear(hid_dim*2, vocab_size),
            nn.Tanh(),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, enc_outputs, x, state):
        """ TO DO: feed the input x to Decoder """
        x = x.unsqueeze(1)
        x = self.embedding(x)
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
        	


