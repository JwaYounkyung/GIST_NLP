import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class Encoder(nn.Module):
	def __init__(self, vocab_size, hidden_size, num_layers=4, **kwargs):
		super(Encoder, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.embedding = nn.Embedding(vocab_size, hidden_size)
		""" TO DO: Implement your LSTM """
		self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
	
	def forward(self, x, state):
		""" TO DO: feed the unpacked input x to Encoder """
		inputs_length = torch.sum(torch.where(x > 0, True, False), dim=1)
		x = self.embedding(x)
		packed = pack(x, inputs_length.tolist(), batch_first=True, enforce_sorted=False)
		output, state = self.rnn(packed, state)
		output, outputs_length = unpack(output, batch_first=True, total_length=x.shape[1])
	
		return output, state
	

class Decoder(nn.Module):
	def __init__(self, vocab_size, hidden_size, num_layers=4, **kwargs):
		super(Decoder, self).__init__()
		self.hidden_size = hidden_size
		
		self.embedding = nn.Embedding(vocab_size, hidden_size)
		""" TO DO: Implement your LSTM """
		self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
		self.classifier = nn.Sequential(
			nn.Linear(hidden_size, vocab_size),
			nn.LogSoftmax(dim=-1)
		)
	
	def forward(self, x, state):
		""" TO DO: feed the input x to Decoder """
		x = self.embedding(x)
		output, state = self.rnn(x, state)
		output = self.classifier(output)
	
		return output#, state


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        # encoder와 decoder의 hid_dim이 일치하지 않는 경우 에러메세지
        assert encoder.hid_dim == decoder.hid_dim, \
            'Hidden dimensions of encoder decoder must be equal'
        # encoder와 decoder의 hid_dim이 일치하지 않는 경우 에러메세지
        assert encoder.n_layers == decoder.n_layers, \
            'Encoder and decoder must have equal number of layers'

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [src len, batch size]
        # trg: [trg len, batch size]
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0] # 타겟 토큰 길이 얻기
        trg_vocab_size = self.decoder.output_dim # context vector의 차원

        # decoder의 output을 저장하기 위한 tensor
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # initial hidden state
        hidden, cell = self.encoder(src)

        # 첫 번째 입력값 <sos> 토큰
        input = trg[0,:]

        for t in range(1,trg_len): # <eos> 제외하고 trg_len-1 만큼 반복
            output, hidden, cell = self.decoder(input, hidden, cell)

            # prediction 저장
            outputs[t] = output

            # teacher forcing을 사용할지, 말지 결정
            teacher_force = random.random() < teacher_forcing_ratio

            # 가장 높은 확률을 갖은 값 얻기
            top1 = output.argmax(1)

            # teacher forcing의 경우에 다음 lstm에 target token 입력
            input = trg[t] if teacher_force else top1

        return outputs

class AttnDecoder(nn.Module):
	pass

