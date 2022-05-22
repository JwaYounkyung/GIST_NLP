import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import time, datetime
import argparse
import numpy as np
from pathlib import Path

import utils, dataloader, lstm

import pandas as pd

parser = argparse.ArgumentParser(description='NMT - Seq2Seq with Attention')
""" recommend to use default settings """
# environmental settings
parser.add_argument('--gpu-id', type=int, default=0)
parser.add_argument('--seed-num', type=int, default=0)
parser.add_argument('--save', action='store_true', default=0)
parser.add_argument('--res-dir', default='nlp-lab-4/result', type=str)
parser.add_argument('--res-tag', default='seq2seq', type=str)
# architecture
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--max-len', type=int, default=20)
parser.add_argument('--hidden-size', type=int, default=512)
parser.add_argument('--max-norm', type=float, default=5.0)
# hyper-parameters
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
# option
parser.add_argument('--autoregressive', action='store_true', default=False)
parser.add_argument('--teacher-forcing', action='store_true', default=False)
parser.add_argument('--attn', action='store_true', default=True)
# etc
parser.add_argument('--k', type=int, default=4, help='hyper-paramter for BLEU score')

args = parser.parse_args()


utils.set_random_seed(seed_num=args.seed_num)

use_cuda = utils.check_gpu_id(args.gpu_id)
device = torch.device('cuda:{}'.format(args.gpu_id) if use_cuda else 'cpu')

t_start = time.time()

vocab_src = utils.read_pkl('nlp-lab-4/data/de-en/nmt_simple.src.vocab.pkl')
vocab_tgt = utils.read_pkl('nlp-lab-4/data/de-en/nmt_simple.tgt.vocab.pkl')

tr_dataset = dataloader.NMTSimpleDataset(max_len=args.max_len,
										 src_filepath='nlp-lab-4/data/de-en/nmt_simple.src.train.txt',
										 tgt_filepath='nlp-lab-4/data/de-en/nmt_simple.tgt.train.txt',
										 vocab=(vocab_src, vocab_tgt))
ts_dataset = dataloader.NMTSimpleDataset(max_len=args.max_len,
										 src_filepath='nlp-lab-4/data/de-en/nmt_simple.src.test.txt',
										 vocab=(tr_dataset.vocab_src, tr_dataset.vocab_tgt))
vocab_src = tr_dataset.vocab_src
vocab_tgt = tr_dataset.vocab_tgt
i2w_src = {v:k for k, v in vocab_src.items()}
i2w_tgt = {v:k for k, v in vocab_tgt.items()}

tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=2)
ts_dataloader = DataLoader(ts_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=2)

encoder = lstm.Encoder(len(vocab_src), args.hidden_size, args.num_layers)
if not args.attn:
	decoder = lstm.Decoder(len(vocab_tgt), args.hidden_size, args.num_layers)
else:
	decoder = lstm.AttnDecoder(len(vocab_tgt), args.hidden_size, max_len=args.max_len, num_layers=args.num_layers)

model = lstm.Seq2Seq(encoder, decoder, device, Auto=True).to(device)
utils.init_weights(model, init_type='uniform')

""" TO DO: (masking) convert this line for masking [PAD] token """
criterion = nn.NLLLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

best_acc = None
def train(model, dataloader, epoch, model_root):
	model.train()
	tr_loss = 0.
	correct = 0

	cnt = 0
	prev_time = time.time()

	global best_acc

	for idx, (src, tgt) in enumerate(dataloader):
		src, tgt = src.to(device), tgt.to(device)

		optimizer.zero_grad()
		outputs = model(src, tgt, teacher_force=False)

		# eos 제외하고 loss 계산
		outputs = outputs[:,1:,:].reshape(args.batch_size * (args.max_len-1), -1)
		tgt = tgt[:,1:].reshape(-1)

		loss = criterion(outputs, tgt)
		tr_loss += loss.item()
		loss.backward()

		""" TO DO: (clipping) convert this line for clipping the 'gradient < args.max_norm' """
		torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm) # 0.5
		optimizer.step()

		# accuracy
		pred = outputs.argmax(dim=1, keepdim=True)
		pred_acc = pred[tgt != 0] # pad가 아닌것만 정확도 추출
		tgt_acc = tgt[tgt != 0]
		correct += pred_acc.eq(tgt_acc.view_as(pred_acc)).sum().item()

		cnt += tgt_acc.shape[0] # number of non pad 

		# verbose
		batches_done = (epoch - 1) * len(dataloader) + idx
		batches_left = args.n_epochs * len(dataloader) - batches_done
		time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
		prev_time = time.time()

	tr_loss /= cnt
	tr_acc = correct / cnt
	
	print("[epoch {:3d}/{:3d}] loss: {:.6f}, accuracy: {:.2f}".format(
		epoch, args.n_epochs, tr_loss, tr_acc*100), end='\n')
		
	if best_acc is None or tr_acc >= best_acc:
		torch.save(model.state_dict(), model_root)
		best_acc = tr_acc 

	return tr_loss, tr_acc


def test(model, dataloader, lengths=None):
	model.eval()
	idx = 0
	total_pred = []

	prev_time = time.time()
	with torch.no_grad():
		for src, tgt in dataloader:
			src, tgt = src.to(device), tgt.to(device)

			outputs = model(src, tgt, teacher_force=False) # in test teacher forcing off

			outputs = outputs[:,1:,:]

			for i in range(outputs.shape[0]): # batch size
				pred = outputs[i].argmax(dim=-1)
				total_pred.append(pred[:lengths[idx+i]].detach().cpu().numpy())

			idx += args.batch_size
	
	total_pred = np.concatenate(total_pred)

	return total_pred


# for epoch in range(1, args.n_epochs + 1):
# 	tr_loss, tr_acc = train(model, tr_dataloader, epoch, 'nlp-lab-4/result/model.pt')

# print("\n[ Elapsed Time: {:.4f} ]".format(time.time() - t_start))
# print("Training complete! Best train accuracy: {:.2f}.".format(best_acc*100))


# for kaggle: by using 'pred_{}.npy' to make your submission file
with open('nlp-lab-4/data/de-en/nmt_simple_len.tgt.test.npy', 'rb') as f:
	lengths = np.load(f)

model.load_state_dict(torch.load('nlp-lab-4/result/model.pt', map_location=device))
pred = test(model, ts_dataloader, lengths=lengths)

test_id = ['S'+'{0:05d}'.format(i+1) for i in range(len(pred))]
result_df = pd.DataFrame(
    {'id': test_id,
     'pred': pred
    })
result_df.to_csv("nlp-lab-4/result/result.csv", index=False)