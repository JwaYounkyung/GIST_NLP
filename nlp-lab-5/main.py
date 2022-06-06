import os
import pdb
import csv
import random
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import time, datetime
import argparse
import numpy as np
from pathlib import Path
import pandas as pd

import utils, dataloader
from model.transformer import Transformer
from torch.optim import SGD, Adam
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


def train(dataloader, epochs, model, criterion, optimizer, args, model_root):

	"""
	Todo: Set your optimizer
	"""

	model.train()
	tr_loss = 0.
	correct = 0

	cnt = 0
	best_acc = None
	for epoch in range(epochs):

		for idx, (src, tgt) in enumerate(dataloader):
			"""
			ToDo: feed the input to model
			src.size() : [batch, max length]
			tgt.size() : [batch, max lenght + 1], the first token is always [SOS] token.
			
			These codes are one of the example to train model, so changing codes are acceptable.
			But when you change it, please left comments.
			If model is run by your code and works well, you will get points.
			"""
			src, tgt = src.to(device), tgt.to(device)

			optimizer.zero_grad()
			outputs = model(src, tgt[:,:-1], train=True)

			outputs = outputs.reshape(src.shape[0] * args.max_len, -1)
			tgt = tgt[:,1:].reshape(-1)

			loss = criterion(outputs, tgt)
			tr_loss += loss.item()
			loss.backward()

			torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
			optimizer.step()

			# accuracy
			pred =  outputs.argmax(dim=1, keepdim=True)
			pred_acc = pred[tgt != 2] # pad가 아닌것만 정확도 추출
			tgt_acc = tgt[tgt != 2]
			correct += pred_acc.eq(tgt_acc.view_as(pred_acc)).sum().item()

			cnt += tgt_acc.shape[0]

		tr_loss /= cnt
		tr_acc = correct / cnt

		print("[epoch {:3d}/{:3d}] loss: {:.6f} acc: {:.4f})".format(
			epoch+1, args.n_epochs, tr_loss, tr_acc*100), end='\n')

		if best_acc is None or tr_acc >= best_acc:
			torch.save(model.state_dict(), model_root)
			best_acc = tr_acc 
	
	print("Training complete! Best train accuracy: {:.2f}.".format(best_acc*100))

	return tr_loss, tr_acc

def test(dataloader, model, args, lengths=None):
	model.eval()
	idx = 0
	total_pred = []

	with torch.no_grad():
		correct = 0.
		cnt = 0.
		for src, tgt in dataloader:
			src, tgt = src.to(device), tgt.to(device)

			outputs = model(src, tgt[:,:-1], train=False) # in test teacher forcing off

			for i in range(outputs.shape[0]):
				"""
				ToDo: Output (total_pred) is the model predict of test dataset
				Variable lenghts is the length information of the target length.
				"""
				pred = outputs[i].argmax(dim=-1)
				total_pred.append(pred[:lengths[idx+i]].detach().cpu().numpy())
			
			idx += args.batch_size

	total_pred = np.concatenate(total_pred)

	return total_pred

def main():
	parser = argparse.ArgumentParser(description='NMT - Transformer')
	""" recommend to use default settings """

	# environmental settings
	parser.add_argument('--gpu_id', type=int, default=0)
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--save', action='store_true', default=0)

	# architecture
	parser.add_argument('--num_enc_layers', type=int, default=6, help='Number of Encoder layers')
	parser.add_argument('--num_dec_layers', type=int, default=6, help='Number of Decoder layers')
	parser.add_argument('--num_token', type=int, help='Number of Tokens')
	parser.add_argument('--max_len', type=int, default=20)
	parser.add_argument('--model_dim', type=int, default=512, help='Dimension size of model dimension')
	parser.add_argument('--hidden_size', type=int, default=2048, help='Dimension size of hidden states')
	parser.add_argument('--d_k', type=int, default=64, help='Dimension size of Key and Query')
	parser.add_argument('--d_v', type=int, default=64, help='Dimension size of Value')
	parser.add_argument('--n_head', type=int, default=8, help='Number of multi-head Attention')
	parser.add_argument('--d_prob', type=float, default=0.1, help='Dropout probability')
	parser.add_argument('--max_norm', type=float, default=5.0)

	# hyper-parameters
	parser.add_argument('--n_epochs', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--lr', type=float, default=0.0005)
	parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 hyper-parameter for Adam optimizer')
	parser.add_argument('--beta2', type=float, default=0.98, help='Beta2 hyper-parameter for Adam optimizer')
	parser.add_argument('--eps', type=float, default=1e-9, help='Epsilon hyper-parameter for Adam optimizer')
	parser.add_argument('--weight_decay', type=float, default=1e-5)
	parser.add_argument('--teacher-forcing', action='store_true', default=False)
	parser.add_argument('--warmup_steps', type=int, default=78, help='Warmup step for scheduler')
	parser.add_argument('--logging_steps', type=int, default=500, help='Logging step for tensorboard')
	# etc
	parser.add_argument('--k', type=int, default=4, help='hyper-paramter for BLEU score')

	args = parser.parse_args()

	utils.set_random_seed(args)
	t_start = time.time()

	tr_dataset = dataloader.NMTSimpleDataset(max_len=args.max_len,
											 src_filepath='nlp-lab-5/data/de-en/nmt_simple.src.train.txt',
											 tgt_filepath='nlp-lab-5/data/de-en/nmt_simple.tgt.train.txt',
											 vocab=(None, None),
											 is_src=True, is_tgt=False, is_train=True)
	ts_dataset = dataloader.NMTSimpleDataset(max_len=args.max_len,
											 src_filepath='nlp-lab-5//data/de-en/nmt_simple.src.test.txt',
											 tgt_filepath=None,
											 vocab=(tr_dataset.vocab, None),
											 is_src=True, is_tgt=False, is_train=False)


	vocab = tr_dataset.vocab
	i2w = {v:k for k, v in vocab.items()}
	num_token = len(vocab)
	pad_idx = vocab['[PAD]']

	tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
	ts_dataloader = DataLoader(ts_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=2)

	model = Transformer(num_token=num_token,
						pad_idx=pad_idx,
						max_seq_len=args.max_len,
						dim_model=args.model_dim,
						n_head=args.n_head,
						dim_hidden=args.hidden_size,
						d_prob=args.d_prob,
						n_enc_layer=args.num_enc_layers,
						n_dec_layer=args.num_dec_layers)
	model.init_weights()
	model = model.to(device)

	criterion = nn.NLLLoss(ignore_index=pad_idx) 
	optimizer = optim.Adam(model.parameters(), lr=args.lr) # weight decay

	### Train
	tr_loss, tr_acc = train(tr_dataloader, args.n_epochs, model, criterion, optimizer, args, 'nlp-lab-5/result/model.pt')
	
	print("\n[ Elapsed Time: {:.4f} ]".format(time.time() - t_start))

	### Test
	with open('nlp-lab-5/data/de-en/length.npy', 'rb') as f:
		lengths = np.load(f)

	model.load_state_dict(torch.load('nlp-lab-5/result/model.pt', map_location=device))
	pred = test(ts_dataloader, model=model, args=args, lengths=lengths)

	test_id = ['S'+'{0:05d}'.format(i) for i in range(len(pred))]
	result_df = pd.DataFrame(
		{'ID': test_id,
		'label': pred
		})
	result_df.to_csv("nlp-lab-5/result/result.csv", index=False)

if __name__ == "__main__":
	main()
















