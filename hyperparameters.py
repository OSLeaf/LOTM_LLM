import torch
import tiktoken

# hyperparameters:
batch_size = 64
block_size = 256
max_iters = 10000
eval_interval = 500
learning_rate = 2e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
eval_iters = 100
n_embd = 512
n_head = 8
n_layer = 8
dropout = 0.2