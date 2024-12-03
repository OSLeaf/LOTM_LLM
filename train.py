import main
import torch
from hyperparameters import *
import tiktoken

torch.manual_seed(1337)

with open('LotM.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#Tokenizer:
def encode(l):
    return enc.encode(l)
def decode(l):
    return enc.decode(l)
print(vocab_size)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))  #Better split at some point
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = main.BigramLanguageModel()
model.load_state_dict(torch.load("TokModel_4500"))
model.eval()
m = model.to(device)

#optimizer:
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

val_loss_stopper = 20
loss_counter = 0

for iter in range(max_iters):

    if iter % eval_interval == 0:
        torch.save(model.state_dict(), f"model_folder/TokModel_{iter}")
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if val_loss_stopper < losses['val'] and loss_counter == 2:
            quit()
        elif val_loss_stopper < losses['val']:
            loss_counter += 1
        else:
            val_loss_stopper = losses['val']
    

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(f"iteration: {iter}")
