import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
import main
from hyperparameters import *

#parameters
model_path = "model_folder/LOTM_trained_model"

model = main.BigramLanguageModel()
model.load_state_dict(torch.load(model_path))
model.eval()
m = model.to(device)

context = torch.ones((1, 1), dtype=torch.long, device=device)
print(main.decode(m.generate(context, max_new_tokens=500)[0].tolist()))