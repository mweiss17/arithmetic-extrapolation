import pickle
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from arithmetic_extrapolation.model import LSTM
from arithmetic_extrapolation.dataset import anbn, anbnEval

parser = argparse.ArgumentParser(description='plot an experiment')
parser.add_argument('--individual_plots', action="store_true")
parser.add_argument('--exp_plots', action="store_true")
parser.add_argument('--all_plots', action="store_true")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--device', type=str, default="cpu")
args = parser.parse_args()


base_dir = "results"
exp_dir = "b_lr_0.01_opt_adam_sched_false"
seed_dir = "seed_1"

path = os.path.join(base_dir, exp_dir, seed_dir)
model_path = os.path.join(path, "model-train.pt")
plot_dir = os.path.join(path, "plots")

if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)

# set seeds
torch.manual_seed(args.seed)
random.seed(args.seed)
torch.set_num_threads(1)
device = torch.device(args.device)

# create datasets and dataloaders
batch_size = 1
hidden_size = 10
input_size = 1
vocab_size = 4
hs = []
cs = []

model = LSTM(input_size, hidden_size, 1, vocab_size, args.device, bias=False)
model.load_state_dict(torch.load(model_path))

def save_activation(module, input, out):
    h, c = out[1]
    h = h.detach().numpy()
    c = c.detach().numpy()
    hs.append(h)
    cs.append(c)

for name, m in model.lstm.named_modules():
    if type(m) == nn.LSTM:
        m.register_forward_hook(save_activation)


results = {}
min_n = 1
max_n = 1000
curval = 0

first_idx_fail = 0
n_neq_m_successes = []
n=200

word = "a" * n + "b" * n
dataloader = DataLoader(anbnEval(word=word), batch_size=1, shuffle=True, collate_fn=pad_collate)
x, y, x_lens, y_lens = next(iter(dataloader))
preds = model(x, x_lens, record_activations=True)

hs = np.array(hs)
hs = hs.squeeze(1).squeeze(1)
cs = np.array(cs)
cs = cs.squeeze(1).squeeze(1)
import pdb; pdb.set_trace()
plt.plot(hs)
plt.title(f"Hidden state")
plt.savefig(os.path.join(plot_dir, f"quantize_hs_{min_n}.png"))
plt.clf()

plt.plot(cs)
plt.title(f"Cell state ")
plt.savefig(os.path.join(plot_dir, f"quantize_cs_{min_n}.png"))
plt.clf()

hs = []
cs = []
preds, hiddens, cells = model.forward_save_hiddens(x, x_lens)
preds = preds.squeeze(0).argmax(axis=1)
y = y.squeeze(0).argmax(1)
word_match = bool(torch.all(y == preds))
print(word_match)
import pdb; pdb.set_trace()