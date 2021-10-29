import pickle
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import LSTM
from dataset import anbn, anbnEval

parser = argparse.ArgumentParser(description='plot an experiment')
parser.add_argument('--individual_plots', action="store_true")
parser.add_argument('--exp_plots', action="store_true")
parser.add_argument('--all_plots', action="store_true")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--device', type=str, default="cpu")
args = parser.parse_args()


base_dir = "results"
exp_dir = "b_lr_0.01_opt_adam_sched_false" # used in paper
# exp_dir = "c_lr_0.001_opt_adam_sched_false" #
seed_dir = "seed_1"

path = os.path.join(base_dir, exp_dir, seed_dir)
model_path = os.path.join(path, "model-valid.pt")
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

model = LSTM(input_size, hidden_size, 1, vocab_size, args.device)
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

def run_model(model, dataloader):
    x, y, x_lens, y_lens = next(iter(dataloader))
    preds = model(x, x_lens)
    preds = preds.squeeze(0).argmax(axis=1)
    y = y.squeeze(0).argmax(1)
    word_match = bool(torch.all(y == preds))
    return word_match

results = {}
min_n = 1
max_n = 1000
curval = 0

first_idx_fail = 0
n_neq_m_successes = []
for n in range(3, max_n):
    for m in range(curval, 100):
        word = "a" * n + "b" * int(n-m) # sometimes int(n+m)
        dataloader = DataLoader(anbnEval(word=word), batch_size=1, shuffle=True, collate_fn=pad_collate)
        word_match = run_model(model, dataloader)

        if m == 0:
            print(f"n: {n}, m: {n + m}, match: {word_match}")
        if m == 0 and not word_match and not first_idx_fail:
            first_idx_fail = n
            print(f"first index of failure to match: {first_idx_fail}")
            break
        if m == 0 and word_match:
            break

        if m != 0 and word_match:
            n_neq_m_successes.append((n, m))
            curval = m
            print(f"n: {n}, m: {n + m}, match: {word_match}")
            break
        # if m != 0 and not word_match:
        #     print(f"not word match {m} , {n}")
print(n_neq_m_successes)
drift = np.zeros(max_n)
for n, n_minus_m in n_neq_m_successes:
    drift[n] = n_minus_m

plt.plot(np.arange(len(drift)), drift)
plt.title(f"Drift amount (a^nb^m)")
plt.ylabel("n-m (amount of drift)")
plt.xlabel("n (number of a's)")
plt.savefig(os.path.join(plot_dir, f"drift.png"))
plt.clf()
