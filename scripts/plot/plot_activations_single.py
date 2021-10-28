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
from arithmetic_extrapolation.dataset import anbn

parser = argparse.ArgumentParser(description='plot an experiment')
parser.add_argument('--individual_plots', action="store_true")
parser.add_argument('--exp_plots', action="store_true")
parser.add_argument('--all_plots', action="store_true")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--device', type=str, default="cpu")
args = parser.parse_args()


base_dir = "experiments"
exp_dir = "exp1"

path = os.path.join(base_dir, exp_dir)
model_path = os.path.join(path, "Weights", "model-23000.pt")
plot_dir = os.path.join(path, "Logs")

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

min_n = 40000
test_dataset = anbn(min_n=min_n, max_n=min_n+1)
dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=LSTM.pad_collate)

model = LSTM(input_size, hidden_size, 1, len(anbn.vocab), args.device)
model.load_state_dict(torch.load(model_path)["model"])

all_hs = []
all_cs = []


for x, y, x_lens, y_lens in dataloader:
    preds, cs, hs = model.forward_save(x)
    all_cs.append(cs)
    all_hs.append(hs)

hs = np.array(all_hs)
hs = hs.squeeze(0)
cs = np.array(all_cs)
cs = cs.squeeze(0)

plt.plot(hs)
plt.title(f"Hidden state")
plt.savefig(os.path.join(plot_dir, f"hs_{min_n}.png"))
plt.clf()

plt.plot(cs)
plt.title(f"Cell state ")
plt.savefig(os.path.join(plot_dir, f"cs_{min_n}.png"))
plt.clf()

for key, weights in model.lstm._parameters.items():
    if "bias" in key:
        continue
    plt.matshow(weights.detach().numpy(), cmap='viridis')
    plt.title(f"{key}, min: {weights.min().item():.1f}, max: {weights.max().item():.1f}")
    plt.savefig(os.path.join(plot_dir, f"weights_{key}"))
    plt.clf()
