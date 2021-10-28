import pickle
import numpy as np
import argparse
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict

parser = argparse.ArgumentParser(description='plot an experiment')
args = parser.parse_args()


base_dir = "results"
exp_dir = "test_lr_0.01_opt_adam_sched_false"
seed_dir = "seed_1"

path = os.path.join(base_dir, exp_dir, seed_dir)
model_path = os.path.join(path, "model-train.pt")
param_path = os.path.join(path, "params.pkl")
plot_dir = os.path.join(path, "plots")

if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)


# create datasets and dataloaders
batch_size = 1
hidden_size = 1
input_size = 1
params = pickle.load(open(param_path, "rb"))
cs = params['allcs']
hs = params['allhs']
allparams = params['allparams']

result = defaultdict(list)
for data in allparams:
    for key, val in data.items():
        result[key].append(val)
ps = {}
for k, v in result.items():
    ps[k] = np.array(result[k]).squeeze()

def make_plot(data, title, plot_dir, fname):
    n_epochs = 20
    legend_skip_n = 1
    lines = data[::n_epochs] # num steps x every n_epochs
    evenly_spaced_interval = np.linspace(0, 1, len(lines))
    colors = [cm.viridis(x) for x in evenly_spaced_interval]
    for i, color in enumerate(colors):
        if i % legend_skip_n == 0:
            plt.plot(lines[i], color=color, label=str(i*n_epochs) + " epochs")
        else:
            plt.plot(lines[i], color=color)
    plt.legend()
    plt.title(title)
    plt.xlabel("Position in Sequence")
    plt.ylabel("Activation value")
    plt.savefig(os.path.join(plot_dir, fname))
    plt.clf()

make_plot(cs, "Cell State activations" , plot_dir, f"_cs.png",)
make_plot(hs, "Hidden State activations", plot_dir, f"_hs.png",)

for k, v in ps.items():

    plt.plot(v)
    plt.title(f"{k}")
    plt.savefig(os.path.join(plot_dir, f"{k}.png"))
    plt.clf()

# at the last run, what are the results?

t = hs.shape[1]-1
hidden = hs[-1]
l_t = ps['linear_weights'][-1]
l_b = ps['linear_bias'][-1]
oih = ps['output_gate_ih'][-1]
ohh = ps['output_gate_hh'][-1]
boih = ps['bias_output_gate_ih'][-1]
bohh = ps['bias_output_gate_hh'][-1]
all_logits = []
preds = []
for i in range(t):
    out = (hidden[i] * l_t) + l_b
    logits = torch.nn.Softmax()(torch.FloatTensor(out))
    all_logits.append(logits.detach().numpy())
    pred = logits.argmax().data.item()
    preds.append(pred)

name_map = {0: "a", 1: "b", 2: "EOS", 3: "PAD"}
all_logits = np.array(all_logits)

for i in range(all_logits.shape[1]):
    plt.plot(all_logits[:, i], label=name_map[i])
    plt.legend()
plt.savefig(os.path.join(plot_dir, "logits.png"))