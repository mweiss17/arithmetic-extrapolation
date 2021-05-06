import pickle
import numpy as np
import argparse
import os
import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from utils import get_plot_path, get_result_pkl_path

parser = argparse.ArgumentParser(description='plot an experiment')
parser.add_argument('--individual_plots', action="store_true")
parser.add_argument('--exp_plots', action="store_true")
parser.add_argument('--all_plots', action="store_true")
args = parser.parse_args()

split_names = ["Train", "Valid"]
metric_names = {"ca": "Character Matches %", "wa": "Word Matches %", "tc": "Total Characters", "tw": "Total Words",
                "cm": "Character Matches", "wm": "Word Matches", "loss": "Loss", "epoch": "Epoch"}


def plot_lines(all_lines, path="", show=False):
    epoch_step = all_lines[0]['epoch'][1]
    xs = np.arange(0, epoch_step * len(all_lines[0]['epoch']), epoch_step)
    for idx, linegroup in enumerate(all_lines):
        for key, line in linegroup.items():
            if key == "epoch":
                continue
            plt.plot(xs, np.array(line))
            plt.title(f"{split_names[idx]}: {metric_names[key]}")
            plt.xlabel("Epochs")
            plt.ylabel(metric_names[key])
            if show:
                plt.show()
            if path:
                plot_path = get_plot_path(path, key)
                plt.savefig(plot_path)
                plt.clf()

def plot_lines_with_std_per_exp(exp_results, path="", show=False):
    for name, exp in tqdm.tqdm(exp_results.items()):
        for split_idx, split in enumerate(exp):
            keys = set(split.keys())
            epoch_step = split['epoch'][0, 1]
            xs = np.arange(0, epoch_step * split['epoch'].shape[1], epoch_step)
            for key in keys:
                mean = split[key].mean(axis=0)
                std = split[key].std(axis=0)
                plt.errorbar(xs, mean, std)
                plt.title(f"{split_names[split_idx]}: {metric_names[key]}")
                plt.xlabel("Epochs")
                plt.ylabel(metric_names[key])
                if show:
                    plt.show()
                if path:
                    plot_path = get_plot_path(path, key)
                    plt.savefig(plot_path)
                plt.clf()
def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plot_exp_by_key_and_splits(key, splits, path):
    metric, split_id = key.split("-")
    split_str = "Train" if int(split_id) == 0 else "Valid"

    epoch_step = splits[0]['epoch'][0][1]
    xs = np.arange(0, epoch_step * splits[0]['epoch'][0].shape[0], epoch_step)
    for split in splits:
        window = 20
        yfit = split[metric].mean(axis=0)
        yfit_smoothed = smooth(yfit, window)
        yfit_smoothed[len(yfit)-window:len(yfit)] = yfit[len(yfit)-window:len(yfit)]
        dyfit = split[metric].std(axis=0)
        dyfit_smoothed = smooth(dyfit, window)
        dyfit_smoothed[len(dyfit)-window:len(dyfit)] = dyfit[len(dyfit)-window:len(dyfit)]
        plt.plot(xs, yfit, '-', label=split["name"])
        plt.fill_between(xs,  np.clip(yfit_smoothed - dyfit_smoothed, 0., 100.),  np.clip(yfit_smoothed + dyfit_smoothed, 0., 100.),  alpha=0.2)
        axes = plt.gca()
        if "ca" in metric:
            axes.set_ylim([50., 100.])
        elif "wa" in metric:
            axes.set_ylim([0., 100.])


    plt.legend()
    plt.title(f"{split_str}: {metric_names[metric]}")
    plt.xlabel("Epochs")
    plt.ylabel(metric_names[metric])
    plot_path = get_plot_path(path, metric + "-" + split_str)

    plt.savefig(plot_path)
    plt.clf()

def plot_all_exps_together(exp_results, path):
    to_plot = defaultdict(list)
    for metric in ["ca", "wa"]:
        for exp_name, exp in tqdm.tqdm(exp_results.items()):
            for split_idx, split in enumerate(exp):
                split["name"] = exp_name
                to_plot[f"{metric}-{split_idx}"].append(split)
    for key, splits in to_plot.items():
        plot_exp_by_key_and_splits(key, splits, path)

def extract_lines(results):
    train_lines = defaultdict(list)
    valid_lines = defaultdict(list)
    keys = set(results[0][0].keys())
    for key in keys:

        for tres, vres in results:
            train_lines[key].append(tres[key])
            if key == "loss" or key == "epoch":
                valid_lines[key].append(tres[key])
            else:
                valid_lines[key].append(vres[key])
    return train_lines, valid_lines

def group_exps_for_plotting(exp_results):
    train_output = defaultdict(list)
    valid_output = defaultdict(list)
    train_results = []
    valid_results = []

    for seed_id in range(len(exp_results)):
        train_results.append(exp_results[seed_id][0])
        valid_results.append(exp_results[seed_id][1])

    for key in set(train_results[0].keys()):
        rs = []
        for result in train_results:
            rs.append(result[key])
        train_output[key] = np.array(rs)

    for key in set(valid_results[0].keys()):
        rs = []
        for result in valid_results:
            rs.append(result[key])
        valid_output[key] = np.array(rs)

    return train_output, valid_output


base_dir = "results"
paths = defaultdict(list)
for exp_dir in os.listdir(base_dir):
    exp_path = os.path.join(base_dir, exp_dir)
    if exp_dir == "plots":
        continue
    for seed_dir in os.listdir(exp_path):
        if seed_dir == "plots":
            continue

        seed_path = os.path.join(exp_path, seed_dir)
        paths[exp_dir].append(get_result_pkl_path(seed_path))

all_results = defaultdict(list)

for exp_dir, result_paths in paths.items():
    results_dir = os.path.join(base_dir, exp_dir)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    exp_results = []
    for result_path in result_paths:
        results = pickle.load(open(result_path, "rb"))
        results = extract_lines(results)

        # plot per seed
        if args.individual_plots:
            individual_plots_path = os.path.join(seed_path, "plots")
            if not os.path.isdir(individual_plots_path):
                os.mkdir(individual_plots_path)
            plot_lines(results, path=individual_plots_path, show=False)

        exp_results.append(results)

    train_output, valid_output = group_exps_for_plotting(exp_results)

    all_results[exp_dir].append(train_output)
    all_results[exp_dir].append(valid_output)

    # plot per experiment
    if args.exp_plots:
        plot_lines_with_std_per_exp(all_results, results_dir)

# one plot for results folder
plot_all_exps_together(all_results, base_dir)