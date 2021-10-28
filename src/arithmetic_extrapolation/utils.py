import os
import pickle
import numpy as np
from torch.utils.data import DataLoader
from arithmetic_extrapolation.model import LSTM
from arithmetic_extrapolation.dataset import anbn, dyckn, addition

def get_result_pkl_path(dirpath):
    return os.path.join(dirpath, "result.pkl")

def get_params_pkl_path(dirpath):
    return os.path.join(dirpath, "params.pkl")

def get_log_path(dirpath):
    return os.path.join(dirpath, "log.txt")

def get_model_path(dirpath, split="train"):
    return os.path.join(dirpath, f"model-{split}.pt")

def get_plot_path(dirpath, key):
    plot_path = os.path.join(dirpath, "plots")
    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)
    return os.path.join(plot_path, key + ".png")

def get_exp_dir(exp_name, seed, optimizer_type, lr, use_schedule):
    dirpath = f"results/{exp_name}_lr_{lr}_opt_{optimizer_type}_sched_{str(use_schedule).lower()}/seed_{seed}"
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    return dirpath

def write_results(all_results, dirpath):
    # dump results
    pickle.dump(all_results, open(get_result_pkl_path(dirpath), "wb"))

    # write log txt
    with open(get_log_path(dirpath), "w") as file:
        for r in all_results:
            file.write(str(r) + "\n")

def write_params(allcs, allhs, allparams, dirpath):
    pickle.dump({"allcs": np.array(allcs), "allhs": np.array(allhs), "allparams": allparams}, open(get_params_pkl_path(dirpath), "wb"))

def get_dataset(name, batch_size):
    print(name)
    if name == "anbn":
        # create datasets and dataloaders
        train_dataset = anbn(min_n=1, max_n=101)
        val_dataset = anbn(min_n=301, max_n=321)
        test_dataset = anbn(min_n=101, max_n=161)
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=LSTM.pad_collate)
        valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=LSTM.pad_collate)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=LSTM.pad_collate)
    elif name == "dyckn":
        # create datasets and dataloaders
        train_dataset = dyckn(n=2, max_len=100)
        val_dataset = dyckn(n=2, max_len=200)
        test_dataset = dyckn(n=2, max_len=300)
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=LSTM.pad_collate)
        valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=LSTM.pad_collate)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=LSTM.pad_collate)
    elif name == "addition":
        train_dataset = addition(min=0, max=99, sample=.1)
        val_dataset = addition(min=0, max=99, sample=.1)
        test_dataset = addition(min=100, max=199, sample=.1)
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=LSTM.pad_collate)
        valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=LSTM.pad_collate)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=LSTM.pad_collate)

    else:
        raise Exception("NotImplemented")
    return trainloader, valloader, testloader