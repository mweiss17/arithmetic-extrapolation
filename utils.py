import os
import pickle

def get_result_pkl_path(dirpath):
    return os.path.join(dirpath, "result.pkl")

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
