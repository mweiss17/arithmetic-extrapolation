import time
import numpy as np
import torch
from utils import get_model_path, write_results
from evaluation import get_acc


def train_epoch(dataloader, model, optimizer, scheduler):
    losses = []
    for x, y, x_lens, y_lens in dataloader:
        model.zero_grad()
        y_hat = model(x, x_lens)
        loss = model.compute_loss(y_hat, y, x_lens)
        losses.append(loss.detach().numpy())
        loss.backward()
        optimizer.step()
    if scheduler:
        scheduler.step()
    return model, optimizer, scheduler, np.mean(losses)

def evaluate_model(all_results, train_dataloader, val_dataloader, model, epoch, losses, start):
    model.eval()
    train_result_dict, train_result_str = get_acc(train_dataloader, model)
    val_result_dict, val_result_str = get_acc(val_dataloader, model)
    model.train()

    train_result_dict["loss"] = np.mean(losses)
    train_result_dict["epoch"] = epoch
    all_results.append((train_result_dict, val_result_dict))

    print(f"epoch: {epoch}, loss: {losses[-1]:.3f}, run(sec):{time.time() - start:.1f}, train: {train_result_str}, valid: {val_result_str}")
    return all_results


def train(model, dirpath, optimizer, scheduler, trainloader, valloader, epochs=1500, eval_e=10):
    all_results = []
    losses = []
    max_valid_perf = 0
    max_train_perf = 0
    start = time.time()

    for epoch in range(epochs + 1):
        model, optimizer, scheduler, loss = train_epoch(trainloader, model, optimizer, scheduler)
        losses.append(loss)

        if epoch % eval_e == 0:
            all_results = evaluate_model(all_results, trainloader, valloader, model, epoch, losses, start)
            cur_valid_perf = all_results[-1][1]['wa']
            cur_train_perf = all_results[-1][0]['wa']
            write_results(all_results, dirpath)
            losses = []

            if cur_valid_perf > max_valid_perf:
                max_valid_perf = all_results[-1][1]['wa']
                print(f"New best val perf, ({max_valid_perf}% word accuracy)! Saving model.")
                torch.save(model.state_dict(), get_model_path(dirpath, "valid"))

            if cur_train_perf > max_train_perf:
                max_train_perf = cur_train_perf
                print(f"New best train perf, ({max_train_perf}% word accuracy)! Saving model.")
                torch.save(model.state_dict(), get_model_path(dirpath, "train"))
            if epoch % (10 * eval_e) == 0:
                print(f"It's a special epoch :) Saving model.")
                torch.save(model.state_dict(), get_model_path(dirpath, f"{epoch}"))

    return model, all_results