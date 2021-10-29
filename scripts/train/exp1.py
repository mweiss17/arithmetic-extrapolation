import os
import time
import wandb
import torch
from torch import optim
import random
import numpy as np
from speedrun import BaseExperiment, WandBMixin, IOMixin
from arithmetic_extrapolation.utils import get_model_path, write_results, get_dataset, get_exp_dir
from arithmetic_extrapolation.model import LSTM, LayerNormLSTM
from arithmetic_extrapolation.evaluation import get_acc

class Trainer(BaseExperiment, WandBMixin, IOMixin):
    WANDB_PROJECT = "arithmetic-extrapolation"
    WANDB_ENTITY = "mweiss10"

    def __init__(self):
        super(Trainer, self).__init__()
        self.auto_setup()
        if self.get("use_wandb"):
            self.initialize_wandb()
        self.all_results = []
        self._build()

    def _build(self):
        # set seeds
        torch.manual_seed(self.get("seed"))
        random.seed(self.get("seed"))
        torch.set_num_threads(1)
        device = torch.device("cpu")

        self.trainloader, self.valloader, self.testloader = get_dataset(self.get("dataset"), self.get("batch_size"))

        # initialize model
        if self.get("use_ln"):
            self.model = LayerNormLSTM(self.get("input_size"), self.get("hidden_size"), self.get("batch_size"), vocab_size=len(self.trainloader.dataset.vocab), bias=self.get("bias"), linear_normal=self.get("linear_normal"))
        else:
            self.model = LSTM(self.get("input_size"), self.get("hidden_size"), self.get("batch_size"),
                              len(self.trainloader.dataset.vocab), device, bias=self.get("bias"), use_embedding=self.get("use_embedding"),
                              set_linear_bias=self.get("set_linear_bias"), linear_normal=self.get("linear_normal"), linear_uniform=self.get("linear_uniform"), use_ln=self.get("use_ln"), ln_preact=self.get("ln_preact"))
        self.model = self.model.to(device)

        # initialize optimizer
        if self.get("optimizer_type") == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.get("lr"), momentum=self.get("momentum", 0.))
        elif self.get("optimizer_type") == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.get("lr"))
        elif self.get("optimizer_type") == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.get("lr"), weight_decay=self.get("weight_decay"))
        else:
            raise Exception("Optimizer Not Defined")

        self.scheduler = None
        if self.get("use_schedule"):
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 150], gamma=0.1)


    def train(self):
        for x, y, x_lens, y_lens in self.trainloader:
            self.model.zero_grad()
            if self.get("use_ln"):
                y_hat = self.model(x.unsqueeze(2).float())
            else:
                y_hat = self.model(x, x_lens)
            loss = self.model.compute_loss(y_hat, y)
            loss.backward()
            self.optimizer.step()
            self.next_step()

        if self.scheduler:
            self.scheduler.step()
        return loss.detach().numpy()

    def validate(self):
        self.model.eval()
        train_result_dict = get_acc(self.trainloader, self.model, use_ln=self.get("use_ln"), split="Train")
        val_result_dict = get_acc(self.valloader, self.model, use_ln=self.get("use_ln"), split="Valid")
        self.model.train()
        return train_result_dict, val_result_dict

    def run(self):
        for epoch in range(self.get("epochs") + 1):
            self.train()

            if self.validate_now:
                # self.wandb_watch(self.model, criterion=self.model.criterion, log="all")
                train_result_dict, val_result_dict = self.validate()
                self.all_results.append((train_result_dict, val_result_dict))
                self.log()

            if self.checkpoint_now:
                self.save_checkpoint()
            self.next_epoch()

    def log(self):
        print(f"epoch: {self.epoch}, results: {self.all_results[-1]}")
        if self.get("use_wandb"):
            self.wandb_log(**self.all_results[-1][0])
        
            self.wandb_log(**self.all_results[-1][1])
        # write_results(self.all_results, os.path.join(self.experiment_directory, "Logs"))

    @property
    def log_now(self):
        return self.step % self.get("log_every") == 0 and self.step > 0

    @property
    def validate_now(self):
        return self.step % self.get("validate_every") == 0 and self.step > 0

    @property
    def checkpoint_now(self):
        return self.step % self.get("checkpoint_every") == 0 and self.step > 0

    def save_checkpoint(self):
        data = {"model": self.model.state_dict(), "optim": self.optimizer.state_dict()}

        checkpoint_path = f"{self.experiment_directory}/Weights/model-{self.step}.pt"
        print(f"checkpointing the model to {checkpoint_path}")
        torch.save(data, checkpoint_path)

    def load_checkpoint(self):
        checkpoint_data = torch.load(self.get("checkpoint_path"))
        self.model.load_state_dict(checkpoint_data["model"])
        self.optimizer.load_state_dict(checkpoint_data["optim"])

if __name__ == '__main__':
    Trainer().run()
