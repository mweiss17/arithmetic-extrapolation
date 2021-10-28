import random
import torch
import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from utils import get_exp_dir, get_dataset
from arithmetic_extrapolation.model import LSTM
from arithmetic_extrapolation.train import train

parser = argparse.ArgumentParser(description='train .')
parser.add_argument('--batch_size', type=int, help='batch_size', default=1)
parser.add_argument('--input_size', type=int, help='input_size', default=1)
parser.add_argument('--hidden_size', type=int, help='hidden_size', default=10)
parser.add_argument('--bias', type=bool, help='bias', default=True)
parser.add_argument('--seed', type=int, help='seed', default=1)
parser.add_argument('--epochs', type=int, help='epochs', default=2500)
parser.add_argument('--eval_e', type=int, help='batch_size', default=10)
parser.add_argument('--device', type=str, help='device', default="cpu") #"cuda:0"
parser.add_argument('--lr', type=float, help='lr', default=0.01)
parser.add_argument('--optimizer_type', type=str, help='optimizer type', default="adam")
parser.add_argument('--use_schedule', help='use schedule', action="store_true")
parser.add_argument('--momentum', help='use momentum', action="store_true")
parser.add_argument('--exp_name', type=str, help='experiment name', default="exp")
parser.add_argument('--dataset', type=str, help='dataset name', default="anbn")

args = parser.parse_args()

# set seeds
torch.manual_seed(args.seed)
random.seed(args.seed)
torch.set_num_threads(1)
device = torch.device(args.device)

trainloader, valloader, testloader = get_dataset(args.dataset, args.batch_size)

# initialize model
model = LSTM(args.input_size, args.hidden_size, args.batch_size, len(trainloader.dataset.vocab), device, bias=args.bias)
model = model.to(device)

# initialize optimizer
if args.optimizer_type == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
elif args.optimizer_type == "sgd" and args.momentum:
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=.99)
elif args.optimizer_type == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
else:
    raise Exception("Optimizer Not Defined")

scheduler = None
if args.use_schedule:
    scheduler = MultiStepLR(optimizer, milestones=[50, 150], gamma=0.1)

dirpath = get_exp_dir(args.exp_name, args.seed, args.optimizer_type, args.lr, args.use_schedule)

model, results = train(model, dirpath, optimizer, scheduler, trainloader, valloader, epochs=args.epochs, eval_e=args.eval_e)
