import torch
import torch.nn as nn
from torchvision import transforms, datasets
import RKHS_NODE as node
import time
import argparse
import os
import pickle
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, choices=['RKHS', 'SHL'], default='RKHS')
parser.add_argument('--in_channels', '-in', type=int, default=32)
parser.add_argument('--layers', '-l', type=int, default=20)
parser.add_argument('--dim_int', type=int, default=32)
parser.add_argument('--method', type=str, choices=['euler', 'midpoint', 'rk4'], default='midpoint')
parser.add_argument('--epochs', '-e', type=int, default=30)
parser.add_argument('--lr_init', '-lr', type=float, default=1)
parser.add_argument('--decay_rate', '-dr', type=float, default=0.1)
parser.add_argument('--decay_steps', '-ds', nargs='*', type=int, default=[25])
parser.add_argument('--batch_size', '-bs', type=int, default=256)
parser.add_argument('--test_batch_size', '-tbs', type=int, default=1000)
parser.add_argument('--save', '-s', type=str, default='experiment.pkl')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

if not os.path.isdir('MNIST_experiments'):
    os.mkdir('MNIST_experiments')
if os.path.isfile('MNIST_experiments/'+args.save):
    sys.exit('A training file already exists: exiting...')

if torch.cuda.is_available():
    print('Using gpu !')
else:
    print('WARNING: using cpu, computation may be slow !')

transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.MNIST(root="MNIST",
                           train=True,
                           download=True,
                           transform=transform)
test_set = datasets.MNIST(root="MNIST",
                          train=False,
                          download=True,
                          transform=transform)

# Data
print('==> Preparing data..')
num_workers = 2 if torch.cuda.is_available() else 16
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=num_workers)
train_eval_loader = torch.utils.data.DataLoader(train_set,
                                                batch_size=args.test_batch_size,
                                                shuffle=False,
                                                num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=args.test_batch_size,
                                          shuffle=False,
                                          num_workers=num_workers)
print('==> Building model ...')
if args.model == 'RKHS':
    print('ResNet with RKHS residuals')
    func = node.RKHS_ODEfunc(dim=args.in_channels,
                             dim_int=args.dim_int,
                             num_steps=args.layers)
else:
    print('ResNet with SHL residuals')
    func = node.SHL_ODEfunc(dim=args.in_channels,
                            dim_int=args.dim_int,
                            num_steps=args.layers)

downsampling_layers = [nn.Conv2d(1, args.in_channels, 4, stride=2, padding=0),
                       nn.BatchNorm2d(args.in_channels),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(args.in_channels, args.in_channels, 4, stride=2, padding=0),
                       nn.BatchNorm2d(args.in_channels)]

fc_layers = [nn.BatchNorm2d(args.in_channels),
             nn.Flatten(),
             nn.Linear(args.in_channels*25, 10)]

model = nn.Sequential(*downsampling_layers,
                      node.ODEBlock(func, num_steps=args.layers, method=args.method),
                      *fc_layers)

num_parameters = node.count_parameters(model)
print(f'Model has {num_parameters} trainable parameters')

print('==> Training...')
start_time = time.time()

losses = node.train_sgd(model,
                        train_loader,
                        train_eval_loader,
                        test_loader,
                        epochs=args.epochs,
                        lr_init=args.lr_init,
                        decay_steps=args.decay_steps,
                        decay_rate=args.decay_rate)

computation_time = time.time() - start_time


training_dict = {
    'model': args.model,
    'num_layers': args.layers,
    'in_channels': args.in_channels,
    'dim_int': args.dim_int,
    'batch_size': args.batch_size,
    'train_loss': losses[0].numpy(),
    'train_classif': losses[1].numpy(),
    'test_classif': losses[2].numpy(),
    'computation_time': computation_time,
    'num_parameters': num_parameters
}

# create a binary pickle file
f = open('MNIST_experiments/'+args.save,"wb")
# write the python object (dict) to pickle file
pickle.dump(training_dict,f)
# close file
f.close()

print('Training data saved at: MNIST_experiments/'+args.save)