import torch
from torchvision import transforms, datasets
import RKHS_NODE as node
import time
import argparse
import os
import sys
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, choices=['RKHS', 'SHL'], default='RKHS')
parser.add_argument('--layers', '-l', type=int, default=10)
parser.add_argument('--epochs', '-e', type=int, default=200)
parser.add_argument('--lr_init', '-lr', type=float, default=5e-3)
parser.add_argument('--decay_rate', '-dr', type=float, default=0.1)
parser.add_argument('--decay_steps', '-ds', nargs='*', type=int, default=[150, 190])
parser.add_argument('--batch_size', '-bs', type=int, default=256)
parser.add_argument('--test_batch_size', '-tbs', type=int, default=1000)
parser.add_argument('--save', '-s', type=str, default='experiment.pkl')
args = parser.parse_args()

if not os.path.isdir('CIFAR_experiments'):
    os.mkdir('CIFAR_experiments')
if os.path.isfile('CIFAR_experiments/'+args.save):
    sys.exit('This training file already exists: exiting script ...')

if torch.cuda.is_available():
    print('Using gpus')
else:
    print('WARNING: using cpu, computation may be slow !')

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_set = datasets.CIFAR10(root='CIFAR',
                              train=True,
                              download=True,
                              transform=transform_train)
test_set = datasets.CIFAR10(root='CIFAR',
                            train=False,
                            download=True,
                            transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=4)
train_eval_loader = torch.utils.data.DataLoader(train_set,
                                                batch_size=args.test_batch_size,
                                                shuffle=True,
                                                num_workers=4)
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=args.test_batch_size,
                                          shuffle=False,
                                          num_workers=4)

# Model
print('==> Building model ...')
if args.model=='RKHS':
    print('ResNet with RKHS residuals')
else:
    print('ResNet with SHL residuals')
model = node.ResNet(node.RKHS_Block if args.model=='RKHS' else node.SHL_Block,
                    [2, 2, args.layers, 2],
                    num_classes=len(train_set.classes)
                    )
model = model.cuda()

# Training
start_time = time.time()
losses = node.train_sgd(model,
                        train_loader,
                        train_eval_loader,
                        test_loader,
                        epochs=args.epochs,
                        lr_init=args.lr_init,
                        decay_steps=args.decay_steps,
                        decay_rate=args.decay_rate,
                        ckpt_dir='CIFAR_checkpoint')
computation_time = time.time() - start_time

training_dict = {
    'model': args.model,
    'num_layers': args.layers,
    'num_parameters': node.count_parameters(model),
    'batch_size': args.batch_size,
    'train_loss': losses[0].numpy(),
    'train_classif': losses[1].numpy(),
    'test_classif': losses[2].numpy(),
    'computation_time': computation_time
}

# create a binary pickle file
f = open('CIFAR_experiments/'+args.save,"wb")
# write the python object (dict) to pickle file
pickle.dump(training_dict,f)
# close file
f.close()

print('Training data saved at: CIFAR_experiments/'+args.save)
