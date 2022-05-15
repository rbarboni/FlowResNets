import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq as tde
from tqdm import tqdm
import time
import argparse
import os
import pickle
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, choices=['RKHS', 'SHL'], default='RKHS')
parser.add_argument('--dataset_size', '-N', type=int, default=100)
parser.add_argument('--data_dim', '-d', type=int, default=2)
parser.add_argument('--num_labels', '-nl', type=int, default=2)
parser.add_argument('--dim', '-q', type=int, default=2)
parser.add_argument('--num_layers', '-l', type=int, default=10)
parser.add_argument('--dim_int', '-r', type=int, default=20)
parser.add_argument('--nu', type=float, default=-1)
parser.add_argument('--method', type=str, choices=['euler', 'midpoint', 'rk4'], default='midpoint')
parser.add_argument('--epochs', '-e', type=int, default=1000)
parser.add_argument('--learning_rate', '-lr', type=float, default=1)
parser.add_argument('--save', '-s', type=str, default='synthetic.pkl')
args = parser.parse_args()

if not os.path.isdir('synthetic_experiments'):
    os.mkdir('synthetic_experiments')
if os.path.isfile('synthetic_experiments/'+args.save):
    sys.exit('A training file already exists: exiting...')

if torch.cuda.is_available():
    print('Using gpu !')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RKHS_ODEfunc(nn.Module):
    def __init__(self, dim, dim_int,
                 num_steps,
                 zero_init=True, nu=-1):
        super(RKHS_ODEfunc, self).__init__()
        self.lin1 = nn.Linear(dim, dim_int, bias=False)
        if nu >= 0:
            Y = torch.randn(dim_int, dim)
            m = torch.distributions.chi2.Chi2(df=2*nu)
            u = m.sample(sample_shape=(dim_int,))
            Omega = Y / torch.sqrt(u / (2*nu))[:, None]
            self.lin1.weight = nn.Parameter(data=(Omega-torch.mean(Omega,axis=0)[None, :]) / dim_int**0.5)
        else:
            self.lin1.weight = nn.Parameter(data=(torch.randn(dim_int, dim) / dim_int**0.5))
        self.lin1.weight.requires_grad = False
        self.num_steps = num_steps
        self.block_list = nn.ModuleList([nn.Linear(dim_int, dim, bias=False) for _ in range(num_steps+1)])
        if zero_init:
            for m in self.block_list:
                m.weight = nn.Parameter(data=torch.zeros_like(m.weight))
            print('ODEfunc initialized at 0')

    def forward(self, t, x):
        k = int(t * self.num_steps - 1)
        out = F.relu(self.lin1(x))
        block1 = self.block_list[k]
        if k < self.num_steps:
            block2 = self.block_list[k + 1]
            out = (1 - t * self.num_steps + k) * block1(out) + (t * self.num_steps - k) * block2(out)
        else:
            out = block1(out)
        return out

class SHL_ODEfunc(nn.Module):
    def __init__(self, dim, dim_int, num_steps, zero_init=True):
        super(SHL_ODEfunc, self).__init__()
        self.num_steps = num_steps
        self.visible_layer_list = nn.ModuleList([nn.Linear(dim, dim_int, bias=False) for _ in range(num_steps+1)])
        self.hidden_layer_list = nn.ModuleList([nn.Linear(dim_int, dim, bias=False) for _ in range(num_steps+1)])
        if zero_init:
            for m in self.visible_layer_list:
                m.weight = nn.Parameter(data=torch.zeros_like(m.weight))
            print('ODEfunc initialized at 0')

    def forward(self, t, x):
        k = int(t * self.num_steps)
        if k < self.num_steps:
            out1 = self.hidden_layer_list[k](x)
            out1 = self.visible_layer_list[k](F.relu(out1))
            out2 = self.hidden_layer_list[k+1](x)
            out2 = self.visible_layer_list[k+1](F.relu(out2))
            out = (1 - t * self.num_steps + k) * out1 + (t * self.num_steps - k) * out2
        else:
            out = self.hidden_layer_list[k](x)
            out = self.visible_layer_list[k](F.relu(out))
        return out

class ODEBlock(nn.Module):
    def __init__(self, odefunc, num_steps, method='euler'):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.method = method
        self.options = {'step_size': 1 / num_steps}
        self.time = torch.tensor([0,1]).float()

    def forward(self, x):
        out = tde.odeint(self.odefunc, x,
                         t=self.time,
                         method=self.method,
                         options=self.options)
        return out[1]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, train_set,
          loss_fn=nn.CrossEntropyLoss(),
          input_embedding=nn.Identity(),
          output_embedding=nn.Identity(),
          epochs=1, lr=0.1):

    model = model.to(device)
    input_embedding = input_embedding.to(device)
    output_embedding = output_embedding.to(device)

    model.train()
    input_embedding.train(mode=False)
    input_embedding.requires_grad_(requires_grad=False)
    output_embedding.train(mode=False)
    output_embedding.requires_grad_(requires_grad=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_loss = torch.empty(0)
    loss_fn = loss_fn.to(device)

    inputs = train_set[0].to(device)
    targets = train_set[1].to(device)

    pbar = tqdm(range(epochs), desc='Loss:', position=0, leave=True)
    for epoch in pbar:
        optimizer.zero_grad()
        outputs = input_embedding(inputs)
        outputs = model(outputs)
        outputs = output_embedding(outputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        if torch.isnan(outputs.detach()).sum() > 0:
            print('Output is NaN: exiting training')
            return train_loss
        train_loss = torch.cat((train_loss, loss.detach().cpu().expand((1,))))

        pbar.set_description(f'Loss: {train_loss[-1]:.3f}', refresh=True)

    return train_loss

# setting experience parameters
N = args.dataset_size
d = args.data_dim
num_layers = args.num_layers
num_labels = args.num_labels
q = args.dim
r = args.dim_int
nu = args.nu

train_set = (torch.randn(N, d), torch.randint(num_labels, (N,)))

# setting input_embedding A
A = torch.zeros((q, d))
for i in range(q // d):
    A[i*d: (i+1)*d, :] = torch.eye(d)
linear_A = nn.Linear(d, q, bias=False)
linear_A.weight = nn.Parameter(data=A)
linear_A.weight.requires_grad = False

# setting output_embedding B
B = torch.zeros((num_labels, q))
B[:, :num_labels] = torch.eye(num_labels)
linear_B = nn.Linear(q, num_labels, bias=False)
linear_B.weight = nn.Parameter(data=B)
linear_B.weight.requires_grad = False

# setting model
if args.model == 'SHL':
    func = SHL_ODEfunc(dim=q,
                       dim_int=r,
                       num_steps=num_layers)
else:
    func = RKHS_ODEfunc(dim=q,
                        dim_int=r,
                        num_steps=num_layers,
                        nu=nu)

model = ODEBlock(func, num_steps=num_layers, method='midpoint')
num_parameters = count_parameters(model)
print(f'Model has {num_parameters} trainable parameters')

start = time.time()
losses = train(model,
               train_set,
               input_embedding=linear_A,
               output_embedding=linear_B,
               epochs=args.epochs,
               lr=args.learning_rate)
computation_time = time.time() - start

training_dict = {
    'model': args.model,
    'num_layers': args.num_layers,
    'N': args.dataset_size,
    'q': args.dim,
    'r': args.dim_int,
    'train_loss': losses.numpy(),
    'computation_time': computation_time,
    'num_parameters': num_parameters
}

# create a binary pickle file
f = open('synthetic_experiments/'+args.save,"wb")
# write the python object (dict) to pickle file
pickle.dump(training_dict,f)
# close file
f.close()


