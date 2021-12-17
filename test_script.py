import FlowResNets as frn
import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms, datasets

flatten_transform = transforms.Lambda(lambda x: x.flatten())
transform = transforms.Compose([transforms.ToTensor(), flatten_transform])
train_set = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_set = datasets.MNIST(root="data", train=False, download=True, transform=transform)

## keeping only 0 an 1
idx = (train_set.targets < 2)
train_set.targets = train_set.targets[idx]
train_set.data = train_set.data[idx]

idx = (test_set.targets < 2)
test_set.targets = test_set.targets[idx]
test_set.data = test_set.data[idx]

## set input dimension (= PCA dimension)
d_in = 5

## perform pca
train_set.data = train_set.data.to(torch.float32).reshape((len(train_set), -1))
m = train_set.data.mean(0)
train_set.data = train_set.data - m[None,:]
U,S,V = torch.pca_lowrank(train_set.data, d_in, niter=3)
train_set.data = torch.matmul(train_set.data, V)

## apply pca on test_set
test_set.data = test_set.data.to(torch.float32).reshape((len(test_set), -1))
test_set.data = torch.matmul(test_set.data - m[None,:], V)

## sub-sample dataset
# train_set = torch.utils.data.Subset(train_set, list(range(1000)))

## loading data
batch_size = 200
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

## define losses
loss_fn = frn.quad_loss
loss_fn_test = lambda x,y: frn.quad_loss(frn.threshold(x), y)

## problem constants
d_out = 1 ## output dimension depends on the training instance
q = 2*d_in
q_int = 2*q
Nt = 10 ## number of layers

## define vector field
Omega_sobolev = frn.Sobolev_RFF(q_int, q, 5)
eval = lambda X, Theta: torch.matmul(frn.RFF(X, Omega_sobolev), Theta / q**0.5)
v = frn.vector_field((q_int, q), eval)

## define embedding matrices
A = torch.A = torch.zeros((d_in, q))
for i in range(q//d_in):
    A[:, i*d_in:(i+1)*d_in] = torch.eye(d_in)
A = A.squeeze()

B = torch.zeros((q, d_out))
B[:d_out, :] = torch.eye(d_out)
B = B.squeeze()

## pretrain A and B
#A, B = frn.trained_embedding(q, d_out, train_loader, loss_fn, lr=1)

## define model
model = frn.FlowResNet(Nt, A, B, v, frn.Euler_Integrator)

## optional: modify initialization
#model.Theta = 5*torch.randn(Nt, *v.parameter_shape)/(Nt*q_int*q)**0.5
#model.Theta.requires_grad = True

Nit = 10 ## number of iterations
train_loss, test_loss = frn.train_sgd(model, train_loader, loss_fn, Nit, lr=1,
                                      test_set=test_set, test_loss=loss_fn_test)
plt.plot(train_loss)
#plt.plot(test_loss)



