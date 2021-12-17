import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


### 2 numerical ODE solver adapted to the FlowResNet model

## 4th order Runge-Kutta solver
def RK4_Integrator(Theta, v, save_traj=False):
    Nt = Theta.shape[0]
    dt = 1 / (Nt - 1)

    def f(x0):
        x = x0.clone()
        if save_traj:
            traj = x.clone().detach().expand(1, *x.shape)
        for k in range(Nt - 1):
            Theta_int = 0.5 * (Theta[k] + Theta[k + 1])
            k1 = v(x, Theta[k])
            k2 = v(x + 0.5 * dt * k1, Theta_int)
            k3 = v(x + 0.5 * dt * k2, Theta_int)
            k4 = v(x + dt * k3, Theta[k + 1])
            x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            if save_traj:
                traj = torch.cat((traj, x.clone().detach().expand(1, *x.shape)))
        if save_traj:
            return x, traj
        else:
            return x
    return f

## Explicit Euler integrator
def Euler_Integrator(Theta, v, save_traj=False):
    Nt = Theta.shape[0]
    dt = 1 / (Nt - 1)

    def f(x0):
        x = x0.clone()
        if save_traj:
            traj = x.clone().detach().expand(1, *x.shape)
        for k in range(Nt - 1):
            x = x + dt * v(x, Theta[k])
            if save_traj:
                traj = torch.cat((traj, x.clone().detach().expand(1, *x.shape)))
        if save_traj:
            return x, traj
        else:
            return x
    return f


## non-linearities
def ReLU(x):
    return x * (x > 0)

def RFF(X, Omega):
    return torch.cos(torch.matmul(X, Omega.T))

def Sobolev_RFF(q_int, q, s=5): ## sampling q-variate t-distribution with shape parameter s
    Y = torch.randn(q_int, q)
    m = torch.distributions.chi2.Chi2(df=s)
    u = m.sample(sample_shape=(q_int,))
    Omega = Y / torch.sqrt(u / s)[:, None]
    return Omega - torch.mean(Omega, axis=0)[None, :]

def Gaussian_RFF(q_int, q): ## sampling q-variate normal distribution
    return torch.randn(q_int, q)

def Polynomial(x, d):
    res = torch.ones(x.shape)
    for k in range(1, d):
        res = torch.cat((res, x ** k), 1)
    return res

## losses
def quad_loss(x, y):
    return 0.5 * ((x - y) ** 2).sum() / y.shape[0]

def threshold(x): ## return x with entries thresholded between 0 and 1
    res = nn.functional.threshold(x, 0, 0)
    res = - nn.functional.threshold(-res, -1, -1)
    return res

def distmat(x, y):
    return ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)

def W2_loss(x, y):
    N, M = x.shape[0], y.shape[0]
    C = distmat(x, y)
    Cnp = C.clone().detach().numpy()

    # linear programming
    P = cp.Variable((N, M))
    u = np.ones((M, 1))
    v = np.ones((N, 1))
    x_weights, y_weights = np.ones((N, 1)) / N, np.ones((M, 1)) / M
    U = [0 <= P, cp.matmul(P, u) == x_weights, cp.matmul(P.T, v) == y_weights]
    objective = cp.Minimize(cp.sum(cp.multiply(P, Cnp)))
    prob = cp.Problem(objective, U)
    prob.solve()

    return (C * torch.tensor(P.value)).sum()

def log_soft_min(H, epsilon, weights):
    Hmin = torch.min(H, dim=0)[0]
    return -epsilon * torch.log(torch.sum(weights[:, None] * torch.exp(-(H - Hmin[None, :]) / epsilon), axis=0)) + Hmin

def Sinkhorn_loss(x, y, eps=0.1, it=20):
    N, M = x.shape[0], y.shape[0]
    C = distmat(x, y)
    Cclone = C.clone().detach()
    x_weights, y_weights = torch.ones(N) / N, torch.ones(M) / M

    # Sinkhorn iterations
    f = torch.zeros(N)
    for _ in range(it):
        g = log_soft_min(Cclone - f[:, None], eps, x_weights)
        f = log_soft_min(Cclone.t() - g[:, None], eps, y_weights).t()
        P = x_weights[:, None] * torch.exp((f[:, None] + g[None, :] - Cclone) / eps) * y_weights[None, :]

    return (C * P).sum()

## FlowResNet objects
class FlowResNet():
    def __init__(self, Nt, A, B, v, Integrator, save_traj=False):
        super(FlowResNet, self).__init__()
        self.Nt = Nt ## number of times steps (i.e. layers)
        self.A = A
        self.B = B
        self.v = v
        self.Integrator = Integrator
        self.save_traj = save_traj
        self.Theta = torch.zeros((Nt, *self.v.parameter_shape), requires_grad=True) ## access to parameter_shape associated to v is needed

    ## forward method of the model
    def forward(self, input):
        f = self.Integrator(self.Theta, self.v.eval, save_traj=self.save_traj)
        if self.save_traj:
            z, traj = f(torch.matmul(input, self.A))
            return torch.matmul(z, self.B.squeeze()), torch.matmul(traj, self.B.squeeze())
        else:
            z = f(torch.matmul(input, self.A))
            return torch.matmul(z, self.B.squeeze())

## vector_filed objects
## they carry the parameter's shape information and are provided with an evaluation method
class vector_field():
    def __init__(self, parameter_shape, v):
        super(vector_field, self).__init__()
        self.parameter_shape = parameter_shape ## access to parameter_shape will be needed in FlowResNet objects
        self.eval = v

## training routine
## performs sgd on model with data stored in train_loader, optional test arguments
def train_sgd(model, train_loader, loss_fn, Nit, lr, test_set=None, loss_fn_test=None):
    optimizer = torch.optim.SGD([model.Theta], lr=lr) ## fixed learning rate, maybe change that

    train_loss = torch.empty(0) ## stores training loss
    if test_set is not None: ## if test arguments are passed
        test_loss = torch.empty(0) ## stores test loss
        if loss_fn_test is None:
            loss_fn_test = loss_fn

    model.save_traj = False ## model.save_traj must be disabled

    ## training loop
    for i in tqdm(range(Nit)):
        for inputs, labels in tqdm(train_loader, desc=f'{i}th pass', position=0, leave=True):

            ## compute and stores test loss
            if test_set is not None:
                outputs = model.forward(test_set.data)
                loss = loss_fn_test(outputs, test_set.targets)
                test_loss = torch.cat((test_loss, loss.detach().expand((1,))))

            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            if torch.isnan(outputs.detach()).sum() > 0: ## stops if degennerated gradient
                break
            train_loss = torch.cat((train_loss, loss.detach().expand((1,)))) ## computes and stores train loss

    if test_set is not None:
        return train_loss, test_loss
    else:
        return train_loss

## routine for pretraining embedding matrices
def trained_embedding(q, d_out, train_loader, loss_fn, Nit=10, lr=1):
    if train_loader.dataset.data.dim() > 1:
        d_in = train_loader.dataset.data.shape[1]
    else:
        d_in = 1

    A = torch.randn(d_in, q).squeeze() / q**0.5
    B = torch.randn(q, d_out).squeeze() / q**0.5
    A.requires_grad = True
    B.requires_grad = True

    optimizer = torch.optim.SGD([A, B], lr=lr)
    for i in tqdm(range(Nit), desc='pretraining embedding'):
        for inputs, labels in tqdm(train_loader, position=0, leave=True):
            optimizer.zero_grad()
            outputs = torch.matmul(torch.matmul(inputs, A), B)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

    return A.detach(), B.detach()