# Global Convergence of ResNets: From finite to infinite width using linear parameterization

This file contains the source code for the RKHS-NODE ResNet model.

## Dependencies

You first need to install dependencies

```
pip install torch torchdiffeq tqdm
```

In particular, the code uses the <code> torchdiffeq </code> package (see documentation here: [GitHub - rtqichen/torchdiffeq: Differentiable ODE solvers with full GPU support and O(1)-memory backpropagation.](https://github.com/rtqichen/torchdiffeq) ).

## MNIST_experiments.py

The python script <code> MNIST_experiments.py </code> was used to perform experiments on the MNIST dataset in `Figure 1` and `Figure 2`.  The script can be run using:

```
python MNIST_experiments.py
```

At the end of training, the architecture achieving the best accuracy on the test set is stored in `MNIST_checkpoint/ckpt.pth`.

The implementation is inpired from the implementation of Neural ODE for the MNIST dataset: [torchdiffeq/odenet_mnist.py at master · rtqichen/torchdiffeq · GitHub](https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py) 

#### Parameters

Several parameters can be parsed into the command line:

- `--model`: set the kind of ResNet used between `'RKHS'` and `'SHL'`. `'RKHS'` corresponds to residuals with fixed hiddenlayers and `'SHL'` to residuals with trained hidden layers (`default='RKHS'`).

- `--in_channels`:  number of channels at input of the ResNet block (`default=32`).

- `--dim_int`: number of channels at hidden layers of the residuals (`default=32`).

- `--layers`: number of layers of the ResNet, corresponding to the number of integration steps in the numerical integration of Neural ODE (`default=20`).

- `--pretrain`: number of pretraining pass on the input network $\mathtt{A}$ and output network $\mathtt{B}$, qualitatively quantifies the starting accuracy (`default=0` corresponds to no pretraining).

- `--method`: numerical integration scheme used by `torchdiffeq.odeint`. Only fixed steps method `'euler'`, `'midpoint'` and `'rk4'` are allowed (`default='midpoint'`).

- `--epochs`: number of training iterations over the whole training dataset (`default=30`).

- `--lr_init`: learning rate initialization (`default=0.5`).

- `--decay_steps`: epochs to which the learning rate is decreased, several arguments can be passed corresponding to several decreasing steps (`default=[25]`).

- `--decay_rate`: factor by which the learning rate is multiplied at decreasing steps (`default=0.1`).

- `--batch_size`: number of images per batch in SGD (`default=256`).

- `--save`: name of the file in which the training data are saved (`default=experiment.pkl`). This file is a `.pkl` file located in the `MNIST_experiments` directory. Warning: script will not run if the file name already exists.

#### Figure 1

  Plots of Figure 1-a can be retrieved running the following scripts:

```
python MNIST_experiments --layers 5 --save RKHS_5layers.pkl
python MNIST_experiments --layers 10 --save RKHS_10layers.pkl
python MNIST_experiments --layers 20 --save RKHS_20layers.pkl
python MNIST_experiments --model 'SHL' --layers 5 --save SHL_5layers.pkl
python MNIST_experiments --model 'SHL' --layers 10 --save SHL_10layers.pkl
python MNIST_experiments --model 'SHL' --layers 20 --save SHL_20layers.pkl
```

  Plots of Figure 1-b can be retrieved running the following scripts:

```
python MNIST_experiments --layers 5 --pretrain 3 --lr_init 0.5 --save RKHS_5layers.pkl
python MNIST_experiments --layers 10 --pretrain 3 --lr_init 0.5 --save RKHS_10layers.pkl
python MNIST_experiments --layers 20 --pretrain 3 --lr_init 0.5 --save RKHS_20layers.pkl
python MNIST_experiments --model SHL --pretrain 3 --lr_init 0.5 --layers 5 --save SHL_5layers.pkl
python MNIST_experiments --model SHL --pretrain 3 --lr_init 0.5 --layers 10 --save SHL_10layers.pkl
python MNIST_experiments --model SHL --pretrain 3 --lr_init 0.5 --layers 20 --save SHL_20layers.pkl
```

#### Figure 2

Plots of Figure 2 can be retrieved running the following scripts:

```
python MNIST_experiments.py --in_channels 32 --dim_int 32 --save 32channels_0pretrain.pkl
python MNIST_experiments.py --in_channels 8 --dim_int 8 --save 8channels_0pretrain.pkl
python MNIST_experiments.py --in_channels 4 --dim_int 4 --save 4channels_0pretrain.pkl
```

## CIFAR_experiments.py

The python script `CIFAR_experiments.py` was used to perform experiments on the CIFAR10 dataset in `Figure 3`. The script can be run using:

```
python CIFAR_experiments.py
```

At the end of training, the architecture achieving the best accuracy on the test set is stored in `CIFAR_checkpoint/ckpt.pth`.

The architecture is inspired from the implementation of ResNet18 that can be found here: [GitHub - kuangliu/pytorch-cifar: 95.47% on CIFAR10 with PyTorch](https://github.com/kuangliu/pytorch-cifar) .

#### Parameters

Several parameters can be parsed into the command line:

- `--model`: set the kind of ResNet used between `'RKHS'` and `'SHL'`. `'RKHS'` corresponds to residuals with fixed hiddenlayers and `'SHL'` to residuals with trained hidden layers (`default='RKHS'`).

- `--layers`: number of layers in the third block of  ResNet18 (`default=10`).

- `--epochs`: number of training iterations over the whole training dataset (`default=300`).

- `--lr_init`: learning rate initialization (`default=3e-3`).

- `--decay_steps`: epochs to which the learning rate is decreased, several arguments can be passed corresponding to several decreasing steps (`default=[260]`).

- `--decay_rate`: factor by which the learning rate is multiplied at decreasing steps (`default=0.1`).

- `--batch_size`: number of images per batch in SGD (`default=256`).

- `--save`: name of the file in which the training data are saved (`default=experiment.pkl`). This file is a `.pkl` file located in the `CIFAR_experiments` directory. Warning: script will not run if the file name already exists.

#### Figure 3

Plots of Figure 3 can be retrieved by running the following scripts:

```
python CIFAR_experiments.py --layers 20 --save RKHS_20layers.pkl
python CIFAR_experiments.py --layers 10 --save RKHS_10layers.pkl
python CIFAR_experiments.py --layers 1 --save RKHS_1layers.pkl
python CIFAR_experiments.py --model SHL --layers 20 --save SHL_20layers.pkl
python CIFAR_experiments.py --model SHL --layers 10 --save SHL_10layers.pkl
python CIFAR_experiments.py --model SHL --layers 5 --save SHL_5layers.pkl
```
