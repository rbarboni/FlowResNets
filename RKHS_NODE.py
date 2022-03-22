import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq as tde
from tqdm import tqdm, trange
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def classif_loss(x: torch.Tensor, y: torch.Tensor) -> float:
    x_max = torch.max(x, dim=1)[1]
    return (x_max != y).to(torch.float).mean()

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

class RKHS_ODEfunc(nn.Module):

    def __init__(self, dim, dim_int,
                 num_steps,
                 zero_init=True,
                 Omega=None):
        super(RKHS_ODEfunc, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim_int, 3, padding=1, bias=False)
        if Omega is not None:
            self.conv1.weight = nn.Parameter(data=Omega)
        else:
            self.conv1.weight = nn.Parameter(data=(torch.randn(dim_int, dim, 3, 3) / (3 * dim_int)**0.5))
        self.conv1.weight.requires_grad = False
        self.num_steps = num_steps
        self.block_list = nn.ModuleList([nn.Conv2d(dim_int, dim, 3, padding=1, bias=False) for _ in range(num_steps+1)])
        if zero_init:
            for m in self.block_list:
                m.weight = nn.Parameter(data=torch.zeros_like(m.weight))
            print('ODEfunc initialized at 0')

    def forward(self, t, x):
        k = int(t * self.num_steps - 1)
        out = F.relu(self.conv1(x))
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
        self.visible_layer_list = nn.ModuleList([nn.Conv2d(dim_int, dim, 3, padding=1, bias=False) for _ in range(num_steps+1)])
        self.hidden_layer_list = nn.ModuleList([nn.Conv2d(dim, dim_int, 3, padding=1, bias=False) for _ in range(num_steps+1)])
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

class RKHS_Block(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, zero_init=True):
        super(RKHS_Block, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.conv1.weight = nn.Parameter(data=(torch.randn(planes, in_planes, 3, 3) / (9 * in_planes) ** 0.5))
        self.conv1.weight.requires_grad = False
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if zero_init:
            self.conv2.weight = nn.Parameter(data=torch.zeros_like(self.conv2.weight))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                          nn.BatchNorm2d(self.expansion*planes)
                          )
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class SHL_Block(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, zero_init=True):
        super(SHL_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if zero_init:
            self.conv2.weight = nn.Parameter(data=torch.zeros_like(self.conv2.weight))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
                          )
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks,
                 num_classes=10,
                 zero_inits=None):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        if zero_inits is None:
            zero_inits = [False, False, True, False]
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, zero_init=zero_inits[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, zero_init=zero_inits[0])
        self.bn2 = nn.BatchNorm2d(128)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, zero_init=zero_inits[0])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, zero_init=zero_inits[0])
        self.pool = nn.AvgPool2d(kernel_size=4)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, zero_init=True):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        if zero_init:
            print('Residuals initialized at 0')
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.bn2(self.layer2(out))
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def accuracy(model, loader,
             criterion=classif_loss,
             input_embedding=nn.Identity(),
             output_embedding=nn.Identity(),
             averaged=True):

    if averaged:
        res = 0
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = output_embedding(model(input_embedding(inputs)))
            res += criterion(outputs, targets) * targets.shape[0]
        return res / len(loader.dataset)
    else:
        inputs, targets = next(iter(loader))
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = output_embedding(model(input_embedding(inputs)))
        return criterion(outputs, targets)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_sgd(model, train_loader, train_eval_loader, test_loader,
              loss_fn=nn.CrossEntropyLoss(),
              input_embedding=nn.Identity().to(device),
              output_embedding=nn.Identity().to(device),
              epochs=1, lr_init=0.1, decay_steps=None, decay_rate=0.1,
              save_best=True, ckpt_dir='checkpoint'):

    if torch.cuda.device_count() > 1:
        input_embedding = torch.nn.DataParallel(input_embedding)
        output_embedding = torch.nn.DataParallel(output_embedding)
        model = torch.nn.DataParallel(model)

    model = model.to(device)
    input_embedding = input_embedding.to(device)
    output_embedding = output_embedding.to(device)

    model.train()
    input_embedding.train(mode=False)
    input_embedding.requires_grad_(requires_grad=False)
    output_embedding.train(mode=False)
    output_embedding.requires_grad_(requires_grad=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr_init)

    if decay_steps is None:
        decay_steps = []
    print(f'Training: epochs={epochs}, lr_init={lr_init}, decay_rate={decay_rate}, decay_steps={decay_steps}')
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=decay_steps,
                                                     gamma=decay_rate)

    train_loss = torch.empty(0)
    train_classif_loss = torch.empty(0)
    test_classif_loss = torch.empty(0)

    loss_fn = loss_fn.to(device)

    with torch.no_grad():
        model.eval()

        loss = accuracy(model, train_eval_loader,
                        input_embedding=input_embedding,
                        output_embedding=output_embedding,
                        averaged=False)
        train_classif_loss = torch.cat((train_classif_loss,
                                        loss.detach().cpu().expand((1,))))

        loss = accuracy(model, test_loader,
                        input_embedding=input_embedding,
                        output_embedding=output_embedding)
        test_classif_loss = torch.cat((test_classif_loss,
                                        loss.detach().cpu().expand((1, ))))
        #print(f'test error rate = {100*test_classif_loss[-1]:.2f}%')

    best = test_classif_loss[-1]

    tepochs = trange(epochs, desc='Best: , Current: ', position=0, leave=True)

    for epoch in tepochs:
        tepochs.set_description(f'Best: {100*best:.2f}%, Current: {100*test_classif_loss[-1]:.2f}%',
                                refresh=True)

        model.train()
        for inputs, targets in train_loader: ## training loop
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            inputs = input_embedding(inputs)
            outputs = model(inputs)
            outputs = output_embedding(outputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            if torch.isnan(outputs.detach()).sum() > 0:
                print('Output is NaN: exiting training')
                return train_loss, train_classif_loss, test_classif_loss
            train_loss = torch.cat((train_loss, loss.detach().cpu().expand((1,))))

        with torch.no_grad():
            model.eval()

            loss = accuracy(model, train_eval_loader,
                            input_embedding=input_embedding,
                            output_embedding=output_embedding,
                            averaged=False)
            train_classif_loss = torch.cat((train_classif_loss,
                                            loss.detach().cpu().expand((1,))))

            loss = accuracy(model, test_loader,
                            input_embedding=input_embedding,
                            output_embedding=output_embedding)
            test_classif_loss = torch.cat((test_classif_loss,
                                            loss.detach().cpu().expand((1, ))))
            #print(f'test error rate= {100*test_classif_loss[-1]:.2f}%')

        if test_classif_loss[-1] - best < -1e-3:
            best = test_classif_loss[-1]
            if save_best:
                #print('Saving...')
                state = {
                    'model': model.state_dict(),
                    'classif_error': best,
                    'epoch': epoch,
                }
                if not os.path.isdir(ckpt_dir):
                    os.mkdir(ckpt_dir)
                torch.save(state, './'+ckpt_dir+'/ckpt.pth')

        scheduler.step()

    print(f'Best classification error rate reached: {100*best:.2f}')
    return train_loss, train_classif_loss, test_classif_loss