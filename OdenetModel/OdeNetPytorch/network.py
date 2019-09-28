import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
import logging
import os
import time
import numpy as np

MAX_NUM_STEPS = 1000  # Maximum number of steps for ODE solver
#LOGGER tools
def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def accuracy(model, data):
    total_correct = 0
    for x, y in data:
        x = x.to(device)

        y = np.array(np.array(y.numpy())[:, None] == np.arange(10)[None, :], dtype=int) #one_hot

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(data)

#Metrics tools
def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates, lr):
    initial_learning_rate = lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()
    def reset(self):
        self.val = None
        self.avg = 0
    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


class ODEfunc(nn.Module):

    def __init__(self,input_dim, hidden_dims,nbLayer, augment_dim=0,
                 time_dependent=False,
                 **kwargs):
        """
        MLP modeling the derivative of ODE system.

        # Arguments:
            input_dim : int
                Dimension of data.
            hidden_dim : int
                Dimension of hidden layers.
            augment_dim: int
                Dimension of augmentation. If 0 does not augment ODE, otherwise augments
                it with augment_dim dimensions.
            time_dependent : bool
                If True adds time as input, making ODE time dependent.
            non_linearity : string
                One of 'relu' and 'softplus'
        """
        super(ODEfunc, self).__init__()
        self.augment_dim = augment_dim

        self.hidden_dims = hidden_dims
        self.nbLayer = nbLayer
        self.nfe = 0  # Number of function evaluations
        self.time_dependent = time_dependent
        self.input_dim = input_dim + augment_dim
        if self.time_dependent:
            self.input_dim +=1

        self.fc = []
        self.fcs += [nn.Linear(self.input_dim,self.hidden_dims[0])]
        for i in range(self.nbLayer-1):
            self.fcs += [nn.Linear(self.hidden_dims[i],self.hidden_dims[i+1])]
        self.non_linearity = nn.ReLu(inplace=True)

    def forward(self, t, x):
        """
        Forward pass. If time dependent, concatenates the time
        dimension onto the input before the call to the dense layer.
        # Arguments:
            t: Tensor. Current time. Shape (1,).
            x: Tensor. Shape (batch_size, input_dim).
        # Returns:
            Output tensor of forward pass.
        """
        # Forward pass of model corresponds to one function evaluation, so
        # increment counter
        self.nfe += 1
        if self.time_dependent:
            out = torch.cat((t,x),0)
        else:
            out = x
        for layer in self.fcs:
            out = layer(out)
            out = self.non_linearity(out)
        return out


class ODEBlock(nn.Module):
    def __init__(self, odefunc, rtol, atol):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.rtol = rtol
        self.atol = atol
    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=self.rtol, atol=self.atol)
        return out[1]
    @property
    def nfe(self):
        return self.odefunc.nfe
    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

if __name__ == '__main__':
    path = ""
    #logger
    logger = get_logger(logpath=os.path.join(path, 'logs'), filepath=os.path.abspath(__file__))

    #device creation
    device = torch.device('cuda:' + 0 if torch.cuda.is_available() else 'cpu')

    #hyperparameters
    #network hyperparameters
    input_dim = 10
    hidden_dims = 10
    nbLayer = 10
    augment_dim= 3
    time_dependent = True
    nbBlock = 1
    #training hyperparameters
    batches_per_epoch = 1
    lr = 0.01
    nepochs = 100
    batch_size = 10

    feature_layers = [ODEBlock(ODEfunc(input_dim,hidden_dims,nbLayer,augment_dim,time_dependent)) for _ in range(nbBlock)]

    model = nn.Sequential(*feature_layers).to(device)

    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)

    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()

    lr_fn = learning_rate_with_decay(
        batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001], lr= lr
    )

    data = None
    data_test = None

    for itr in range(nepochs * batches_per_epoch):

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)

        optimizer.zero_grad()
        x, y = None #TODO
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        nfe_forward = feature_layers[0].nfe
        feature_layers[0].nfe = 0

        loss.backward()
        optimizer.step()

        nfe_backward = feature_layers[0].nfe
        feature_layers[0].nfe = 0
        batch_time_meter.update(time.time() - end)
        f_nfe_meter.update(nfe_forward)
        b_nfe_meter.update(nfe_backward)
        end = time.time()

        if itr % batches_per_epoch == 0:
            with torch.no_grad():
                train_acc = accuracy(model, data=data)
                val_acc = accuracy(model, data=data_test)
                if val_acc > best_acc:
                    torch.save({'state_dict': model.state_dict()}, os.path.join(path, 'model.pth'))
                    best_acc = val_acc
                logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                    "Train Acc {:.4f} | Test Acc {:.4f}".format(
                        itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                        b_nfe_meter.avg, train_acc, val_acc
                    )
                )