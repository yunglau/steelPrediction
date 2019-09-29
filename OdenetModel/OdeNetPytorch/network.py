import torch
import torch.nn as nn

from OdenetModel.torchdiffeq import odeint_adjoint as odeint
import logging
import os
import time
import numpy as np
import OdenetModel.OdeNetPytorch.simpleDumpedOscillator as sDO


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
        x = torch.from_numpy(x).to(device)
        y = np.array(y[:, None] == np.arange(10)[None, :], dtype=int) #one_hot

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
            hidden_dim : int array
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

        self.fcs = []
        self.fcs += [nn.Linear(self.input_dim,self.hidden_dims[0])]
        for i in range(self.nbLayer-1):
            self.fcs += [nn.Linear(self.hidden_dims[i],self.hidden_dims[i+1])]
        self.fcs = nn.ModuleList(self.fcs)
        self.non_linearity = nn.Tanh()

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
        out = x
        for layer in self.fcs:
            out = layer(out)
            out = self.non_linearity(out)
        return out


class ODEBlock(nn.Module):
    def __init__(self, odefunc, rtol, atol, integration_time):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.from_numpy(integration_time).float()
        #HERE DEFINE INFERENCE TIMES
        #Should be updated depending on the inference time we are trying to make!
        self.rtol = rtol
        self.atol = atol
    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=self.rtol, atol=self.atol)
        return out
    @property
    def nfe(self):
        return self.odefunc.nfe
    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

if __name__ == '__main__':
    #Creation of simple train dataset:
    inputTime = np.arange(0,1000)
    nbrBatches = 100
    nbrSteps = np.random.randint(5,10,nbrBatches) #For real: nbrsteps-1
    Us = []
    Vs = []
    Xs = []
    splits= []
    for i in range(nbrBatches):
        splits += [np.concatenate((np.array([0]),np.sort(np.unique((np.random.randint(0,1000,nbrSteps[i]-2)))),np.array([1000])))]
        Us+=[sDO.getU(inputTime,splits[-1])]
        Vs+=[sDO.getV(inputTime,Us[-1])]
        Xs+=[np.stack((Us[-1],Vs[-1]),axis=1)]


    #creation of test dataset: we switch a little bit the input:
    UnwantedEventTest = [0,100,200,400,700,1000]
    U_test = sDO.getU(inputTime,itv=[0,100,200,400,700,1000])
    V_test = sDO.getV(inputTime,U_test)
    X_test = np.stack((U_test,V_test),axis=1)


    print("Initial data shape:",X_test.shape)
    path = ""
    #logger
    logger = get_logger(logpath=os.path.join(path, 'logs'), filepath=os.path.abspath(__file__))

    #device creation
    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

    #hyperparameters
    #network hyperparameters
    input_dim = 2
    nbLayer = 10
    hidden_dims = [10 for n in range(nbLayer)]
    hidden_dims[-1] = input_dim
    augment_dim= 0
    time_dependent = False
    rtol = 10**(-6)
    atol =  10**(-3)
    #training hyperparameters
    batches_per_epoch = nbrBatches #Nbr of available batches
    lr = 0.01
    nepochs = 10
    batch_size = 1

    feature_layers = [ODEBlock(ODEfunc(input_dim,hidden_dims,nbLayer,augment_dim,time_dependent),rtol,atol,inputTime)]

    model = nn.Sequential(*feature_layers).to(device)

    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))

    criterion = nn.MSELoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)

    best_loss = -1
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()

    lr_fn = learning_rate_with_decay(
        batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001], lr= lr
    )

    print("MODEL BUILT, Starting training")
    X_train = torch.from_numpy(np.array(Xs)).to(device)
    print("Data loaded on GPU, starting inference")
    for itr in range(nepochs * batches_per_epoch):
        batch_Idx = int( itr / batches_per_epoch )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)
        optimizer.zero_grad()
        #For every time range between perturbations:
        # we ask the model to predict the system  evolutions on this time range.
        # We store the output and start from scratch with different updated initial values
        # The model will progressively learns the real system dynamic without being able to predict "Uncorelated perturbations...."
        # Doing this over many different time ranges enable model to memorize effect of various instantaneous perturbations
        mini_logits = []
        for idx,init in enumerate(splits[itr][:-1]):
            x = X_train[itr][init] #,torch.from_numpy(Xs[itr][init:splits[itr][idx+1]])
            #Predict the system at all time step before another perturbation!
            x = x.to(device)
            #Update of the range on which the ODEsolver should be ran
            feature_layers[0].integration_time = torch.from_numpy(np.arange(init,splits[itr][idx+1])).float().to(device)
            mini_logits += [model(x)]
        logits = torch.cat(mini_logits,0)
        Y = X_train[itr]
        loss = criterion(logits, Y)

        nfe_forward = feature_layers[0].nfe
        feature_layers[0].nfe = 0
        print("Starting loss backward")
        loss.backward()
        optimizer.step()

        nfe_backward = feature_layers[0].nfe
        feature_layers[0].nfe = 0
        batch_time_meter.update(time.time() - end)
        f_nfe_meter.update(nfe_forward)
        b_nfe_meter.update(nfe_backward)
        end = time.time()
        print("==========Next itr===========")
        if itr % batches_per_epoch == 0:
            with torch.no_grad():
                train_loss = loss
                #Test loss is against one the splittest
                mini_logits_test =[]
                for idx,init in enumerate(UnwantedEventTest):
                    x, y = torch.from_numpy(X_test[init]),torch.from_numpy(X_test[init:UnwantedEventTest[idx+1]])
                    #Predict the system at all time step before another perturbation!
                    x = x.to(device)
                    #Update of the range on which the ODEsolver should be ran
                    feature_layers[0].integration_time = torch.from_numpy(np.arange(init,UnwantedEventTest[idx+1])).float().to(device)
                    mini_logits_test += [model(x)]
                logits_test = torch.cat(mini_logits_test)
                y_test = torch.from_numpy(X_test).to(device)
                test_loss = criterion(logits_test, y_test) #todo

                if best_loss > test_loss:
                    torch.save({'state_dict': model.state_dict()}, os.path.join(path, 'model.pth'))
                    best_loss = test_loss
                logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                    "Train Loss {:.4f} | Test Loss {:.4f}".format(
                        itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                        b_nfe_meter.avg, train_loss, test_loss
                    )
                )

    x_test, y_test = torch.from_numpy(X_test[:-1]),torch.from_numpy(X_test[1:]) #The prediction should be the value at next time step!
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    logits_test = model(x_test)
    Vpred_test = logits_test.cpu().detach().numpy()[:,1]

    mini_logits_test =[]
    for idx,init in enumerate(UnwantedEventTest):
        x, y = torch.from_numpy(X_test[init]),torch.from_numpy(X_test[init:UnwantedEventTest[idx+1]])
        #Predict the system at all time step before another perturbation!
        x = x.to(device)
        #Update of the range on which the ODEsolver should be ran
        feature_layers[0].integration_time = torch.from_numpy(np.arange(init,UnwantedEventTest[idx+1])).float().to(device)
        mini_logits_test += [model(x)]
    logits_test = torch.cat(mini_logits_test)
    y_test = torch.from_numpy(X_test).to(device)

    Vpred_test = logits_test.cpu().detach().numpy()[:,1]

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(inputTime[:-1],U_test[:-1],c="r")
    plt.plot(inputTime[:-1],V_test[:-1],c="g")
    plt.plot(inputTime[:-1],Vpred_test,c="b")
    plt.show()
    fig.savefig("predTestSet.png")



