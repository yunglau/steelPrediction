import torch
import torch.nn as nn
import random
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
        nn.init.constant_(self.fcs[-1].weight,0) # We initialize our last layer at 0 so that the initial derivative is identity!
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
        for idx,layer in enumerate(self.fcs):
            out = layer(out)
            if idx!=len(self.fcs)-1:
                out = self.non_linearity(out)
        return out


class ODEBlock(nn.Module):
    def __init__(self, odefunc, rtol, atol, integration_time):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.from_numpy(integration_time).float().to(device)
        #HERE DEFINE INFERENCE TIMES
        #Should be updated depending on the inference time we are trying to make!
        self.rtol = rtol
        self.atol = atol
    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time) #, rtol=self.rtol, atol=self.atol
        return out
    @property
    def nfe(self):
        return self.odefunc.nfe
    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

def get_batch(data_size,batch_time,batch_size,true_y,t):
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False)).to(device)
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y


if __name__ == '__main__':
    #Creation of simple train dataset:
    data_size=1000
    inputTime = np.array(np.arange(0,100,100/data_size),dtype=np.float32)
    nbrBatches = 10
    nbrSteps = np.random.randint(5,10,nbrBatches) #For real: nbrsteps-1
    Us = []
    Vs = []
    Xs = []
    splits= []
    for i in range(nbrBatches):
        newRange = np.sort(np.array(random.sample(range(0,100,10),nbrSteps[i]-2)))
        while 0 in newRange:
            newRange =np.sort(np.array(random.sample(range(0,100,10),nbrSteps[i]-2)))
        splits += [np.concatenate((np.array([0]),newRange,np.array([100])))]
        Us+=[sDO.getU(inputTime,splits[-1])]
        Vs+=[sDO.getV(inputTime,Us[-1])]
        Xs+=[np.stack((Us[-1],Vs[-1]),axis=1)]
    #creation of test dataset: we switch a little bit the input:
    UnwantedEventTest = [0,10,20,40,70,100]
    U_test = sDO.getU(inputTime,itv=[0,10,20,40,70,100])
    V_test = sDO.getV(inputTime,U_test)
    X_test = np.stack((U_test,V_test),axis=1)

    print("Initial data shape:",X_test.shape)
    path = ""
    #logger
    logger = get_logger(logpath=os.path.join(path, 'logs'), filepath=os.path.abspath(__file__))
    #device creation
    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    print("Running on GPU? ",torch.cuda.is_available())
    #hyperparameters
    #network hyperparameters
    input_dim = 2
    nbLayer = 3
    hidden_dims = [100 for n in range(nbLayer)]
    augment_dim= 2
    hidden_dims[-1] = input_dim + augment_dim

    time_dependent = False
    rtol = 10**(-6)
    atol =  10**(-3)
    #training hyperparameters
    batches_per_epoch = nbrBatches #Nbr of available batches
    lr = 0.1
    nepochs = 10
    batch_size = 20
    batch_time = 10
    mini_batch_size = 20
    gradientSparsity = 0.1
    integration_time = np.arange(0,10,1)
    feature_layers = [ODEBlock(ODEfunc(input_dim,hidden_dims,nbLayer,augment_dim,time_dependent),rtol,atol,integration_time)]

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

    print("MODEL BUILT, Moving data set to device")
    #First need to add to the input the additional freedom degree:
    Xs = np.array(Xs)
    freedom = np.zeros(list(Xs.shape[:-1])+[2])
    Xs = np.concatenate((Xs,freedom),axis=-1)
    X_train = torch.from_numpy(np.array(Xs,dtype=np.float32)).to(device)

    X_test = np.array(X_test)
    freedom = np.zeros(list(X_test.shape[:-1])+[2])
    X_test = np.concatenate((X_test,freedom),axis=-1)
    X_test_torch = torch.from_numpy(np.array(X_test,dtype=np.float32)).to(device)

    print("Data loaded on GPU, starting inference")
    for itr in range(nepochs * batches_per_epoch):
        batch_Idx = int( itr / batches_per_epoch )
        batch_itr = itr % batches_per_epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)
        optimizer.zero_grad()

        # mini_logits = []
        # mini_output = []
        # out = X_train[batch_itr][0].detach()
        nfe_forward = feature_layers[0].nfe
        feature_layers[0].nfe = 0
        #feature_layers[0].integration_time = torch.from_numpy(np.arange(0,10,1)).to(device)
        # for idx,init in enumerate(splits[batch_itr][:-1]):
        #     optimizer.zero_grad()
        #     x = out
        #     x[0] = X_train[batch_itr][0,0] # !!PERTURBATION!!! ==> only on one value here, other one remains the previous value
        #     #,torch.from_numpy(Xs[itr][init:splits[itr][idx+1]])
        #     #Predict the system at all time step before another perturbation!
        #     x = x.to(device)
        #     #Update of the range on which the ODEsolver should be ran
        #     #max(int(gradientSparsity*(splits[batch_itr][idx+1]-init)),1)
        #     newRange = np.sort(np.array(random.sample(range(init,splits[batch_itr][idx+1],1),10)))
        #     feature_layers[0].integration_time = torch.from_numpy(newRange).float().to(device)
        #     mini_logits += [model(x)] #No backprop on free values
        #     mini_output += [X_train[batch_itr][newRange,:input_dim]]
        #     loss = criterion(mini_logits[-1][:,:input_dim],mini_output[-1])
        #     print("starting backward")
        #     loss.backward()
        #     print("starting opti")
        #     optimizer.step()
        #     print("ended opti")
        #     out = mini_logits[-1][-1].detach()
        batch_y0, batch_t, batch_y = get_batch(data_size,batch_time,mini_batch_size,X_train[batch_itr],inputTime)
        feature_layers[0].integration_time = torch.from_numpy(batch_t).float().to(device)
        pred_y = model(batch_y0)
        loss = criterion(pred_y[:,:,:input_dim],batch_y[:,:,:input_dim])
        loss.backward()
        optimizer.step()

        nfe_backward = feature_layers[0].nfe
        feature_layers[0].nfe = 0
        batch_time_meter.update(time.time() - end)
        f_nfe_meter.update(nfe_forward)
        b_nfe_meter.update(nfe_backward)
        end = time.time()
        print("===== ITERATION ===== ",itr," last loss",loss)
        if itr % batches_per_epoch == 0:
            with torch.no_grad():
                train_loss = loss

                mini_logits_test =[]
                for idx,init in enumerate(UnwantedEventTest[:-1]):
                    x = X_test_torch[init]
                    #Predict the system at all time step before another perturbation!
                    x = x.to(device)
                    #Update of the range on which the ODEsolver should be ran
                    feature_layers[0].integration_time = torch.from_numpy(np.arange(init,UnwantedEventTest[idx+1],0.1)).float().to(device)
                    mini_logits_test += [model(x)[:,:input_dim]]
                logits_test = torch.cat(mini_logits_test)
                y_test = X_test_torch[:,:input_dim]
                test_loss = criterion(logits_test, y_test) #todo

                if best_loss > test_loss or best_loss==-1:
                    torch.save({'state_dict': model.state_dict()}, os.path.join(path, 'model.pth'))
                    best_loss = test_loss
                logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                    "Train Loss {:.4f} | Test Loss {:.4f}".format(
                        itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                        b_nfe_meter.avg, train_loss, test_loss
                    )
                )
    #Train example:
    batch_y0, batch_t, batch_y = get_batch(data_size,batch_time,mini_batch_size,X_train[-1],inputTime)
    feature_layers[0].integration_time = torch.from_numpy(batch_t).float().to(device)
    pred_V = model(batch_y0)[:,0,1].cpu().detach().numpy()
    timeArray = batch_t
    true_V = batch_y[:,0,1].cpu().detach().numpy()
    true_U = batch_y[:,0,0].cpu().detach().numpy()
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(timeArray,true_U,c="r")
    plt.plot(timeArray,true_V,c="g")
    plt.plot(timeArray,pred_V,c="b")
    plt.show()
    fig.savefig("predTrainSet.png")

    mini_logits_test =[]
    for idx,init in enumerate(UnwantedEventTest[:-1]):
        x, y = X_test_torch[init],X_test_torch[init:UnwantedEventTest[idx+1]]
        #Predict the system at all time step before another perturbation!
        x = x.to(device)
        #Update of the range on which the ODEsolver should be ran
        feature_layers[0].integration_time = torch.from_numpy(np.arange(init,UnwantedEventTest[idx+1],0.1)).float().to(device)
        mini_logits_test += [model(x)[:,:input_dim]]
    logits_test = torch.cat(mini_logits_test)
    y_test = X_test_torch[:input_dim]
    Vpred_test = logits_test.cpu().detach().numpy()[:,1]

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(inputTime,U_test,c="r")
    plt.plot(inputTime,V_test,c="g")
    plt.plot(inputTime,Vpred_test,c="b")
    plt.show()
    fig.savefig("predTestSet.png")



