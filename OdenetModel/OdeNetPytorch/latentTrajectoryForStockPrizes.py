import os
import argparse
import logging
import time
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=False)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
args = parser.parse_args()
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

def obtainData(ntotal,noise_std,HORIZON,WINDOW_SIZE):
    # convert series to supervised learning

    # def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    #     n_vars = 1 if type(data) is list else data.shape[1]
    #     df = DataFrame(data)
    #     cols, names = list(), list()
    #     # input sequence (t-n, ... t-1)
    #     for i in range(n_in, 0, -1):
    #         cols.append(df.shift(i))
    #         names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    #     # forecast sequence (t, t+1, ... t+n)
    #     for i in range(0, n_out):
    #         cols.append(df.shift(-i))
    #         if i == 0:
    #             names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    #         else:
    #             names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    #     # put it all together
    #     agg = concat(cols, axis=1)
    #     agg.columns = names
    #     # drop rows with NaN values
    #     if dropnan:
    #         agg.dropna(inplace=True)
    #     return agg
    # def value_to_float(x):
    #     if type(x) == float or type(x) == int:
    #         return x
    #     if 'K' in x:
    #         if len(x) > 1:
    #             return float(x.replace('K', '')) * 1000
    #         return 1000.0
    # # load dataset
    # dataset = read_csv(PATH, header=0, index_col=0)
    # feature_num = dataset.shape[1]
    # values = dataset.values.reshape(-1,feature_num)
    # # ensure all data is float
    # values = values.astype('float32')
    # # frame as supervised learning
    # reframed = series_to_supervised(values,WINDOW_SIZE,HORIZON)
    # # normalize features
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled = scaler.fit_transform(reframed)
    # def split(dataset,horizon,feature_num):
    #     return dataset[:, :-horizon*feature_num], dataset[:, -horizon*feature_num:]
    # # split data into train, validation and test sets
    # train_size=int(len(scaled)*0.7)
    # val_size=int(len(scaled)*0.2)
    # test_size=int(len(scaled)*0.1)
    # trainset = scaled[:train_size, :]
    # print(trainset.shape)
    # valset = scaled[train_size:train_size+val_size, :]
    # testset = scaled[-test_size:, :]
    # # split data into inputs and outputs
    # train_X, train_y = split(trainset,HORIZON,feature_num)
    # val_X, val_y = split(valset,HORIZON,feature_num)
    # test_X, test_y = split(testset,HORIZON,feature_num)
    # # reshape input to be 3D [#samples, #timesteps, #features]
    # train_X = train_X.reshape(train_X.shape[0], WINDOW_SIZE, feature_num)
    # val_X = val_X.reshape(val_X.shape[0], WINDOW_SIZE, feature_num)
    # test_X = test_X.reshape(test_X.shape[0], WINDOW_SIZE, feature_num)
    # print(train_X.shape, train_y.shape, val_X.shape, val_y.shape, test_X.shape, test_y.shape)
    #DATA Generation for latent model:
    PATH = 'Data_Merged.csv'
    dataset = read_csv(PATH, header=0, index_col=0)
    feature_num = dataset.shape[1]
    values = dataset.values.reshape(-1,feature_num)
    # ensure all data is float
    values = values.astype('float32') #todo for debug
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    x_train = scaled[:-2*HORIZON] #All Except final 2 horizons
    x_test = scaled[-2*HORIZON:-1*HORIZON]
    x_valid = scaled[-1*HORIZON:]

    orig_ts = np.arange(0, x_train.shape[0])
    samp_ts = orig_ts[:WINDOW_SIZE]
    nsample = WINDOW_SIZE

    orig_trajs = []
    samp_trajs = []
    samp_times = []
    for _ in range(ntotal):
        # don't sample t0 very near the start or the end
        t0_idx = npr.multinomial(1, [1. / (ntotal - 2. * nsample)] * (ntotal - int(2 * nsample)))
        t0_idx = np.argmax(t0_idx) + nsample
        orig_trajs.append(x_train)

        samp_traj = x_train[t0_idx:t0_idx + WINDOW_SIZE, :].copy()
        samp_traj += npr.randn(*samp_traj.shape) * noise_std
        samp_trajs.append(samp_traj)
        samp_times += [[t0_idx,t0_idx + WINDOW_SIZE]]

    # batching for sample trajectories is good for RNN; batching for original
    # trajectories only for ease of indexing
    orig_trajs = np.stack(orig_trajs, axis=0)
    samp_trajs = np.stack(samp_trajs, axis=0)

    return orig_trajs, samp_trajs, orig_ts, samp_ts, x_test,x_valid,samp_times

class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class RecognitionRNN(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return torch.zeros(self.nbatch, self.nhidden)


class Decoder(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out


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


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


if __name__ == '__main__':

    HORIZON = 10 #The prediction that our model should be able to achieve !
    WINDOW_SIZE = 30 #How much data we use per mini-batch to fit the models

    niters = 2 #set to around 1000 for inference
    latent_dim = 4
    nhidden = 20
    rnn_nhidden = 25
    obs_dim = 9
    nspiral = 1000
    noise_std =.3
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    # generate toy spiral data
    orig_trajs, samp_trajs, orig_ts, samp_ts ,x_test,x_valid,samp_times= obtainData(nspiral,noise_std,HORIZON,WINDOW_SIZE)

    orig_trajs = torch.from_numpy(orig_trajs).float().to(device)
    samp_trajs = torch.from_numpy(samp_trajs).float().to(device)
    samp_ts = torch.from_numpy(samp_ts).float().to(device)

    # model
    func = LatentODEfunc(latent_dim, nhidden).to(device)
    rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, nspiral).to(device)
    dec = Decoder(latent_dim, obs_dim, nhidden).to(device)
    params = (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)
    loss_meter = RunningAverageMeter()

    if args.train_dir is not None:
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
        ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            rec.load_state_dict(checkpoint['rec_state_dict'])
            dec.load_state_dict(checkpoint['dec_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            orig_trajs = checkpoint['orig_trajs']
            samp_trajs = checkpoint['samp_trajs']
            orig_ts = checkpoint['orig_ts']
            samp_ts = checkpoint['samp_ts']
            print('Loaded ckpt from {}'.format(ckpt_path))

    try:
        for itr in range(1, niters + 1):
            optimizer.zero_grad()
            # backward in time to infer q(z_0)
            h = rec.initHidden().to(device)
            for t in reversed(range(samp_trajs.size(1))):
                obs = samp_trajs[:, t, :]
                out, h = rec.forward(obs, h)
            qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

            # forward in time and solve ode for reconstructions
            pred_z = odeint(func, z0, samp_ts).permute(1, 0, 2)
            pred_x = dec(pred_z)

            # compute loss
            noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
            noise_logvar = 2. * torch.log(noise_std_).to(device)
            logpx = log_normal_pdf(
                samp_trajs, pred_x, noise_logvar).sum(-1).sum(-1)
            pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
            analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                    pz0_mean, pz0_logvar).sum(-1)
            loss = torch.mean(-logpx + analytic_kl, dim=0)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())

            print('Iter: {}, running avg elbo: {:.4f}'.format(itr, -loss_meter.avg))

    except KeyboardInterrupt:
        if args.train_dir is not None:
            ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
            torch.save({
                'func_state_dict': func.state_dict(),
                'rec_state_dict': rec.state_dict(),
                'dec_state_dict': dec.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'orig_trajs': orig_trajs,
                'samp_trajs': samp_trajs,
                'orig_ts': orig_ts,
                'samp_ts': samp_ts,
            }, ckpt_path)
            print('Stored ckpt at {}'.format(ckpt_path))
    print('Training complete after {} iters.'.format(itr))

    # if args.visualize:
    #FIRST: we visualize that the model retained the training trajectory:
    with torch.no_grad():
        # sample from trajectorys' approx. posterior

        ##Heuristic: we devide the input into slides of size WINDOW_SIZE and verify that the model manages it over those slides
        #No sliding mean!!
        L = orig_ts.shape[0]
        results = np.zeros(L)
        orig_ts = torch.from_numpy(orig_ts).float().to(device)
        for i in range(0,L-WINDOW_SIZE,WINDOW_SIZE):
            trajectory = orig_trajs[0,i:i+WINDOW_SIZE]
            # access the original trajectory
            h = rec.initHidden().to(device)[0].reshape(1,25)
            for t in reversed(range(trajectory.size(0))):
                obs = trajectory[t, :].reshape((1,9))
                out, h = rec.forward(obs, h)
            qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

            #ONLY ONE TRAJECTORY:
            print("starting inference,",i)
            xs_pos = odeint(func, z0[0],samp_ts) #.permute(1, 0, 2) #We can use samp_ts since the time is not used by the model...
            results[i:i+WINDOW_SIZE] = dec(xs_pos).cpu().numpy()[:,-1]
    true_sotck_prize = orig_trajs[0].cpu().numpy()
    original_ts = orig_ts.cpu().numpy()
    plt.figure()
    plt.plot(original_ts, results, 'r', label='learned trajectory (t>0)')
    plt.plot(original_ts, true_sotck_prize[:, -1], label='true trajectory')
    plt.legend()
    plt.savefig('./visTrainRegression.png', dpi=500)
    print('Saved visualization figure at {}'.format('./visTrainRegression.png'))
    #COMPUTING TRAINING LOSS FOR :
    #Sum of distance between training and results
    s_sliding = np.sum(np.square(results-true_sotck_prize[:, -1]))
    print("Score for regression with sliding window on training set is:",s_sliding)

    #INFERENCE OF NEXT HORIZONS:
    #The sliding window is now the horizon window!!!
    inference_ts = torch.from_numpy(np.arange(0,WINDOW_SIZE+HORIZON)).float().to(device)
    results = np.zeros(L-WINDOW_SIZE)
    for i in range(0,L-WINDOW_SIZE-HORIZON,HORIZON):
        trajectory = orig_trajs[0,i:i+WINDOW_SIZE] #trajectory input is still of size WINDOW_SIZE
        # access the original trajectory
        h = rec.initHidden().to(device)[0].reshape(1,25)
        for t in reversed(range(trajectory.size(0))):
            obs = trajectory[t, :].reshape((1,9))
            out, h = rec.forward(obs, h)
        qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

        #ONLY ONE TRAJECTORY:
        print("starting inference,",i)
        xs_pos = odeint(func, z0[0],inference_ts) #.permute(1, 0, 2) #We can use samp_ts since the time is not used by the model...
        results[i:i+HORIZON] = dec(xs_pos).cpu().detach().numpy()[WINDOW_SIZE:,-1]

    true_sotck_prize = orig_trajs[0].cpu().numpy()
    original_ts = orig_ts.cpu().numpy()[WINDOW_SIZE:]
    plt.figure()
    plt.plot(original_ts, results, 'r', label='learned horizon (t>0)')
    plt.plot(original_ts, true_sotck_prize[WINDOW_SIZE:, -1], label='true trajectory')
    plt.legend()
    plt.savefig('./visTrainInference.png', dpi=500)
    print('Saved visualization figure at {}'.format('./visTrainInference.png'))
    #COMPUTING TRAINING LOSS FOR :
    #Sum of distance between training and results
    s_horizon = np.sum(np.square(results-true_sotck_prize[WINDOW_SIZE:, -1]))
    print("Score for prediction with sliding window on training set is:",s_horizon)

    ##NOW WE PREDICT AGAINST THE VALIDATION SET!!!
    #So we predict a latent space with the last data slide and build an horizon on it.
    inference_ts = torch.from_numpy(np.arange(0,WINDOW_SIZE+HORIZON)).float().to(device)
    results = np.zeros(HORIZON)
    trajectory = orig_trajs[0,-WINDOW_SIZE:,:] #trajectory input is still of size WINDOW_SIZE
    # access the original trajectory
    h = rec.initHidden().to(device)[0].reshape(1,25)
    for t in reversed(range(trajectory.size(0))):
        obs = trajectory[t, :].reshape((1,9))
        out, h = rec.forward(obs, h)
    qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
    epsilon = torch.randn(qz0_mean.size()).to(device)
    z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean


    #ONLY ONE TRAJECTORY:
    xs_pos = odeint(func, z0[0],inference_ts) #.permute(1, 0, 2) #We can use samp_ts since the time is not used by the model...
    results = dec(xs_pos).cpu().detach().numpy()[WINDOW_SIZE:,-1]

    true_sotck_prize = x_test[:,-1]
    original_ts = np.arange(L,L+HORIZON)
    plt.figure()
    plt.plot(original_ts, results, 'r', label='learned horizon for never seen')
    plt.plot(original_ts, true_sotck_prize, label='true trajectory')
    plt.legend()
    plt.savefig('./visTestInference.png', dpi=500)
    print('Saved visualization figure at {}'.format('./visTestInference.png'))
    #COMPUTING TRAINING LOSS FOR :
    #Sum of distance between training and results
    s_horizon = np.sum(np.square(results-true_sotck_prize))
    print("Score for prediction with sliding window on end of training set to predict testing set is:",s_horizon)




