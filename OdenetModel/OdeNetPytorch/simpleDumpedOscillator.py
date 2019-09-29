import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
def getU(time,itv = [0,100,200,400,600,1000]):
    u = np.zeros(time.shape[0],dtype=np.float32)
    for idx,i in enumerate(itv[:-1]):
        if idx%2==0:
            u[itv[idx]:itv[idx+1]] = 0
        else:
            u[itv[idx]:itv[idx+1]] = 1
    return u

kappa = 0.5
eta = 0.2

def g(t):
    if t>=1000:
        return 999
    return int(t)

def f(y, t, arg1,arg2, u):
    return [y[1], arg1*u[g(t)]-arg1*y[0]-arg2*y[1]]

def getV(time,u):
    y0 = np.array([0,0],dtype=np.float32)
    V =  odeint(f, y0, time, args=(kappa, eta,u))
    return np.array(V[:,0],dtype=np.float32)

if __name__ == '__main__':
    time = np.arange(1000)
    U = getU(time)
    V = getV(time,U)
    print(U.shape)
    print(V.shape)
    fig = plt.figure()
    plt.plot(time,U,c="r")
    plt.plot(time,V,c="b")
    plt.show()
    fig.savefig("trueSet.png")