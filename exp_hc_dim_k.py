#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pickle

from PRW import ProjectedRobustWasserstein
from Optimization.RiemannianBCD import RiemannianBlockCoordinateDescent
from Optimization.RiemannianGAS import RiemannianGradientAscentSinkhorn


def T(x,d,dim=2):
    assert dim <= d
    assert dim >= 1
    assert dim == int(dim)
    return x + 2*np.sign(x)*np.array(dim*[1]+(d-dim)*[0])

def fragmented_hypercube(n,d,dim):
    assert dim <= d
    assert dim >= 1
    assert dim == int(dim)
    
    a = (1./n) * np.ones(n)
    b = (1./n) * np.ones(n)

    # First measure : uniform on the hypercube
    X = np.random.uniform(-1, 1, size=(n,d))

    # Second measure : fragmentation
    Y = T(X, d, dim)
    
    return a,b,X,Y

def InitialStiefel(d, k):
    Z = np.random.randn(d, k)
    q, r = np.linalg.qr(Z)
    return q

def main():

    n = 100 # Number of points for each measure
    d = 30 # Total dimension
    maxK = 30 # Compute SRW for all parameters 'k'
    nb_exp = 50 # Do 100 experiments
    kstars = [2, 4, 7, 10] # Plot for 'true' dimension k* = 2, 4, 7, 10
    
    eta = 0.2
    tau = 0.005
    verb = True

    values = np.zeros((2, len(kstars), maxK, nb_exp))
    
    if 1==1:
        for t in range(nb_exp):
            for kstar_index in range(len(kstars)):
                kstar = kstars[kstar_index]
                for kdim in range(1, maxK+1):

                    a,b,X,Y = fragmented_hypercube(n,d,kstar)

                    algo = RiemannianBlockCoordinateDescent(eta=eta, tau=None, max_iter=3000, threshold=0.1, verbose=verb)
                    PRW = ProjectedRobustWasserstein(X, Y, a, b, algo, kdim)
                    PRW.run('RBCD', tau)
                    values[0, kstar_index, kdim - 1, t] = np.abs(PRW.get_value())

                    algo1 = RiemannianGradientAscentSinkhorn(eta=eta, tau=None, max_iter=3000, threshold=0.1,
                                                     sink_threshold=1e-4, verbose=verb)
                    PRW1 = ProjectedRobustWasserstein(X, Y, a, b, algo1, kdim)
                    PRW1.run('RGAS', tau/eta)
                    values[1, kstar_index, kdim - 1, t] = np.abs(PRW1.get_value())
                    
        with open('./results/exp1_hc_dim_k.pkl', 'wb') as f:
            pickle.dump(values, f)
    else:
        with open('./results/exp1_hypercube_dim_k.pkl', 'rb') as f:
            values = pickle.load(f)

    colors = [['b', 'orange', 'g', 'r'], ['c', 'm', 'y', 'purple']]
    plt.figure(figsize=(20, 8))

    Xs = list(range(1, maxK + 1))
    line_styles = ['-', '--']
    captions = ['RBCD', 'RGAS']
    for t in range(2):
        for i, kstar in enumerate(kstars):
            values_mean = np.mean(values[t, i, :, :], axis=1)
            values_min = np.min(values[t, i, :, :], axis=1)
            values_max = np.max(values[t, i, :, :], axis=1)

            mean, = plt.plot(Xs, values_mean, ls=line_styles[t],
                             c=colors[t][i], lw=4, ms=20,
                             label='$k^*=%d$, %s' % (kstar,captions[t]))
            col = mean.get_color()
            plt.fill_between(Xs, values_min, values_max, facecolor=col, alpha=0.15)

    for i in range(len(kstars)):
        ks = kstars[i]
        vm1 = np.mean(values[0, i, ks, :], axis=0)
        vm2 = np.mean(values[1, i, ks, :], axis=0)
        print(vm1,vm2)
        tt = max(vm1,vm2)
        plt.plot([ks, ks], [0, tt], color=colors[0][i], linestyle='--')


    plt.xlabel('Dimension k', fontsize=25)
    plt.ylabel('PRW values', fontsize=25)
    plt.ylabel('$P_k^2(\hat\mu, \hat\\nu)$', fontsize=25)
    plt.xticks(Xs, fontsize=20)
    plt.yticks(np.arange(10, 70+1, 10), fontsize=20)
    plt.legend(loc='best', fontsize=18, ncol=2)
    plt.ylim(0)
    plt.title('$P_k^2(\hat\mu, \hat\\nu)$ depending on dimension k', fontsize=30)
    plt.minorticks_on()
    plt.grid(ls=':')
    plt.savefig('figs/exp1_dim_k.png')
    
if __name__ == "__main__":
    main()