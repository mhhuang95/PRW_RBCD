#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle

from PRW import ProjectedRobustWasserstein
from SRW import SubspaceRobustWasserstein
from Optimization.RiemannianBCD import RiemannianBlockCoordinateDescent
from Optimization.RiemannianGAS import RiemannianGradientAscentSinkhorn
from Optimization.SRW.projectedascent import ProjectedGradientAscent


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
    Y = T(np.random.uniform(-1, 1, size=(n,d)), d, dim)
    
    return a,b,X,Y

def InitialStiefel(d, k):
    Z = np.random.randn(d, k)
    q, r = np.linalg.qr(Z)
    return q

def main():
    
    d = 20 # Total dimension
    k = 2 # k* = 2 and compute SRW with k = 2
    nb_exp = 100 # Do 500 experiments
    ns = [25, 50, 100, 250, 500, 1000] # Compute SRW between measures with 'n' points for 'n' in 'ns'

    values = np.zeros((3, len(ns), nb_exp))
    values_subspace = np.zeros((3, len(ns), nb_exp))

    proj = np.zeros((d,d)) # Real optimal subspace
    proj[0,0] = 1
    proj[1,1] = 1
    
    eta= 0.2
    tau = 0.001
    verb = True
    
    if 1 == 1:
    
        for indn in range(len(ns)):
            n = ns[indn]
            # Sample nb_exp times
            for t in range(nb_exp):
                a,b,X,Y = fragmented_hypercube(n,d,dim=2)
                
                U0 = np.zeros((d, k))
                U0[:k, :] = np.eye(k)

                algo = RiemannianBlockCoordinateDescent(eta=eta, tau=None, max_iter=3000, threshold=0.1, verbose=verb)
                PRW = ProjectedRobustWasserstein(X, Y, a, b, algo, k)
                PRW.run('RBCD', tau, U0)
                values[0, indn, t] = np.abs(8 - PRW.get_value())
                values_subspace[0, indn, t] = np.linalg.norm(PRW.get_Omega() - proj)

                algo1 = RiemannianGradientAscentSinkhorn(eta=eta, tau=None, max_iter=3000, threshold=0.1,
                                                 sink_threshold=1e-4, verbose=verb)
                PRW1 = ProjectedRobustWasserstein(X, Y, a, b, algo1, k)
                PRW1.run('RGAS', tau/eta, U0)
                values[1, indn, t] = np.abs(8 - PRW1.get_value())
                values_subspace[1, indn, t] = np.linalg.norm(PRW1.get_Omega() - proj)

                # Compute Wasserstein
                algo2 = ProjectedGradientAscent(reg=eta, step_size_0=tau, max_iter=1, max_iter_sinkhorn=50,
                                                threshold=0.001, threshold_sinkhorn=1e-04, use_gpu=False)
                W_ = SubspaceRobustWasserstein(X, Y, a, b, algo2, k=d)
                W_.run()
                values[2, indn, t] = np.abs(8 - W_.get_value())
                values_subspace[2, indn, t] = np.linalg.norm(W_.get_Omega() - proj)
                print(W_.get_value())

        with open('./results/exp1_hypercube_value.pkl', 'wb') as f:
            pickle.dump([values, values_subspace], f)

    else:
        with open('./results/exp1_hypercube_value.pkl', 'rb') as f:
            values, values_subspace = pickle.load(f)


        print('n =',n,'/', np.mean(values[indn,:]), '/', np.mean(values1[indn,:]))


    captions = ['PRW (RBCD)', 'PRW (RGAS)', 'W']

    line = ['o-', 'o--', '-']
    plt.figure(figsize=(12, 8))
    for t in range(3):
        values_mean = np.mean(values[t,:,:], axis=1)
        values_min = np.min(values[t,:,:], axis=1)
        values_10 = np.percentile(values[t,:,:], 10, axis=1)
        values_25 = np.percentile(values[t,:,:], 25, axis=1)
        values_75 = np.percentile(values[t,:,:], 75, axis=1)
        values_90 = np.percentile(values[t,:,:], 90, axis=1)
        values_max = np.max(values[t,:,:], axis=1)

        mean, = plt.semilogy(ns, values_mean, line[t], lw=4, ms=11,
                           label=captions[t])
        col = mean.get_color()
        plt.fill_between(ns, values_25, values_75, facecolor=col, alpha=0.3)
        plt.fill_between(ns, values_10, values_90, facecolor=col, alpha=0.2)

    plt.xlabel('Number of points n', fontsize=25)
    plt.ylabel('MEE', fontsize=25)
    plt.legend(loc='best', fontsize=25)
    plt.title('Mean estimation error', fontsize=30)

    plt.xticks(ns, fontsize=20)
    plt.yticks(np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]), fontsize=20)
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    plt.grid(ls=':')
    plt.savefig('figs/exp1_hypercube_value_1.png')
    plt.show()
    plt.close()
    plt.clf()

    plt.figure(figsize=(12, 8))
    for t in range(3):
        values_subspace_mean = np.mean(values_subspace[t,:,:], axis=1)
        values_subspace_min = np.min(values_subspace[t,:,:], axis=1)
        values_subspace_10 = np.percentile(values_subspace[t,:,:], 10, axis=1)
        values_subspace_25 = np.percentile(values_subspace[t,:,:], 25, axis=1)
        values_subspace_75 = np.percentile(values_subspace[t,:,:], 75, axis=1)
        values_subspace_90 = np.percentile(values_subspace[t,:,:], 90, axis=1)
        values_subspace_max = np.max(values_subspace[t,:,:], axis=1)

        mean, = plt.loglog(ns, values_subspace_mean, line[t], lw=4, ms=11,
                           label=captions[t])
        col = mean.get_color()
        plt.fill_between(ns, values_subspace_25, values_subspace_75, facecolor=col, alpha=0.3)
        plt.fill_between(ns, values_subspace_10, values_subspace_90, facecolor=col, alpha=0.2)
        plt.fill_between(ns, values_subspace_min, values_subspace_max, facecolor=col, alpha=0.15)

    plt.xlabel('Number of points n', fontsize=25)
    plt.ylabel('$||\Omega^* - \widehat\Omega||_F$', fontsize=25)
    plt.legend(loc='best', fontsize=25)
    plt.title('Mean subspace estimation error', fontsize=30)
    plt.xticks(ns, fontsize=20)
    plt.yticks(np.array(range(1, 8)) / 10, fontsize=20)
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    plt.grid(ls=':')
    plt.savefig('figs/exp1_hypercube_value_2.png')


    
if __name__ == "__main__":
    main()
