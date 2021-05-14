#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from PRW import ProjectedRobustWasserstein
from SRW import SubspaceRobustWasserstein
from Optimization.RiemannianBCD import RiemannianBlockCoordinateDescent
from Optimization.RiemannianGAS import RiemannianGradientAscentSinkhorn
from Optimization.SRW.projectedascent import ProjectedGradientAscent


def main():
    d = 20 # Total dimension
    n = 100 # Number of points in each measure
    k = 5 # Dimension of the Wishart (i.e. of support of the measures)
    nb_exp = 100 # Number of experiments to run
    reg = 0. # No regularization
    # max_iter = 1000 # Maximum number of iterations (the bigger the more precise)
    # thr = 1e-5 # Stopping threshold (not attained here since we are in unregularized SRW)

    a = (1./n) * np.ones(n)
    b = (1./n) * np.ones(n)

    mean_1 = np.zeros(d)
    mean_2 = np.zeros(d)

    # Noise levels to test
    ind = [0., 0.01, 0.1, 1, 2, 4, 7, 10]

    PRW = np.zeros((nb_exp, len(ind)))
    PRW1 = np.zeros((nb_exp, len(ind)))
    W = np.zeros((nb_exp, len(ind)))


    for t in range(nb_exp):
        print(t)
        cov_1 = np.random.randn(d,k)
        cov_1 = cov_1.dot(cov_1.T)
        cov_2 = np.random.randn(d,k)
        cov_2 = cov_2.dot(cov_2.T)

        # Draw the measures
        X = np.random.multivariate_normal(mean_1, cov_1, size=n)
        Y = np.random.multivariate_normal(mean_2, cov_2, size=n)
        
        verb = True

        lst_rsw = []
        lst_rsw1 = []
        lst_w = []
        for epsilon in ind:
            # Add noise of level epsilon
            noiseX = np.random.randn(n,d)
            noiseY = np.random.randn(n,d)
            Xe = X + epsilon*noiseX
            Ye = Y + epsilon*noiseY

            if  epsilon < 4:
                eta = 2 
                stepsize = 0.01
                thre = 0.1
            else:
                eta = 10
                stepsize = 0.002
                thre = 0.1

            algo = RiemannianBlockCoordinateDescent(eta=eta, tau=stepsize, max_iter = 5000, threshold=thre, verbose=verb)
            PRW_ = ProjectedRobustWasserstein(Xe, Ye, a, b, algo, k=10)
            PRW_.run('RBCD', tau=stepsize)


            algo1 = RiemannianGradientAscentSinkhorn(eta=eta, tau= stepsize/eta, max_iter = 5000, sink_threshold=1e-4, threshold=thre, verbose=verb)
            PRW1_ = ProjectedRobustWasserstein(Xe, Ye, a, b, algo1, k=10)
            PRW1_.run('RGAS', tau=stepsize/eta)

            # Choice of step size
            ones = np.ones((n, n))
            C = np.diag(np.diag(Xe.dot(Xe.T))).dot(ones) + ones.dot(np.diag(np.diag(Ye.dot(Ye.T)))) - 2 * Xe.dot(Ye.T)
            step_size_0 = 1. / np.max(C)

            # Compute Wasserstein
            algo = ProjectedGradientAscent(reg=reg, step_size_0=step_size_0, max_iter=1, max_iter_sinkhorn=50, threshold=0.001, threshold_sinkhorn=1e-04, use_gpu=False)
            W_ = SubspaceRobustWasserstein(Xe, Ye, a, b, algo, k=d)
            W_.run()
            print(W_.get_value())

            lst_rsw.append(PRW_.get_value())
            lst_rsw1.append(PRW1_.get_value())
            lst_w.append(W_.get_value())

        PRW[t,:] = np.array(lst_rsw)
        PRW1[t,:] = np.array(lst_rsw1)
        W[t,:] = np.array(lst_w)

    # Relative change
    PRW_percent = np.abs(PRW-np.array([PRW[:,0],]*len(ind)).transpose())/np.array([PRW[:,0],]*len(ind)).transpose()
    PRW1_percent = np.abs(PRW1-np.array([PRW1[:,0],]*len(ind)).transpose())/np.array([PRW1[:,0],]*len(ind)).transpose()
    W_percent = np.abs(W-np.array([W[:,0],]*len(ind)).transpose())/np.array([W[:,0],]*len(ind)).transpose()

    PRW_percent = PRW_percent[:,1:]
    PRW1_percent = PRW1_percent[:,1:]
    W_percent = W_percent[:,1:]

    PRW_mean = np.mean(PRW_percent, axis=0)
    PRW_min = np.min(PRW_percent, axis=0)
    PRW_10 = np.percentile(PRW_percent, 10, axis=0)
    PRW_25 = np.percentile(PRW_percent, 25, axis=0)
    PRW_75 = np.percentile(PRW_percent, 75, axis=0)
    PRW_90 = np.percentile(PRW_percent, 90, axis=0)
    PRW_max = np.max(PRW_percent, axis=0)

    PRW1_mean = np.mean(PRW1_percent, axis=0)
    PRW1_min = np.min(PRW1_percent, axis=0)
    PRW1_10 = np.percentile(PRW1_percent, 10, axis=0)
    PRW1_25 = np.percentile(PRW1_percent, 25, axis=0)
    PRW1_75 = np.percentile(PRW1_percent, 75, axis=0)
    PRW1_90 = np.percentile(PRW1_percent, 90, axis=0)
    PRW1_max = np.max(PRW1_percent, axis=0)

    W_mean = np.mean(W_percent, axis=0)
    W_min = np.min(W_percent, axis=0)
    W_10 = np.percentile(W_percent, 10, axis=0)
    W_25 = np.percentile(W_percent, 25, axis=0)
    W_75 = np.percentile(W_percent, 75, axis=0)
    W_90 = np.percentile(W_percent, 90, axis=0)
    W_max = np.max(W_percent, axis=0)


    # PLOT
    import matplotlib.ticker as ticker
    plt.figure(figsize=(12,8))

    plotW, = plt.loglog(ind[1:], W_mean, 'o-', label='Wasserstein', lw=5, ms=10)
    col_W = plotW.get_color()
    plt.fill_between(ind[1:], W_25, W_75, facecolor=col_W, alpha=0.3)
    plt.fill_between(ind[1:], W_10, W_90, facecolor=col_W, alpha=0.2)

    plotPRW, = plt.loglog(ind[1:], PRW_mean, 'o-', label='RBCD', lw=5, ms=10)
    col_PRW = plotPRW.get_color()
    plt.fill_between(ind[1:], PRW_25, PRW_75, facecolor=col_PRW, alpha=0.3)
    plt.fill_between(ind[1:], PRW_10, PRW_90, facecolor=col_PRW, alpha=0.2)

    plotPRW1, = plt.loglog(ind[1:], PRW1_mean, 'o--', label='RGAS', lw=5, ms=10)
    col_PRW1 = plotPRW1.get_color()
    plt.fill_between(ind[1:], PRW1_25, PRW1_75, facecolor=col_PRW1, alpha=0.3)
    plt.fill_between(ind[1:], PRW1_10, PRW1_90, facecolor=col_PRW1, alpha=0.2)


    plt.xlabel('Noise level (log scale)', fontsize=25)
    plt.ylabel('Relative error (log scale)', fontsize=25)

    plt.yticks(fontsize=20)
    plt.xticks(ind[1:], fontsize=20)
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2g'))

    plt.legend(loc=2, fontsize=18)
    plt.grid(ls=':')
    plt.savefig('figs/exp2_noise_level.png')
    
    
if __name__ == "__main__":
    main()