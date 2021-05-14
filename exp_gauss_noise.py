#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pickle

from PRW import ProjectedRobustWasserstein
from Optimization.RiemannianBCD import RiemannianBlockCoordinateDescent

def main():
    
    noise_level = 1
    d = 20 # Total dimension
    n = 100 # Number of points for each measure
    l = 5 # Dimension of Wishart
    nb_exp = 100 # Number of experiments
    k = list(range(1,d+1)) # Compute SRW for all dimension parameter k

    # Save the values
    no_noise = np.zeros((nb_exp,d))
    noise = np.zeros((nb_exp,d))
    
    eta = 1
    tau = 0.0005
    verb = True
    
    if 1 == 1:
        for t in range(nb_exp): # Fore each experiment
            print(t)

            a = (1./n) * np.ones(n)
            b = (1./n) * np.ones(n)

            mean_1 = 0.*np.random.randn(d)
            mean_2 = 0.*np.random.randn(d)

            cov_1 = np.random.randn(d,l)
            cov_1 = cov_1.dot(cov_1.T)
            cov_2 = np.random.randn(d,l)
            cov_2 = cov_2.dot(cov_2.T)

            # Draw measures
            X = np.random.multivariate_normal(mean_1, cov_1, size=n)
            Y = np.random.multivariate_normal(mean_2, cov_2, size=n)

            # Add noise
            Xe = X + noise_level*np.random.randn(n,d)
            Ye = Y + noise_level*np.random.randn(n,d)

            vals = []
            for k in range(1, d + 1):
                algo = RiemannianBlockCoordinateDescent(eta=eta, tau=tau , max_iter = 5000, threshold=0.1, verbose=verb)
                PRW = ProjectedRobustWasserstein(X, Y, a, b, algo , k)
                PRW.run('RBCD', tau=tau )
                vals.append(PRW.get_value())
            no_noise[t,:] = np.sort(vals)

            vals = []
            for k in range(1, d + 1):
                algoe = RiemannianBlockCoordinateDescent(eta=eta, tau=tau , max_iter = 5000, threshold=0.1, verbose=verb)
                PRWe = ProjectedRobustWasserstein(Xe, Ye, a, b, algoe, k)
                PRWe.run('RBCD', tau=0.0005)
                vals.append(PRWe.get_value())
            noise[t,:] = np.sort(vals)

            no_noise[t,:] /= no_noise[t,(d-1)]
            noise[t,:] /= noise[t,(d-1)]
        with open('./results/exp2_noise_12.pkl', 'wb') as f:
            pickle.dump([no_noise, noise], f)
        
        
    else:

        with open('./results/exp2_noise_12.pkl', 'rb') as f:
            no_noise, noise = pickle.load(f)
            
    captions = ['PRW']
    plt.figure(figsize=(12, 8))

    no_noise_t = no_noise[:, :]
    no_noise_mean = np.mean(no_noise_t, axis=0)
    no_noise_min = np.min(no_noise_t, axis=0)
    no_noise_10 = np.percentile(no_noise_t, 10, axis=0)
    no_noise_25 = np.percentile(no_noise_t, 25, axis=0)
    no_noise_75 = np.percentile(no_noise_t, 75, axis=0)
    no_noise_90 = np.percentile(no_noise_t, 90, axis=0)
    no_noise_max = np.max(no_noise_t, axis=0)

    noise_t = noise[:, :]
    noise_mean = np.mean(noise_t, axis=0)
    noise_min = np.min(noise_t, axis=0)
    noise_10 = np.percentile(noise_t, 10, axis=0)
    noise_25 = np.percentile(noise_t, 25, axis=0)
    noise_75 = np.percentile(noise_t, 75, axis=0)
    noise_90 = np.percentile(noise_t, 90, axis=0)
    noise_max = np.max(noise_t, axis=0)

    plotnonoise, = plt.plot(range(d), no_noise_mean, 'C1', label='Without Noise', lw=6)
    col_nonoise = plotnonoise.get_color()
    plt.fill_between(range(d), no_noise_25, no_noise_75, facecolor=col_nonoise, alpha=0.3)
    plt.fill_between(range(d), no_noise_10, no_noise_90, facecolor=col_nonoise, alpha=0.2)
    plt.fill_between(range(d), no_noise_min, no_noise_max, facecolor=col_nonoise, alpha=0.15)

    plotnoise, = plt.plot(range(d), noise_mean, 'C2', label='With Noise', lw=6)
    col_noise = plotnoise.get_color()
    plt.fill_between(range(d), noise_25, noise_75, facecolor=col_noise, alpha=0.3)
    plt.fill_between(range(d), noise_10, noise_90, facecolor=col_noise, alpha=0.2)
    plt.fill_between(range(d), noise_min, noise_max, facecolor=col_noise, alpha=0.15)

    plt.xlabel('Dimension', fontsize=25)
    plt.ylabel('Normalized %s value' % (captions[0]), fontsize=25)
    plt.legend(loc='best', fontsize=20)

    plt.yticks(fontsize=20)
    plt.xticks(range(d), range(1, d + 1), fontsize=20)
    plt.ylim(0.1)

    plt.legend(loc='best', fontsize=25)
    plt.title('%s distance with different dimensions' % (captions[0],), fontsize=30)
    plt.grid(ls=':')
    # plt.savefig('figs/exp2_noise_%d.png' % (0,))
    plt.show()
    plt.close()
    plt.clf()

    
if __name__ == "__main__":
    main()