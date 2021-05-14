#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from SRW import SubspaceRobustWasserstein
from Optimization.RiemannianBCD import RiemannianBlockCoordinateDescent
from Optimization.RiemannianGAS import RiemannianGradientAscentSinkhorn
from Optimization.SRW.projectedascent import ProjectedGradientAscent

def InitialStiefel(d, k):
        
    U = np.random.randn(d, k)
    q, r = np.linalg.qr(U)
    return q
    

if __name__ == "__main__":
    
    d = 20 # Dimension of the Gaussians
    n = 100 # Number of points for the empirical distributions
    k_star = 5  # Order of the Wishart distribution, i.e. dimension of the support of the Gaussians
    
    # Equal weights
    a = (1./n) * np.ones(n)
    b = (1./n) * np.ones(n)

    # Zero means
    mean_1 = np.zeros(d)
    mean_2 = np.zeros(d)

    # Covariances from Wishart
    cov_1 = np.random.randn(d,k_star)
    cov_1 = cov_1.dot(cov_1.T)
    cov_2 = np.random.randn(d,k_star)
    cov_2 = cov_2.dot(cov_2.T)
    
    k = 10

    # Empirical measures with n points
    X = np.random.multivariate_normal(mean_1, cov_1, size=n)
    Y = np.random.multivariate_normal(mean_2, cov_2, size=n)

#     # Add noise
#     noise_level = 1
#     X = X + noise_level*np.random.randn(n,d)
#     Y = Y + noise_level*np.random.randn(n,d)
    
    U0 = InitialStiefel(d, k)

    gamma = 0.1
    eta = 10

    params = {'eta': eta, 'tau': gamma / eta, 'max_iter': 4000, 'threshold': 0.01, 'verbose': True}
    algo1 = RiemannianGradientAscentSinkhorn(**params)
    algo1.run_RGAS(a, b, X, Y, k, U0)
    algo1.run_RAGAS(a, b, X, Y, k, U0)


    params = {'eta':eta, 'tau':gamma, 'max_iter':4000, 'threshold':0.01, 'verbose':True}
    algo3 = RiemannianBlockCoordinateDescent(**params)
    algo3.run_RBCD(a, b, X, Y,  k, U0)
    algo3.run_RABCD(a, b, X, Y,  k, U0)


# Compute Wasserstein
    algo2 = ProjectedGradientAscent(reg=20, step_size_0=gamma, max_iter=1, max_iter_sinkhorn=50, threshold=0.001, threshold_sinkhorn=1e-04, use_gpu=False)
    W_ = SubspaceRobustWasserstein(X, Y, a, b, algo2, k=d)
    W_.run()
    print(W_.get_value())



