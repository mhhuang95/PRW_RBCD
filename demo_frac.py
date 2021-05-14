#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

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
    U = np.random.randn(d, k)
    q, r = np.linalg.qr(U)
    return q

if __name__ == "__main__":
    
    d = 100 # Dimension of the Gaussians
    n = 100 # Number of points for the empirical distributions
    k = 3  # Order of the Wishart distribution, i.e. dimension of the support of the Gaussians
    dim = 5
    a,b,X,Y = fragmented_hypercube(n,d,dim)
    
    n_ep = 1
    
    time_GRAS = np.zeros(n_ep)
    time_AGRAS = np.zeros(n_ep)
    time_GRDD = np.zeros(n_ep)
    time_AGRDD = np.zeros(n_ep)

    for i in range(n_ep):
        
        U0 = InitialStiefel(d, k)
        a,b,X,Y = fragmented_hypercube(n,d,dim)
        
        gamma = 0.001
        eta = 0.2
    
        params = {'eta':eta, 'tau':gamma/eta, 'max_iter':2000, 'threshold':0.1, 'verbose':True}
        algo1 = RiemannianGradientAscentSinkhorn(**params)
        algo1.run_RGAS(a, b, X, Y, k, U0)
        algo1.run_RAGAS(a, b, X, Y, k, U0)


        params = {'eta':eta, 'tau':gamma, 'max_iter':2000, 'threshold':0.1, 'verbose':True}
        algo2 = RiemannianBlockCoordinateDescent(**params)
        algo2.run_RBCD(a, b, X, Y, k, U0)
        algo2.run_RABCD(a, b, X, Y, k, U0)
    

#         Compute Wasserstein
        algo3 = ProjectedGradientAscent(reg=eta, step_size_0=gamma, max_iter=1, max_iter_sinkhorn=50, threshold=0.001, threshold_sinkhorn=1e-04, use_gpu=False)
        W_ = SubspaceRobustWasserstein(X, Y, a, b, algo3, k=d)
        W_.run()
        print(W_.get_value())



