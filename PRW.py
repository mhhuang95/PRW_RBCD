#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

try:
    import cupy as cp
except:
    pass

class ProjectedRobustWasserstein:
    
    def __init__(self, X, Y, a, b, algo, k):
        """
        X    : (number_points_1, dimension) matrix of atoms for the first measure
        Y    : (number_points_2, dimension) matrix of atoms for the second measure
        a    : (number_points_1,) vector of weights for the first measure
        b    : (number_points_2,) vector of weights for the second measure
        algo : algorithm to compute the SRW distance (instance of class 'ProjectedGradientAscent' or 'FrankWolfe')
        k    : dimension parameter (can be of type 'int', 'list' or 'set' in order to compute SRW for several paremeters 'k').
        """
        
        # Check shapes
        d = X.shape[1]
        n = X.shape[0]
        m = Y.shape[0]
        assert d == Y.shape[1]
        assert n == a.shape[0]
        assert m == b.shape[0]
        
        if isinstance(k, int):
            assert k <= d
            assert k == int(k)
            assert 1 <= k
        elif isinstance(k, list) or isinstance(k, set):
            assert len(k) > 0
            k = list(set(k))
            k.sort(reverse=True)
            assert k[0] <= d
            assert k[-1] >= 1
            for l in k:
                assert l == int(l)
        else:
            raise TypeError("Parameter 'k' should be of type 'int' or 'list' or 'set'.")
        
        # Measures
        if algo.use_gpu:
            self.X = cp.asarray(X)
            self.Y = cp.asarray(Y)
            self.a = cp.asarray(a)
            self.b = cp.asarray(b)
        else:
            self.X = X
            self.Y = Y
            self.a = a
            self.b = b
        self.d = d
        
        # Algorithm
        self.algo = algo
        self.k = k
        
        self.pi = None
        self.values = None
        self.U = None
        self.running_time = None
        self.iter = None
        
        
    def InitialStiefel(self, d, k):
        U = np.random.randn(d, k)
        q, r = np.linalg.qr(U)
        return q
    

    def run(self, alg,tau, U_init=None):
        """Run algorithm algo on the data."""
        
        if U_init is None:
            U0 = self.InitialStiefel(self.d, self.k)
        else:
            U0 = U_init
        self.algo.tau = tau
        if alg == 'RBCD':
            self.pi, self.U, self.running_time, self.values, self.iter  = self.algo.run_RBCD(self.a, self.b, self.X, self.Y, self.k, U0)
        elif alg == 'RABCD':
            self.pi, self.U, self.running_time, self.values, self.iter  = self.algo.run_RABCD(self.a, self.b, self.X, self.Y, self.k, U0)
        elif alg == 'RGAS':
            self.pi, self.U, self.running_time, self.values, self.iter  = self.algo.run_RGAS(self.a, self.b, self.X, self.Y, self.k, U0)
        elif alg == 'RAGAS':
            self.pi, self.U, self.running_time, self.values, self.iter  = self.algo.run_RAGAS(self.a, self.b, self.X, self.Y, self.k, U0)


    def get_pi(self):
        return self.pi
    
    def get_Omega(self):
        return self.U.dot(self.U.T)
    
    def get_value(self):
        """Return the PRW distance."""    
        return self.values
     
    def get_time(self):
        """Return the running time."""    
        return self.running_time
    
    def get_iter(self):
        """Return the iteration number."""    
        return self.iter
               

    def plot_values(self, real_value=None):
        """Plot values if computed for several dimension parameters 'k'."""
        assert not isinstance(self.k, int)
        values = self.get_value()
        plt.plot(values.keys(), values.values(), lw=4)
        if real_value is not None:
            plt.plot(values.keys(), len(values)*[real_value])
        plt.grid(ls=':')
        plt.xticks(np.sort(list(values.keys())))
        plt.xlabel('Dimension parameter $k$', fontsize=25)
        plt.show()

    
    def plot_transport_plan(self, l=None, path=None):
        """Plot the transport plan."""
        isnotdict = False
        if isinstance(self.k, int):
            isnotdict = True
        if isinstance(self.k, int) and l is None:
            l = self.k
        elif isinstance(self.k, int) and l != self.k:
            raise ValueError("Argument 'l' should match class attribute 'k'.")
        elif l is None and isinstance(self.k, list):
            raise ValueError("When class attribute 'k' is a list, argument 'l' should be specified.")
        elif isinstance(self.k, list) and l not in self.k:
            raise ValueError("When class attribute 'k' is a list, argument 'l' should be in the list 'k'.")
        
        for i in range(self.X.shape[0]):
            for j in range(self.Y.shape[0]):
                if isnotdict and self.pi[i,j] > 0.:
                    plt.plot([self.X[i,0], self.Y[j,0]], [self.X[i,1], self.Y[j,1]], c='k', lw=30*self.pi[i,j])
                elif not isnotdict and self.pi[l][i,j] > 0.:
                    plt.plot([self.X[i,0], self.Y[j,0]], [self.X[i,1], self.Y[j,1]], c='k', lw=30*self.pi[l][i,j])
        plt.scatter(self.X[:,0], self.X[:,1], s=self.X.shape[0]*20*self.a, c='r', zorder=10, alpha=0.7)
        plt.scatter(self.Y[:,0], self.Y[:,1], s=self.Y.shape[0]*20*self.b, c='b', zorder=10, alpha=0.7)
        plt.title('Optimal PRW transport plan ('+str(self.algo)+',n=100)', fontsize=10)
        plt.axis('equal')
        if path is not None:
            plt.savefig(path)
        plt.show()
