# -*- coding: utf-8 -*-

import numpy as np
import time

try:
    import cupy as cp
except:
    pass


# Riemannian Block Coordinate Descent:

class RiemannianBlockCoordinateDescent:
    
    def __init__(self, eta, tau, max_iter, threshold, verbose=False, use_gpu=False):
        """
        eta : Entropic regularization strength
        tau : Initial step size for ProjectedGradientAscent
        max_iter : Maximum number of iterations to be run
        threshold : Stopping threshold (stops when precision 'threshold' is attained or 'max_iter' iterations are run)
        use_gpu : 'True' to use GPU, 'False' otherwise
        verbose : 'True' to print additional messages, 'False' otherwise
        """
        
        assert eta >= 0
        if tau is not None:
            assert tau > 0
        assert isinstance(max_iter, int)
        assert max_iter > 0
        assert threshold > 0
        assert isinstance(verbose, bool)
        
        self.eta = eta
        self.tau = tau
        self.max_iter = max_iter
        self.threshold = threshold
        self.verbose = verbose
        self.use_gpu = use_gpu
        
    @staticmethod    
    def InitialStiefel(d, k):
        U = np.random.randn(d, k)
        q, r = np.linalg.qr(U)
        return q
    
    @staticmethod
    def StiefelRetraction(U, G):
        q, r = np.linalg.qr(U + G)
        return q
    
    @staticmethod
    def StiefelGradientProj(G, U):
        # project G onto the tangent space of Stiefel manifold at Z
        temp = G.T.dot(U)
        PG = G - U.dot(temp + temp.T) / 2
        return PG
    
    def Vpi(self, X, Y,a,b, pi):
        #Return the second order matrix of the displacements: sum_ij { (OT_plan)_ij (X_i-Y_j)(X_i-Y_j)^T }.
        A = X.T.dot(pi).dot(Y)
#         return X.T.dot(np.diag(a)).dot(X) + Y.T.dot(np.diag(b)).dot(Y) - A - A.T
        return X.T.dot(np.diag(a)).dot(X) + Y.T.dot(np.diag(np.sum(pi, 0))).dot(Y) - A - A.T
    
    def run_RBCD(self, a, b, X, Y, k, U):
        #Riemannian Block Coordinate Descent
        
        #initialization
        n, d = X.shape
        m, d = Y.shape
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        u = np.ones(n) / n
        v = np.ones(m) / m
                
        eta = self.eta
        step_size = self.tau
        ones = np.ones((n,m))
        
        iter = 0
        UUT = U.dot(U.T)
        M = np.diag(np.diag(X.dot(UUT.dot(X.T)))).dot(ones) + ones.dot(np.diag(np.diag(Y.dot(UUT.dot(Y.T))))) - 2*X.dot(UUT.dot(Y.T))
        
        # Next 3 lines equivalent to A= np.exp(-M/reg), but faster to compute
        A = np.empty(M.shape, dtype=M.dtype)
        np.divide(M, -eta, out=A)
        np.exp(A, out=A)
        
        Ap = (1 / a).reshape(-1, 1) * A
        AtransposeU = np.dot(A.T, u)
        v = np.divide(b, AtransposeU)
        u = 1. / np.dot(Ap, v)
        
        pi = u.reshape((-1, 1)) * A * v.reshape((1, -1))
        V = self.Vpi(X, Y, a, b, pi)
        
        G = 2/eta  * V.dot(U)  # Euclidean gradient
        xi = self.StiefelGradientProj(G, U)  # Riemannian gradient
        grad_norm = np.linalg.norm(xi)

        if k == d:
            grad_norm = 1000
            
        time_iter = np.zeros(self.max_iter + 1)
        grad_iter = np.zeros(self.max_iter + 1)
        time_iter[0] = 0
        grad_iter[0] = np.linalg.norm(xi)
        
        while eta*grad_norm > self.threshold and iter < self.max_iter:
            
            tic = time.perf_counter()
            
            UUT = U.dot(U.T)
            M = np.diag(np.diag(X.dot(UUT.dot(X.T)))).dot(ones) + ones.dot(np.diag(np.diag(Y.dot(UUT.dot(Y.T))))) - 2*X.dot(UUT.dot(Y.T))
            
            #One step Sinkhorn
            A = np.empty(M.shape, dtype=M.dtype)
            np.divide(M, -eta, out=A)
            np.exp(A, out=A)
            
            Ap = (1 / a).reshape(-1, 1) * A
            AtransposeU = np.dot(A.T, u)
            v = np.divide(b, AtransposeU)
            u = 1. / np.dot(Ap, v)
            pi = u.reshape((-1, 1)) * A * v.reshape((1, -1))
                   
            V = self.Vpi(X, Y, a, b, pi)
            
            G = 2/eta * V.dot(U)
            xi = self.StiefelGradientProj(G, U)
            U = self.StiefelRetraction(U, step_size * xi)
            grad_norm = np.linalg.norm(xi)
            
            toc = time.perf_counter()
            time_iter[iter + 1] = time_iter[iter] + toc - tic
            
            grad_iter[iter + 1] = np.linalg.norm(xi)
            iter = iter + 1
        
        f_val = np.trace(U.T.dot(V.dot(U)))
        if self.verbose:
            print('RBCD:Iteration: ', iter, ' grad', eta*grad_norm , '\t Time: ', time_iter[iter], '\t fval: ', f_val )
  
        return pi, U, time_iter[iter], f_val, iter


    def run_RABCD(self, a, b, X, Y, k, U):
        #Riemannian Adaptive Block Coordinate Descent
        
        #initialization
        n, d = X.shape
        m, d = Y.shape
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        u = np.ones(n) / n
        v = np.ones(m) / m
        alpha = 1e-6
        beta = 0.8
        ones = np.ones((n,m))
        C = np.diag(np.diag(X.dot(X.T))).dot(ones) + ones.dot(np.diag(np.diag(Y.dot(Y.T)))) - 2 * X.dot(Y.T)
        p = np.zeros(d)
        q = np.zeros(k)
        p_hat = alpha * np.max(np.abs(C))**2 * np.ones(d)
        q_hat = alpha * np.max(np.abs(C))**2 * np.ones(k)  
        eta = self.eta
        step_size = self.tau 
        
          
        iter = 0
        UUT = U.dot(U.T)
        M = np.diag(np.diag(X.dot(UUT.dot(X.T)))).dot(ones) + ones.dot(np.diag(np.diag(Y.dot(UUT.dot(Y.T))))) - 2*X.dot(UUT.dot(Y.T))
        
        # Next 3 lines equivalent to A= np.exp(-M/reg), but faster to compute
        A = np.empty(M.shape, dtype=M.dtype)
        np.divide(M, -eta, out=A)
        np.exp(A, out=A)
        
        Ap = (1 / a).reshape(-1, 1) * A
        AtransposeU = np.dot(A.T, u)
        v = np.divide(b, AtransposeU)
        u = 1. / np.dot(Ap, v)
        pi = u.reshape((-1, 1)) * A * v.reshape((1, -1))
        
        V = self.Vpi(X, Y, a, b, pi)
        
        G = 2/eta * V.dot(U)
        xi = self.StiefelGradientProj(G, U)
        grad_norm = np.linalg.norm(xi)
           
        time_iter = np.zeros(self.max_iter + 1)
        grad_iter = np.zeros(self.max_iter + 1)
        time_iter[0] = 0
        grad_iter[0] = np.linalg.norm(xi)
        
        while eta * grad_norm > self.threshold and iter < self.max_iter:
            
            tic = time.perf_counter()
            
            UUT = U.dot(U.T)
            M = np.diag(np.diag(X.dot(UUT.dot(X.T)))).dot(ones) + ones.dot(np.diag(np.diag(Y.dot(UUT.dot(Y.T))))) - 2*X.dot(UUT.dot(Y.T))
            
            #One step Sinkhorn
            A = np.empty(M.shape, dtype=M.dtype)
            np.divide(M, -eta, out=A)
            np.exp(A, out=A)
            
            Ap = (1 / a).reshape(-1, 1) * A
            AtransposeU = np.dot(A.T, u)
            v = np.divide(b, AtransposeU)
            u = 1. / np.dot(Ap, v)
            pi = u.reshape((-1, 1)) * A * v.reshape((1, -1))
                  
            V = self.Vpi(X, Y, a, b, pi)
            
            G = - 2 * V.dot(U) 
            G_t = self.StiefelGradientProj(G, U)
            
            p = beta * p + (1-beta) * np.diag(G_t.dot(G_t.T))/k
            p_hat = np.maximum(p_hat, p)
            q = beta * q + (1-beta) * np.diag(G_t.T.dot(G_t))/d
            q_hat = np.maximum(q_hat, q)
            
            xi = self.StiefelGradientProj(np.diag(np.power(p_hat, -1/4)).dot(G_t).dot(np.diag(np.power(q_hat, -1/4))), U)
            
            U = self.StiefelRetraction(U, - step_size / eta *  xi)
            grad_norm = np.linalg.norm(G_t/eta)
            
            toc = time.perf_counter()
            time_iter[iter + 1] = time_iter[iter] + toc - tic
            grad_iter[iter + 1] = np.linalg.norm(xi)
            iter = iter + 1
        
        f_val = np.trace(U.T.dot(V.dot(U)))
        if self.verbose:
            print('RABCD: Iteration: ', iter, 'grad', eta * grad_norm , 'Time: ', time_iter[iter], 'fval: ', f_val )
        
        return pi, U, time_iter[iter], f_val, iter

