# -*- coding: utf-8 -*-#

import numpy as np
import matplotlib.pyplot as plt
import time

from Optimization.RiemannianBCD import RiemannianBlockCoordinateDescent
from Optimization.RiemannianGAS import RiemannianGradientAscentSinkhorn

from SRW import SubspaceRobustWasserstein
from PRW import ProjectedRobustWasserstein
from Optimization.SRW.frankwolfe import FrankWolfe
from Optimization.sinkhorn import sinkhorn_knopp


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
    U = np.random.randn(d, k)
    q, r = np.linalg.qr(U)
    return q
    

def plot_fix_n():
    
    ds = [25, 50, 100, 250, 500] # Dimensions
    nb_ds = len(ds)
    n = 100 # Number of points in the measures
    k = 2 # Dimension parameter
    max_iter = 2000 # Maximum number of iterations
    max_iter_sinkhorn = 1000 # Maximum number of iterations in Sinkhorn
    threshold = 0.1 # Stopping threshold
    threshold_sinkhorn = 1e-9 # Stopping threshold in Sinkhorn
    nb_exp = 100 # Number of experiments
    
    tau = 0.001

    times_RBCD = np.zeros((nb_exp, nb_ds))
    times_RGAS = np.zeros((nb_exp, nb_ds))
    times_RABCD = np.zeros((nb_exp, nb_ds))
    times_RAGAS = np.zeros((nb_exp, nb_ds))
    times_SRW = np.zeros((nb_exp, nb_ds))
      
    for t in range(nb_exp):
        print(t)
        reg = 0.2
        for ind_d in range(nb_ds):
            d = ds[ind_d]

            a,b,X,Y = fragmented_hypercube(n,d,dim=2)
            
            if d>=250:
                reg=0.5

            U0 = InitialStiefel(d, k)

            print('RBCD')
            RBCD = RiemannianBlockCoordinateDescent(eta=reg, tau=tau, max_iter=max_iter, threshold=threshold, verbose=False)
            PRW = ProjectedRobustWasserstein(X, Y, a, b, RBCD, k)
            PRW.run( 'RBCD',tau, U0)
            times_RBCD[t,ind_d] = PRW.running_time
            print('RABCD')
            PRW.run( 'RABCD',tau, U0)
            times_RABCD[t,ind_d] = PRW.running_time


            RGAS = RiemannianGradientAscentSinkhorn(eta=reg, tau = tau/reg, max_iter=max_iter, threshold=threshold, 
                                                    sink_threshold=1e-8, verbose=False)
            PRW1 = ProjectedRobustWasserstein(X, Y, a, b, RGAS, k)
            PRW1.run('RGAS',tau/reg, U0)
            times_RGAS[t,ind_d] = PRW1.running_time
            print('RAGAS')
            PRW1.run('RAGAS',tau/reg, U0)
            times_RAGAS[t,ind_d] = PRW1.running_time
            
            print('FWSRW')
            algo = FrankWolfe(reg=reg, step_size_0=tau, max_iter=max_iter, max_iter_sinkhorn=max_iter_sinkhorn,
                              threshold=(0.1*tau)**2, threshold_sinkhorn=threshold_sinkhorn, use_gpu=False)
            SRW = SubspaceRobustWasserstein(X, Y, a, b, algo, k)
            tic = time.time()
            SRW.run()
            tac = time.time()
            times_SRW[t, ind_d] = tac - tic
    
    print("exp_hypercubic_fix_n")
    
    times_RBCD_mean = np.mean(times_RBCD, axis=0)
    times_RABCD_mean = np.mean(times_RABCD, axis=0)
    times_RGAS_mean = np.mean(times_RGAS, axis=0)
    times_RAGAS_mean = np.mean(times_RAGAS, axis=0)
    times_SRW_mean = np.mean(times_SRW, axis=0)
    
    
    print('RBCD &', "%.2f &" %times_RBCD_mean[0], "%.2f &" %times_RBCD_mean[1],"%.2f &" %times_RBCD_mean[2], "%.2f &" %times_RBCD_mean[3], "%.2f "% times_RBCD_mean[4],"\\ \hline")
    print('RABCD &', "%.2f &" %times_RABCD_mean[0], "%.2f &" %times_RABCD_mean[1],"%.2f &" %times_RABCD_mean[2], "%.2f &" %times_RABCD_mean[3], "%.2f "% times_RABCD_mean[4], "\\ \hline")
    print('RGAS &', "%.2f &" %times_RGAS_mean[0], "%.2f &" %times_RGAS_mean[1],"%.2f &" %times_RGAS_mean[2], "%.2f &" %times_RGAS_mean[3], "%.2f "% times_RGAS_mean[4], "\\ \hline")
    print('RAGAS &', "%.2f &" %times_RAGAS_mean[0], "%.2f &" %times_RAGAS_mean[1],"%.2f &" %times_RAGAS_mean[2], "%.2f &" %times_RAGAS_mean[3], "%.2f "% times_RAGAS_mean[4], "\\ \hline")
    print('SRW &',  "%.2f &" %times_SRW_mean[0], "%.2f &" %times_SRW_mean[1],"%.2f &" %times_SRW_mean[2], "%.2f &" %times_SRW_mean[3], "%.2f "% times_SRW_mean[4], "\\ \hline")
    
    
            

#     times_RBCD_mean = np.mean(times_RBCD, axis=0)
#     times_RBCD_min = np.min(times_RBCD, axis=0)
#     times_RBCD_10 = np.percentile(times_RBCD, 10, axis=0)
#     times_RBCD_25 = np.percentile(times_RBCD, 25, axis=0)
#     times_RBCD_75 = np.percentile(times_RBCD, 75, axis=0)
#     times_RBCD_90 = np.percentile(times_RBCD, 90, axis=0)
#     times_RBCD_max = np.max(times_RBCD, axis=0)
    
#     times_RABCD_mean = np.mean(times_RABCD, axis=0)
#     times_RABCD_min = np.min(times_RABCD, axis=0)
#     times_RABCD_10 = np.percentile(times_RABCD, 10, axis=0)
#     times_RABCD_25 = np.percentile(times_RABCD, 25, axis=0)
#     times_RABCD_75 = np.percentile(times_RABCD, 75, axis=0)
#     times_RABCD_90 = np.percentile(times_RABCD, 90, axis=0)
#     times_RABCD_max = np.max(times_RABCD, axis=0)

#     times_RGAS_mean = np.mean(times_RGAS, axis=0)
#     times_RGAS_min = np.min(times_RGAS, axis=0)
#     times_RGAS_10 = np.percentile(times_RGAS, 10, axis=0)
#     times_RGAS_25 = np.percentile(times_RGAS, 25, axis=0)
#     times_RGAS_75 = np.percentile(times_RGAS, 75, axis=0)
#     times_RGAS_90 = np.percentile(times_RGAS, 90, axis=0)
#     times_RGAS_max = np.max(times_RGAS, axis=0)

#     times_RAGAS_mean = np.mean(times_RAGAS, axis=0)
#     times_RAGAS_min = np.min(times_RAGAS, axis=0)
#     times_RAGAS_10 = np.percentile(times_RAGAS, 10, axis=0)
#     times_RAGAS_25 = np.percentile(times_RAGAS, 25, axis=0)
#     times_RAGAS_75 = np.percentile(times_RAGAS, 75, axis=0)
#     times_RAGAS_90 = np.percentile(times_RAGAS, 90, axis=0)
#     times_RAGAS_max = np.max(times_RAGAS, axis=0)
    
#     times_SRW_mean = np.mean(times_SRW, axis=0)
#     times_SRW_min = np.min(times_SRW, axis=0)
#     times_SRW_10 = np.percentile(times_SRW, 10, axis=0)
#     times_SRW_25 = np.percentile(times_SRW, 25, axis=0)
#     times_SRW_75 = np.percentile(times_SRW, 75, axis=0)
#     times_SRW_90 = np.percentile(times_SRW, 90, axis=0)
#     times_SRW_max = np.max(times_SRW, axis=0)


#     import matplotlib.ticker as ticker
#     plt.figure(figsize=(12,8))
  
#     mean, = plt.loglog(ds[:], times_RBCD_mean[:], 'o-', lw=5, ms=10, label='RBCD')
#     col = mean.get_color()
#     plt.fill_between(ds[:], times_RBCD_25[:], times_RBCD_75[:], facecolor=col, alpha=0.3)
#     plt.fill_between(ds[:], times_RBCD_10[:], times_RBCD_90[:], facecolor=col, alpha=0.2)
#     plt.fill_between(ds[:], times_RBCD_min[:], times_RBCD_max[:], facecolor=col, alpha=0.15)
    
#     mean, = plt.loglog(ds[:], times_RABCD_mean[:], 'o-', lw=5, ms=10, label='RABCD')
#     col = mean.get_color()
#     plt.fill_between(ds[:], times_RABCD_25[:], times_RABCD_75[:], facecolor=col, alpha=0.3)
#     plt.fill_between(ds[:], times_RABCD_10[:], times_RABCD_90[:], facecolor=col, alpha=0.2)
#     plt.fill_between(ds[:], times_RABCD_min[:], times_RABCD_max[:], facecolor=col, alpha=0.15)

#     mean, = plt.loglog(ds[:], times_RGAS_mean[:], 'o-', lw=5, ms=10, label='RGAS')
#     col = mean.get_color()
#     plt.fill_between(ds[:], times_RGAS_25[:], times_RGAS_75[:], facecolor=col, alpha=0.3)
#     plt.fill_between(ds[:], times_RGAS_10[:], times_RGAS_90[:], facecolor=col, alpha=0.2)
#     plt.fill_between(ds[:], times_RGAS_min[:], times_RGAS_max[:], facecolor=col, alpha=0.15)

#     mean, = plt.loglog(ds[:], times_RAGAS_mean[:], 'o-', lw=5, ms=10, label='RAGAS')
#     col = mean.get_color()
#     plt.fill_between(ds[:], times_RAGAS_25[:], times_RAGAS_75[:], facecolor=col, alpha=0.3)
#     plt.fill_between(ds[:], times_RAGAS_10[:], times_RAGAS_90[:], facecolor=col, alpha=0.2)
#     plt.fill_between(ds[:], times_RAGAS_min[:], times_RAGAS_max[:], facecolor=col, alpha=0.15)

#     mean, = plt.loglog(ds[:], times_SRW_mean[:], 'o-', lw=5, ms=10, label='SRW')
#     col = mean.get_color()
#     plt.fill_between(ds[:], times_SRW_25[:], times_SRW_75[:], facecolor=col, alpha=0.3)
#     plt.fill_between(ds[:], times_SRW_10[:], times_SRW_90[:], facecolor=col, alpha=0.2)
#     plt.fill_between(ds[:], times_SRW_min[:], times_SRW_max[:], facecolor=col, alpha=0.15)



#     plt.xlabel('Dimension d', fontsize=25)
#     plt.ylabel('Execution time', fontsize=25)
#     plt.xticks(ds[:], fontsize=20)
#     plt.yticks(fontsize=20)
#     plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
#     plt.grid(ls=':')
#     plt.legend(loc='best', fontsize=25)
#     plt.savefig('figs/exp4_computation_time_fixn.png')

    
def plot_fix_d():
    
    ds = [50, 100, 250, 500, 1000] # Number of points in the measures

    nb_ds = len(ds)
    d = 50 # Dimensions 
    k = 2 # Dimension parameter
    max_iter = 2000 # Maximum number of iterations
    max_iter_sinkhorn = 1000 # Maximum number of iterations in Sinkhorn
    threshold = 0.1 # Stopping threshold
    threshold_sinkhorn = 1e-9 # Stopping threshold in Sinkhorn
    nb_exp = 100 # Number of experiments
    tau = 0.001

    times_RBCD = np.zeros((nb_exp, nb_ds))
    times_RGAS = np.zeros((nb_exp, nb_ds))
    times_RABCD = np.zeros((nb_exp, nb_ds))
    times_RAGAS = np.zeros((nb_exp, nb_ds))
    times_SRW = np.zeros((nb_exp, nb_ds))


    for t in range(nb_exp):
        print(t)
        reg = 0.2
        for ind_d in range(nb_ds):
            n = ds[ind_d]
            a,b,X,Y = fragmented_hypercube(n,d,dim=2)

            U0 = InitialStiefel(d, k)

            print('RBCD')
            RBCD = RiemannianBlockCoordinateDescent(eta=reg, tau=tau, max_iter=max_iter, threshold=threshold, verbose=False)
            PRW = ProjectedRobustWasserstein(X, Y, a, b, RBCD, k)
            PRW.run( 'RBCD',tau, U0)
            times_RBCD[t,ind_d] = PRW.running_time
            print('RABCD')
            PRW.run( 'RABCD',tau, U0)
            times_RABCD[t,ind_d] = PRW.running_time


            print('RGAS')
            RGAS = RiemannianGradientAscentSinkhorn(eta=reg, tau = tau/reg, max_iter=max_iter, threshold=threshold, 
                                                    sink_threshold=1e-8, verbose=False)
            PRW1 = ProjectedRobustWasserstein(X, Y, a, b, RGAS, k)
            PRW1.run('RGAS',tau/reg, U0)
            times_RGAS[t,ind_d] = PRW1.running_time
            print('RAGAS')
            PRW1.run('RAGAS',tau/reg, U0)
            times_RAGAS[t,ind_d] = PRW1.running_time
            
            print('FWSRW')
            algo = FrankWolfe(reg=reg, step_size_0=tau, max_iter=max_iter, max_iter_sinkhorn=max_iter_sinkhorn,
                              threshold=(0.1*tau)**2, threshold_sinkhorn=threshold_sinkhorn, use_gpu=False)
            SRW = SubspaceRobustWasserstein(X, Y, a, b, algo, k)
            tic = time.time()
            SRW.run()
            tac = time.time()
            times_SRW[t, ind_d] = tac - tic
            
    print("exp_hypercubic_fix_d")
            
    times_RBCD_mean = np.mean(times_RBCD, axis=0)
    times_RABCD_mean = np.mean(times_RABCD, axis=0)
    times_RGAS_mean = np.mean(times_RGAS, axis=0)
    times_RAGAS_mean = np.mean(times_RAGAS, axis=0)
    times_SRW_mean = np.mean(times_SRW, axis=0)
    
    
    print('RBCD &', "%.2f &" %times_RBCD_mean[0], "%.2f &" %times_RBCD_mean[1],"%.2f &" %times_RBCD_mean[2], "%.2f &" %times_RBCD_mean[3], "%.2f "% times_RBCD_mean[4], "\\ \hline")
    print('RABCD &', "%.2f &" %times_RABCD_mean[0], "%.2f &" %times_RABCD_mean[1],"%.2f &" %times_RABCD_mean[2], "%.2f &" %times_RABCD_mean[3], "%.2f "% times_RABCD_mean[4], "\\ \hline")
    print('RGAS &', "%.2f &" %times_RGAS_mean[0], "%.2f &" %times_RGAS_mean[1],"%.2f &" %times_RGAS_mean[2], "%.2f &" %times_RGAS_mean[3], "%.2f "% times_RGAS_mean[4], "\\ \hline")
    print('RAGAS &', "%.2f &" %times_RAGAS_mean[0], "%.2f &" %times_RAGAS_mean[1],"%.2f &" %times_RAGAS_mean[2], "%.2f &" %times_RAGAS_mean[3], "%.2f "% times_RAGAS_mean[4], "\\ \hline")
    print('SRW &',  "%.2f &" %times_SRW_mean[0], "%.2f &" %times_SRW_mean[1],"%.2f &" %times_SRW_mean[2], "%.2f &" %times_SRW_mean[3], "%.2f "% times_SRW_mean[4], "\\ \hline")
    
    
    
    
    
#     times_RBCD_min = np.min(times_RBCD, axis=0)
#     times_RBCD_10 = np.percentile(times_RBCD, 10, axis=0)
#     times_RBCD_25 = np.percentile(times_RBCD, 25, axis=0)
#     times_RBCD_75 = np.percentile(times_RBCD, 75, axis=0)
#     times_RBCD_90 = np.percentile(times_RBCD, 90, axis=0)
#     times_RBCD_max = np.max(times_RBCD, axis=0)
#     print('RBCD &', "%.2f &" %times_RBCD_mean[0], "%.2f &" %times_RBCD_mean[1],"%.2f &" %times_RBCD_mean[2], "%.2f &" %times_RBCD_mean[3], "%.2f "% times_RBCD_mean[4])
    
#     times_RABCD_mean = np.mean(times_RABCD, axis=0)
#     times_RABCD_min = np.min(times_RABCD, axis=0)
#     times_RABCD_10 = np.percentile(times_RABCD, 10, axis=0)
#     times_RABCD_25 = np.percentile(times_RABCD, 25, axis=0)
#     times_RABCD_75 = np.percentile(times_RABCD, 75, axis=0)
#     times_RABCD_90 = np.percentile(times_RABCD, 90, axis=0)
#     times_RABCD_max = np.max(times_RABCD, axis=0)
#     print('RABCD &', "%.2f &" %times_RABCD_mean[0], "%.2f &" %times_RABCD_mean[1],"%.2f &" %times_RABCD_mean[2], "%.2f &" %times_RABCD_mean[3], "%.2f "% times_RABCD_mean[4])
#     times_RGAS_mean = np.mean(times_RGAS, axis=0)
#     times_RGAS_min = np.min(times_RGAS, axis=0)
#     times_RGAS_10 = np.percentile(times_RGAS, 10, axis=0)
#     times_RGAS_25 = np.percentile(times_RGAS, 25, axis=0)
#     times_RGAS_75 = np.percentile(times_RGAS, 75, axis=0)
#     times_RGAS_90 = np.percentile(times_RGAS, 90, axis=0)
#     times_RGAS_max = np.max(times_RGAS, axis=0)
#     print('RGAS &', "%.2f &" %times_RGAS_mean[0], "%.2f &" %times_RGAS_mean[1],"%.2f &" %times_RGAS_mean[2], "%.2f &" %times_RGAS_mean[3], "%.2f "% times_RGAS_mean[4])
#     times_RAGAS_mean = np.mean(times_RAGAS, axis=0)
#     times_RAGAS_min = np.min(times_RAGAS, axis=0)
#     times_RAGAS_10 = np.percentile(times_RAGAS, 10, axis=0)
#     times_RAGAS_25 = np.percentile(times_RAGAS, 25, axis=0)
#     times_RAGAS_75 = np.percentile(times_RAGAS, 75, axis=0)
#     times_RAGAS_90 = np.percentile(times_RAGAS, 90, axis=0)
#     times_RAGAS_max = np.max(times_RAGAS, axis=0)
#     print('RAGAS &', "%.2f &" %times_RAGAS_mean[0], "%.2f &" %times_RAGAS_mean[1],"%.2f &" %times_RAGAS_mean[2], "%.2f &" %times_RAGAS_mean[3], "%.2f "% times_RAGAS_mean[4])
#     times_SRW_mean = np.mean(times_SRW, axis=0)
#     times_SRW_min = np.min(times_SRW, axis=0)
#     times_SRW_10 = np.percentile(times_SRW, 10, axis=0)
#     times_SRW_25 = np.percentile(times_SRW, 25, axis=0)
#     times_SRW_75 = np.percentile(times_SRW, 75, axis=0)
#     times_SRW_90 = np.percentile(times_SRW, 90, axis=0)
#     times_SRW_max = np.max(times_SRW, axis=0)
#     print('SRW &',  "%.2f &" %times_SRW_mean[0], "%.2f &" %times_SRW_mean[1],"%.2f &" %times_SRW_mean[2], "%.2f &" %times_SRW_mean[3], "%.2f "% times_SRW_mean[4])


#     import matplotlib.ticker as ticker
#     plt.figure(figsize=(12,8))

   
#     mean, = plt.loglog(ds[:], times_RBCD_mean[:], 'o-', lw=5, ms=10, label='RBCD')
#     col = mean.get_color()
#     plt.fill_between(ds[:], times_RBCD_25[:], times_RBCD_75[:], facecolor=col, alpha=0.3)
#     plt.fill_between(ds[:], times_RBCD_10[:], times_RBCD_90[:], facecolor=col, alpha=0.2)
#     plt.fill_between(ds[:], times_RBCD_min[:], times_RBCD_max[:], facecolor=col, alpha=0.15)
    
#     mean, = plt.loglog(ds[:], times_RABCD_mean[:], 'o-', lw=5, ms=10, label='RABCD')
#     col = mean.get_color()
#     plt.fill_between(ds[:], times_RABCD_25[:], times_RABCD_75[:], facecolor=col, alpha=0.3)
#     plt.fill_between(ds[:], times_RABCD_10[:], times_RABCD_90[:], facecolor=col, alpha=0.2)
#     plt.fill_between(ds[:], times_RABCD_min[:], times_RABCD_max[:], facecolor=col, alpha=0.15)

#     mean, = plt.loglog(ds[:], times_RGAS_mean[:], 'o-', lw=5, ms=10, label='RGAS')
#     col = mean.get_color()
#     plt.fill_between(ds[:], times_RGAS_25[:], times_RGAS_75[:], facecolor=col, alpha=0.3)
#     plt.fill_between(ds[:], times_RGAS_10[:], times_RGAS_90[:], facecolor=col, alpha=0.2)
#     plt.fill_between(ds[:], times_RGAS_min[:], times_RGAS_max[:], facecolor=col, alpha=0.15)

#     mean, = plt.loglog(ds[:], times_RAGAS_mean[:], 'o-', lw=5, ms=10, label='RAGAS')
#     col = mean.get_color()
#     plt.fill_between(ds[:], times_RAGAS_25[:], times_RAGAS_75[:], facecolor=col, alpha=0.3)
#     plt.fill_between(ds[:], times_RAGAS_10[:], times_RAGAS_90[:], facecolor=col, alpha=0.2)
#     plt.fill_between(ds[:], times_RAGAS_min[:], times_RAGAS_max[:], facecolor=col, alpha=0.15)

#     mean, = plt.loglog(ds[:], times_SRW_mean[:], 'o-', lw=5, ms=10, label='SRW')
#     col = mean.get_color()
#     plt.fill_between(ds[:], times_SRW_25[:], times_SRW_75[:], facecolor=col, alpha=0.3)
#     plt.fill_between(ds[:], times_SRW_10[:], times_SRW_90[:], facecolor=col, alpha=0.2)
#     plt.fill_between(ds[:], times_SRW_min[:], times_SRW_max[:], facecolor=col, alpha=0.15)

#     plt.xlabel('Number of points n', fontsize=25)
#     plt.ylabel('Execution time', fontsize=25)
#     plt.xticks(ds[:], fontsize=20)
#     plt.yticks(fontsize=20)
#     plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
#     plt.grid(ls=':')
#     plt.legend(loc='best', fontsize=25)
#     plt.savefig('figs/exp4_computation_time_fixd.png')
    
def plot_n_equal_d():
    
    ds = [10, 20, 50, 100, 250] # Number of points in the measures

    nb_ds = len(ds)
    k = 2 # Dimension parameter
    max_iter = 2000 # Maximum number of iterations
    max_iter_sinkhorn = 1000 # Maximum number of iterations in Sinkhorn
    threshold = 0.1 # Stopping threshold
    threshold_sinkhorn = 1e-9 # Stopping threshold in Sinkhorn
    nb_exp = 5 # Number of experiments
    
    tau = 0.001

    times_RBCD = np.zeros((nb_exp, nb_ds))
    times_RGAS = np.zeros((nb_exp, nb_ds))
    times_RABCD = np.zeros((nb_exp, nb_ds))
    times_RAGAS = np.zeros((nb_exp, nb_ds))
    times_SRW = np.zeros((nb_exp, nb_ds))


    for t in range(nb_exp):
        print(t)
        reg = 0.2
        for ind_d in range(nb_ds):
            d = ds[ind_d]
            n = 10*d
            print(d, n)
            a,b,X,Y = fragmented_hypercube(n,d,dim=2)

            if d>=250:
                reg=0.5

            U0 = InitialStiefel(d, k)

            print('RBCD')
            RBCD = RiemannianBlockCoordinateDescent(eta=reg, tau=tau, max_iter=max_iter, threshold=threshold, verbose=True)
            PRW = ProjectedRobustWasserstein(X, Y, a, b, RBCD, k)
            PRW.run( 'RBCD',tau, U0)
            times_RBCD[t,ind_d] = PRW.running_time
            print('RABCD')
            PRW.run( 'RABCD',tau, U0)
            times_RABCD[t,ind_d] = PRW.running_time


            print('RGAS')
            RGAS = RiemannianGradientAscentSinkhorn(eta=reg, tau = tau/reg, max_iter=max_iter, threshold=threshold, 
                                                    sink_threshold=threshold_sinkhorn, verbose=True)
            PRW1 = ProjectedRobustWasserstein(X, Y, a, b, RGAS, k)
            PRW1.run('RGAS',tau/reg, U0)
            times_RGAS[t,ind_d] = PRW1.running_time
            print('RAGAS')
            PRW1.run('RAGAS',tau/reg, U0)
            times_RAGAS[t,ind_d] = PRW1.running_time
            
            print('FWSRW')
            algo = FrankWolfe(reg=reg, step_size_0=tau, max_iter=max_iter, max_iter_sinkhorn=max_iter_sinkhorn,
                              threshold=(0.1*tau)**2, threshold_sinkhorn=threshold_sinkhorn, use_gpu=False)
            SRW = SubspaceRobustWasserstein(X, Y, a, b, algo, k)
            tic = time.time()
            SRW.run()
            tac = time.time()
            times_SRW[t, ind_d] = tac - tic
            
    print("exp_hypercubic_n_equal_10d")
            
    times_RBCD_mean = np.mean(times_RBCD, axis=0)
    times_RABCD_mean = np.mean(times_RABCD, axis=0)
    times_RGAS_mean = np.mean(times_RGAS, axis=0)
    times_RAGAS_mean = np.mean(times_RAGAS, axis=0)
    times_SRW_mean = np.mean(times_SRW, axis=0)
    
    
    print('RBCD &', "%.2f &" %times_RBCD_mean[0], "%.2f &" %times_RBCD_mean[1],"%.2f &" %times_RBCD_mean[2], "%.2f &" %times_RBCD_mean[3], "%.2f "% times_RBCD_mean[4], "\\ \hline")
    print('RABCD &', "%.2f &" %times_RABCD_mean[0], "%.2f &" %times_RABCD_mean[1],"%.2f &" %times_RABCD_mean[2], "%.2f &" %times_RABCD_mean[3], "%.2f "% times_RABCD_mean[4], "\\ \hline")
    print('RGAS &', "%.2f &" %times_RGAS_mean[0], "%.2f &" %times_RGAS_mean[1],"%.2f &" %times_RGAS_mean[2], "%.2f &" %times_RGAS_mean[3], "%.2f "% times_RGAS_mean[4], "\\ \hline")
    print('RAGAS &', "%.2f &" %times_RAGAS_mean[0], "%.2f &" %times_RAGAS_mean[1],"%.2f &" %times_RAGAS_mean[2], "%.2f &" %times_RAGAS_mean[3], "%.2f "% times_RAGAS_mean[4], "\\ \hline")
    print('SRW &',  "%.2f &" %times_SRW_mean[0], "%.2f &" %times_SRW_mean[1],"%.2f &" %times_SRW_mean[2], "%.2f &" %times_SRW_mean[3], "%.2f "% times_SRW_mean[4], "\\ \hline")
    
            
#     times_RBCD_mean = np.mean(times_RBCD, axis=0)
#     times_RBCD_min = np.min(times_RBCD, axis=0)
#     times_RBCD_10 = np.percentile(times_RBCD, 10, axis=0)
#     times_RBCD_25 = np.percentile(times_RBCD, 25, axis=0)
#     times_RBCD_75 = np.percentile(times_RBCD, 75, axis=0)
#     times_RBCD_90 = np.percentile(times_RBCD, 90, axis=0)
#     times_RBCD_max = np.max(times_RBCD, axis=0)
    
#     times_RABCD_mean = np.mean(times_RABCD, axis=0)
#     times_RABCD_min = np.min(times_RABCD, axis=0)
#     times_RABCD_10 = np.percentile(times_RABCD, 10, axis=0)
#     times_RABCD_25 = np.percentile(times_RABCD, 25, axis=0)
#     times_RABCD_75 = np.percentile(times_RABCD, 75, axis=0)
#     times_RABCD_90 = np.percentile(times_RABCD, 90, axis=0)
#     times_RABCD_max = np.max(times_RABCD, axis=0)

#     times_RGAS_mean = np.mean(times_RGAS, axis=0)
#     times_RGAS_min = np.min(times_RGAS, axis=0)
#     times_RGAS_10 = np.percentile(times_RGAS, 10, axis=0)
#     times_RGAS_25 = np.percentile(times_RGAS, 25, axis=0)
#     times_RGAS_75 = np.percentile(times_RGAS, 75, axis=0)
#     times_RGAS_90 = np.percentile(times_RGAS, 90, axis=0)
#     times_RGAS_max = np.max(times_RGAS, axis=0)

#     times_RAGAS_mean = np.mean(times_RAGAS, axis=0)
#     times_RAGAS_min = np.min(times_RAGAS, axis=0)
#     times_RAGAS_10 = np.percentile(times_RAGAS, 10, axis=0)
#     times_RAGAS_25 = np.percentile(times_RAGAS, 25, axis=0)
#     times_RAGAS_75 = np.percentile(times_RAGAS, 75, axis=0)
#     times_RAGAS_90 = np.percentile(times_RAGAS, 90, axis=0)
#     times_RAGAS_max = np.max(times_RAGAS, axis=0)
    
#     times_SRW_mean = np.mean(times_SRW, axis=0)
#     times_SRW_min = np.min(times_SRW, axis=0)
#     times_SRW_10 = np.percentile(times_SRW, 10, axis=0)
#     times_SRW_25 = np.percentile(times_SRW, 25, axis=0)
#     times_SRW_75 = np.percentile(times_SRW, 75, axis=0)
#     times_SRW_90 = np.percentile(times_SRW, 90, axis=0)
#     times_SRW_max = np.max(times_SRW, axis=0)


#     import matplotlib.ticker as ticker
#     plt.figure(figsize=(12,8))

   
#     mean, = plt.loglog(ds[:], times_RBCD_mean[:], 'o-', lw=5, ms=10, label='RBCD')
#     col = mean.get_color()
#     plt.fill_between(ds[:], times_RBCD_25[:], times_RBCD_75[:], facecolor=col, alpha=0.3)
#     plt.fill_between(ds[:], times_RBCD_10[:], times_RBCD_90[:], facecolor=col, alpha=0.2)
#     plt.fill_between(ds[:], times_RBCD_min[:], times_RBCD_max[:], facecolor=col, alpha=0.15)
    
#     mean, = plt.loglog(ds[:], times_RABCD_mean[:], 'o-', lw=5, ms=10, label='RABCD')
#     col = mean.get_color()
#     plt.fill_between(ds[:], times_RABCD_25[:], times_RABCD_75[:], facecolor=col, alpha=0.3)
#     plt.fill_between(ds[:], times_RABCD_10[:], times_RABCD_90[:], facecolor=col, alpha=0.2)
#     plt.fill_between(ds[:], times_RABCD_min[:], times_RABCD_max[:], facecolor=col, alpha=0.15)

#     mean, = plt.loglog(ds[:], times_RGAS_mean[:], 'o-', lw=5, ms=10, label='RGAS')
#     col = mean.get_color()
#     plt.fill_between(ds[:], times_RGAS_25[:], times_RGAS_75[:], facecolor=col, alpha=0.3)
#     plt.fill_between(ds[:], times_RGAS_10[:], times_RGAS_90[:], facecolor=col, alpha=0.2)
#     plt.fill_between(ds[:], times_RGAS_min[:], times_RGAS_max[:], facecolor=col, alpha=0.15)

#     mean, = plt.loglog(ds[:], times_RAGAS_mean[:], 'o-', lw=5, ms=10, label='RAGAS')
#     col = mean.get_color()
#     plt.fill_between(ds[:], times_RAGAS_25[:], times_RAGAS_75[:], facecolor=col, alpha=0.3)
#     plt.fill_between(ds[:], times_RAGAS_10[:], times_RAGAS_90[:], facecolor=col, alpha=0.2)
#     plt.fill_between(ds[:], times_RAGAS_min[:], times_RAGAS_max[:], facecolor=col, alpha=0.15)

#     mean, = plt.loglog(ds[:], times_SRW_mean[:], 'o-', lw=5, ms=10, label='SRW')
#     col = mean.get_color()
#     plt.fill_between(ds[:], times_SRW_25[:], times_SRW_75[:], facecolor=col, alpha=0.3)
#     plt.fill_between(ds[:], times_SRW_10[:], times_SRW_90[:], facecolor=col, alpha=0.2)
#     plt.fill_between(ds[:], times_SRW_min[:], times_SRW_max[:], facecolor=col, alpha=0.15)

#     plt.xlabel('Dimension d', fontsize=25)
#     plt.ylabel('Execution time', fontsize=25)
#     plt.xticks(ds[:], fontsize=20)
#     plt.yticks(fontsize=20)
#     plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
#     plt.grid(ls=':')
#     plt.legend(loc='best', fontsize=25)
#     plt.savefig('figs/exp4_computation_hypercubic_time_n_equal_100d.png')

    
if __name__ == "__main__":
#     plot_fix_n()
#     plot_fix_d()
    plot_n_equal_d()