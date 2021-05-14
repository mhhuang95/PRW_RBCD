#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from collections import Counter
from heapq import nlargest

import os
import sys
import string
import io
sys.path.insert(0, "../")

from SRW import SubspaceRobustWasserstein
from PRW import ProjectedRobustWasserstein
from Optimization.RiemannianBCD import RiemannianBlockCoordinateDescent
from Optimization.RiemannianGAS import RiemannianGradientAscentSinkhorn


def load_vectors(fname, size=None):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    i = 0
    for line in fin:
        if size and i >= size:
            break
        if i >= 2000:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.array(tokens[1:], dtype='f8')
        i += 1
    return data


def textToMeasure(text):
    text = text.replace('\n', '')
    text = text.replace('\t', '')
    words = text.split(' ')
    table = str.maketrans('', '', string.punctuation.replace("'", ""))
    words = [w.translate(table) for w in words if len(w) > 0]
    words = [w for w in words if w in dictionnary.keys()]
    words = [w for w in words if not w[0].isupper()]
    words = [w for w in words if not w.isdigit()]
    cX = Counter(words)
    size = len(words)
    cX = Counter(words)
    words = list(set(words))
    a = np.array([cX[w] for w in words])/size
    X = np.array([dictionnary[w] for w in words])
    return X, a, words


def load_text(file):
    """return X,a,words"""
    with open(file) as fp:
        text = fp.read()
    return textToMeasure(text)

if not os.path.isfile('./wiki-news-300d-1M.vec'):
    print('[Warning]')
    print('Please download word vector at https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip')
    print('Put the unzipped "wiki-news-300d-1M.vec" in Data/Text folder, then try again')
    exit()

dictionnary = load_vectors('./wiki-news-300d-1M.vec', size=20000)
dictionnary_pd = pd.DataFrame(dictionnary).T

def plot_pushforwards_wordcloud(SRW, words_X, words_Y):
    """Plot the projected measures as word clouds."""
    proj_X, proj_Y = SRW.get_projected_pushforwards()
    N_print_words = 30
    plt.figure(figsize=(10,10))
    plt.scatter(proj_X[:,0], proj_X[:,1], s=X.shape[0]*20*a, c='r', zorder=10, alpha=0.)
    plt.scatter(proj_Y[:,0], proj_Y[:,1], s=Y.shape[0]*20*b, c='b', zorder=10, alpha=0.)
    large_a = nlargest(N_print_words, [a[words_X.index(i)] for i in words_X if i not in words_Y])[-1]
    large_b = nlargest(N_print_words, [b[words_Y.index(i)] for i in words_Y if i not in words_X])[-1]
    large_ab = nlargest(N_print_words, [0.5*a[words_X.index(i)] + 0.5*b[words_Y.index(i)] for i in words_Y if i in words_X])[-1]
    for i in range(a.shape[0]):
        if a[i] > large_a:
            if words_X[i] not in words_Y:
                plt.gca().annotate(words_X[i], proj_X[i,:], size=2500*a[i], color='b', ha='center', alpha=0.8)
    for j in range(b.shape[0]):
        if b[j] > large_b and words_Y[j] not in words_X:
            plt.gca().annotate(words_Y[j], proj_Y[j,:], size=2500*b[j], color='r', ha='center', alpha=0.8)
        elif words_Y[j] in words_X and 0.5*b[j]+0.5*a[words_X.index(words_Y[j])] > large_ab:
            size = 0.5*b[j] + 0.5*a[words_X.index(words_Y[j])]
            plt.gca().annotate(words_Y[j], proj_Y[j,:], size=2500*size, color='darkviolet', ha='center', alpha=0.8)
    plt.axis('equal')
    plt.axis('off')
    plt.show()


def InitialStiefel(d, k):
    U = np.random.randn(d, k)
    q, r = np.linalg.qr(U)
    return q

    
def main():
    scripts = ['DUNKIRK.txt', 'GRAVITY.txt', 'INTERSTELLAR.txt', 'KILL_BILL_VOLUME_1.txt', 'KILL_BILL_VOLUME_2.txt', 'THE_MARTIAN.txt', 'TITANIC.txt']
    Nb_scripts = len(scripts)
    PRW_matrix = np.zeros((Nb_scripts, Nb_scripts))
    PRW1_matrix = np.zeros((Nb_scripts, Nb_scripts))
    PRW_matrix_adap = np.zeros((Nb_scripts, Nb_scripts))
    PRW1_matrix_adap = np.zeros((Nb_scripts, Nb_scripts))
    measures = []
    for film in scripts:
        measures.append(load_text('Data/movies/'+film))

    nb_exp = 10

    times_RBCD = np.zeros((Nb_scripts, Nb_scripts))
    times_RGAS = np.zeros((Nb_scripts, Nb_scripts))
    times_RABCD = np.zeros((Nb_scripts, Nb_scripts))
    times_RAGAS = np.zeros((Nb_scripts, Nb_scripts))

    for t in range(nb_exp): 
        for film1 in scripts:
            for film2 in scripts:
                i = scripts.index(film1)
                j = scripts.index(film2)
                if i < j:
                    X,a,words_X = measures[i]
                    Y,b,words_Y = measures[j]

                    U0 = InitialStiefel(300, 2)

                    reg = 0.1
                    tau = 0.1

                    RBCD = RiemannianBlockCoordinateDescent(eta=reg, tau=tau, max_iter=1000, threshold=0.001)
                    PRW_ = ProjectedRobustWasserstein(X, Y, a, b, RBCD, k=2)
                    PRW_.run('RBCD',tau,  U0)
                    PRW_matrix[i, j] = PRW_.get_value()
                    times_RBCD[i, j] = times_RBCD[i, j] + PRW_.get_time()
                    print('RBCD (', film1, ',', film2, ') =', PRW_matrix[i, j])

                    RGAS = RiemannianGradientAscentSinkhorn(eta=reg, tau = tau/reg, max_iter = 1000, threshold=0.001)
                    PRW1_ = ProjectedRobustWasserstein(X, Y, a, b, RGAS, k=2)
                    PRW1_.run('RGAS',tau/reg, U0)
                    PRW1_matrix[i,j] = PRW1_.get_value()
                    times_RGAS[i,j] = times_RBCD[i,j] + PRW1_.get_time()
                    print('RGAS (', film1, ',', film2, ') =', PRW1_matrix[i,j])

                    # reg = 0.1
                    # tau = 0.005

                    # RBCD = RiemannianBlockCoordinateDescent(eta=reg, tau = tau, max_iter = 1000, threshold=0.001)
                    # PRW_ = ProjectedRobustWasserstein(X, Y, a, b, RBCD, k=2)
                    # PRW_.run('RABCD',tau, U0)
                    # PRW_matrix_adap[i,j] = PRW_.get_value()
                    # times_RABCD[i,j] = times_RBCD[i,j] + PRW_.get_time()
                    # print('RABCD (', film1, ',', film2, ') =', PRW_matrix[i,j])

                    # RGAS = RiemannianGradientAscentSinkhorn(eta=reg, tau = tau/reg, max_iter = 1000, threshold=0.001)
                    # PRW1_ = ProjectedRobustWasserstein(X, Y, a, b, RGAS, k=2)
                    # PRW1_.run('RAGAS',tau/reg, U0)
                    # PRW1_matrix_adap[i,j] = PRW1_.get_value()
                    # times_RAGAS[i,j] = times_RBCD[i,j] + PRW1_.get_time()
                    # print('RAGAS (', film1, ',', film2, ') =', PRW1_matrix_adap[i,j])




if __name__ == "__main__":
    main()

