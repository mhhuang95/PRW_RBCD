#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################################################################
# Experiment for showing PRW on learning topics on Shakespeare corpus
############################################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from heapq import nlargest
import io
import string
from collections import Counter

sys.path.insert(0, "../")
from PRW import ProjectedRobustWasserstein
from Optimization.RiemannianBCD import RiemannianBlockCoordinateDescent
from Optimization.RiemannianGAS import RiemannianGradientAscentSinkhorn

sys.path.insert(0, "./Data/Text/")


# from exp3_cinema import load_vectors, load_text, plot_pushforwards_wordcloud


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


def plot_pushforwards_wordcloud(PRW, words_X, words_Y, X, Y, a, b, corpus_name, mode_str):
    """Plot the projected measures as word clouds."""
    proj_X, proj_Y = PRW.get_projected_pushforwards()
    N_print_words = 30
    plt.figure(figsize=(10, 10))
    plt.scatter(proj_X[:, 0], proj_X[:, 1], s=X.shape[0] * 20 * a, c='r', zorder=10, alpha=0.)
    plt.scatter(proj_Y[:, 0], proj_Y[:, 1], s=Y.shape[0] * 20 * b, c='b', zorder=10, alpha=0.)
    large_a = nlargest(N_print_words, [a[words_X.index(i)] for i in words_X if i not in words_Y])[-1]
    large_b = nlargest(N_print_words, [b[words_Y.index(i)] for i in words_Y if i not in words_X])[-1]
    large_ab = \
        nlargest(N_print_words,
                 [0.5 * a[words_X.index(i)] + 0.5 * b[words_Y.index(i)] for i in words_Y if i in words_X])[
            -1]
    for i in range(a.shape[0]):
        if words_X[i] in ['thou', 'thee', 'thy']:
            continue
        if a[i] > large_a:
            if words_X[i] not in words_Y:
                plt.gca().annotate(words_X[i], proj_X[i, :], size=2500 * a[i], color='b', ha='center', alpha=0.8)
    for j in range(b.shape[0]):
        if words_Y[j] in ['thou', 'thee', 'thy']:
            continue
        if b[j] > large_b and words_Y[j] not in words_X:
            plt.gca().annotate(words_Y[j], proj_Y[j, :], size=2500 * b[j], color='r', ha='center', alpha=0.8)
        elif words_Y[j] in words_X and 0.5 * b[j] + 0.5 * a[words_X.index(words_Y[j])] > large_ab:
            size = 0.5 * b[j] + 0.5 * a[words_X.index(words_Y[j])]
            plt.gca().annotate(words_Y[j], proj_Y[j, :], size=2500 * size, color='darkviolet', ha='center', alpha=0.8)
    plt.axis('equal')
    plt.axis('off')
    plt.savefig('./figs/exp3_%s_wordcloud_%s.png' % (corpus_name, mode_str))
    plt.title('PRW projection of word clouds', fontsize=15)
    plt.close()


#########################################################################
# Shakespeare plays are downloadable from
# https://www.folgerdigitaltexts.org/download/txt.html
# we pre-downloaded them in Data/Text/Shakespeare
#########################################################################
names = ['Henry V', 'Hamlet', 'Julius Caesar', 'The Merchant of Venice', 'Othello',
         'Romeo and Juliet']
scripts = ['H5.txt', 'Ham.txt', 'JC.txt', 'MV.txt', 'Oth.txt', 'Rom.txt']

# print(len(names), len(scripts))
assert len(names) == len(scripts)

Nb_scripts = len(scripts)
PRW_matrix = np.zeros((Nb_scripts, Nb_scripts))
PRW1_matrix = np.zeros((Nb_scripts, Nb_scripts))
measures = []
for art in scripts:
    measures.append(load_text('Data/Text/Shakespeare/' + art))

np.random.seed(357)

def InitialStiefel(d, k):
    U = np.random.randn(d, k)
    q, r = np.linalg.qr(U)
    return q


def main():

    k = 2
    reg = 0.1
    tau = 0.001

    for art1 in scripts:
        for art2 in scripts:
            i = scripts.index(art1)
            j = scripts.index(art2)
            if i < j:
                X, a, words_X = measures[i]
                Y, b, words_Y = measures[j]
                
                U0 = InitialStiefel(300, 2)

                algo = RiemannianBlockCoordinateDescent(eta=reg, tau=tau, max_iter=1000, threshold=0.001,use_gpu=False, verbose=False)

                PRW = ProjectedRobustWasserstein(X, Y, a, b, algo, k)
                PRW.run('RABCD',tau,  U0)

                PRW_matrix[i, j] = PRW.get_value()
                PRW_matrix[j, i] = PRW_matrix[i, j]
                print('RBCD  PRW (', art1, ',', art2, ') =', PRW_matrix[i, j])
                
                
                algo1 = RiemannianGradientAscentSinkhorn(eta=reg, tau=tau/reg, max_iter=1000, threshold=0.001,
                                        sink_threshold=1e-9, use_gpu=False, verbose=True)

                PRW1 = ProjectedRobustWasserstein(X, Y, a, b, algo1, k)
                PRW1.run('RAGAS', tau/reg, U0)

                PRW1_matrix[i, j] = PRW1.get_value()
                PRW1_matrix[j, i] = PRW1_matrix[i, j]
                print('RGAS PRW (', art1, ',', art2, ') =', PRW1_matrix[i, j])
                

if __name__ == '__main__':
    main()

