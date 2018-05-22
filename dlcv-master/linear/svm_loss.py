# encoding=utf8 
'''
Created on May 11, 2018

@author: mitom
'''
import numpy as np
from numpy import random as nprand
import cPickle as pickle

def svm_loss_native(W, X, y, reg):
    """
    Inputs:
    - W: (C,D) (classes,dimension)->(10,32*32*3)
    - X: (D,N) (dimension,train_num)->(3072, 100)
    - y: (N,)  train_num N sample，i->(1~N), y[i] right class index, 
                y[i]=c, c->[0,C),and X[i] has label c,score_label = score[y[i]]
    - reg: regularization strength
    Returns:
    - loss_sum: total loss
    Description:
    - loss= 1/N * EE(max(0,W*X-score[y[i]]+1)) + reg*||W*W||
    """
    train_num = X.shape[1]
    train_class_num = W.shape[0]
    loss_i = np.zeros(train_num)
    scores = np.dot(W, X)
    score_label = np.zeros(train_num)
    for i in range(train_num):
        score_label[i] = scores[y[i]][i]
        for j in range(train_class_num):
            if(j != y[i]):
                loss_i[i] = scores[j][i] - score_label[i] + 1
                if(loss_i[i] < 0):
                    loss_i[i] = 0
    loss_sum = sum(loss_i)/train_num + reg*np.sum(W*W)
    return loss_sum

if __name__ == '__main__':
    print 'loss= (1/N) * EE(max(0,W*X-score[y[i]]+1)) + reg*||W*W||'
    np.set_printoptions(precision=2)
    W = nprand.randn(10,3072) * 0.0001
    y = nprand.randint(0,9, size=(100))
    reg = 0.000005
    with open('../data/cifar-10-batches-py/data_batch_1', 'rb') as ftrain:
        datadict = pickle.load(ftrain)
        X = datadict['data'][:100]
        X = X.astype('int64').T
    print 'W:', W.shape
    print 'X:', X.shape
    print 'y:', y.shape
    print 'reg:', reg
    loss = svm_loss_native(W, X, y, reg)
    print 'loss:', loss
        
        
        
        