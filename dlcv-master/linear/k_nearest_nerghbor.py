'''
Created on Apr 26, 2018

@author: Mitom Zhou
'''

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle


def load_data(root='../cifar-10-batches-py'):
    """
    Inputs:
    - root: data input path
    Returns:
    - X_train: training data
    - Y_train: training labels
    - X_test: test data
    - Y_test: test label
    Description:
    - data download: http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    """
    with open('../data/cifar-10-batches-py/data_batch_1', 'rb') as ftrain:
        datadict = pickle.load(ftrain)
        X = datadict['data'][:5000]
        Y = datadict['labels'][:5000]
        X_train = X.astype('int64')
        Y_train = np.array(Y)
    with open('../data/cifar-10-batches-py/test_batch', 'rb') as ftest:
        datadict = pickle.load(ftest)
        X = datadict['data'][:500]
        Y = datadict['labels'][:500]
        X_test = X.astype('int64')
        Y_test = np.array(Y)
    return X_train, Y_train, X_test, Y_test
    

class KNearestNeighbor(object):
    """
    K-Nearest Neighbor
    """
    def __init__(self):
        pass
   
    
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
        
        
    def predict(self, X, k=5):
        """
        Inputs:
        - X: input test data
        - k: the number of most nearest elements
        Returns:
        - distance: the distance between the training data and test data
        """
        num_input = X.shape[0]
        num_train = self.X_train.shape[0]
        print('distance mat:',num_input, num_train)
        distance = np.zeros([num_input, num_train])
        X_X_sum = np.sum(X**2, axis=1)
        train_X_X_sum = np.sum(self.X_train**2, axis=1)
        #euclidean metric
        distance = np.sqrt(np.matrix(X_X_sum).T + np.matrix(train_X_X_sum) - 2*np.matrix(np.dot(X, self.X_train.T)))
        distance = np.array(distance)
        return distance
        
        
    def predict_labels(self, distance, k=5):
        num_test = distance.shape[0]
        y_pred = np.zeros(num_test)
        for i in xrange(num_test):
            closest_y = []
            #the most nearest k elements
            closest_y = self.y_train[np.argsort(distance[i])[:k]]
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred
    
    
if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load_data()
    print('X_train:', X_train.shape)
    print('Y_train:', Y_train.shape)
    print('X_test:', X_test.shape)
    print('Y_test:', Y_test.shape)
    knn = KNearestNeighbor()
    knn.train(X_train, Y_train) 
    distance = knn.predict(X_test)
    plt.imshow(distance, interpolation='none')
    y_pred = knn.predict_labels(distance)
    #plt.show()
    num_correct = np.sum(Y_test == y_pred)
    accuracy = num_correct*1.0/X_test.shape[0]
    print('accuracy:', accuracy)
  

   
    
    