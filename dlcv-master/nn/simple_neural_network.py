'''
Created on Jun 20, 2018

@author: mitom
'''
import numpy as np
from data_utils.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from data_utils.vis_utils import visualize_grid


class SimpleNeuralNetwork(object):
    """
    two layer fully connected neural network
    """
    def __init__(self, input_size, hidden_size, output_size, std = 1e-4):
        """
        Description:
        - init params and net structure
        Inputs:
        - input_size: input image dimension. eg. 32*32*3
        - hidden_size: the number of hidden layers node. eg.50
        - output_size: the number of classification. eg.10
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
    
    def loss(self, X, y, reg):
        """
        Description:
        - get loss and grads by softmax loss function
        - Li=-E(p*logP)  p->(0,1)  P=e^f(W,x)/E(e^f(W,x))
        - Li=-log(e^f(W,x)/E(e^f(W,x)))
        - Loss = (1/n)*E(Li) + 0.5*reg*W*W
        - detail:https://blog.csdn.net/shenxiaoming77/article/details/76858593
        -        https://blog.csdn.net/CV_YOU/article/details/78077514
        Inputs:
        - X: (N, D): (train_num, dim) -> (200, 32*32*3)
        - y: (N,): train_num sample
        - reg: regularization param
        Returns:
        - loss: value
        - grads: a dict of grads
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape
        #loss
        score = None
        X1 = np.maximum(0,np.dot(X, W1) + b1)
        output = np.dot(X1, W2) + b2
        score = output
        e_s = np.exp(score)
        e_s_sum = np.sum(e_s, axis=1)
        e_s_yi = e_s[range(e_s.shape[0]), y]
        Li = - np.log(e_s_yi / e_s_sum)
        reg_loss = np.sum(W1*W1) + np.sum(W2*W2)
        loss = np.mean(Li, axis=0) + 0.5 * reg * reg_loss
        #gradient
        grads = {}
        ones = np.zeros(score.shape)
        for i in range(ones.shape[0]):
            ones[i][y[i]] = 1
        P = (e_s.T / e_s_sum).T
        dout = P - ones
        dW2 = ((dout.T).dot(X1)).T/N + reg*W2
        db2 = (dout.T).dot(np.ones(X1.shape[0]))/N
        dX1 = dout.dot(W2.T)
        dW1 = ((dX1*(X1>0)).T).dot(X).T/N + reg*W1
        db1 = (((dX1*(X1>0)).T).dot(np.ones(X.shape[0]))).T/N
        # get grads
        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W1'] = dW1
        grads['b1'] = db1
        return loss, grads
    
    
    def train(self, X, y, X_val, y_val, learning_rate=1e-3, learning_rate_decay=0.95, reg=0.5,
              num_iters=1000, batch_size=200, isprint=False):
        """
        Description:
        - train date by SGD
        Inputs:
        c
        - X_val: validation set
        - y_val: validation set
        - learning_rate: 
        - learning_rate_decay: decay the learing rate after each epoch
        - reg: regularization strength
        - num_iters: number of iterations
        - batch_size: the batch size each epoch
        - isprint: print result
        Returns:
        - results: a dict incude loss, training accuracy and validation accuracy
        """
        num_train = X.shape[0]
        iter_per_epoch = max(num_train / batch_size, 1)
        # Use SGD
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        if isprint:
            print "=====training accuracy and validation accuracy====="
        for it in xrange(num_iters):
            X_batch_data = None
            y_batch_data = None
            #random extract of batch_size from num_train each time
            # 200->(0~48999)
            batch_data_index = np.random.choice(range(num_train), batch_size)
            X_batch_data = X[batch_data_index, :]
            y_batch_data = y[batch_data_index]
            #compute loss and grads
            loss, grads = self.loss(X_batch_data, y_batch_data, reg)
            loss_history.append(loss)
            #gradients update
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b2'] -= learning_rate * grads['b2']
            #print loss and train-val accuracy 
            if isprint and (it % 100) == 0:
                print 'iteration %d / %d : loss %f' % (it, num_iters, loss)
                #computing training accuracy and validate accuracy
                training_acc = (self.predict(X_batch_data) == y_batch_data).mean()
                validate_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(training_acc)
                val_acc_history.append(validate_acc)
                learning_rate *= learning_rate_decay
                print 'train-val %f   %f' % (training_acc ,validate_acc)
            result={}
            result['loss_history'] = loss_history
            result['train_acc_history'] = train_acc_history
            result['val_acc_history'] = val_acc_history
        return result
            
            
         
    def predict(self, X):
        """
        Description:
        - predict correct classification index 
        Inputs:
        - X: (N, D): (train_num, dim)
        Returns:
        - y_predict: the index of correct classification. 0 <= y_predict < C
        """
        y_predict = None
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape
        X1 = np.maximum(0,np.dot(X, W1) + b1)
        output = np.dot(X1, W2) + b2
        scores = output
        y_predict = np.argmax(scores, axis=1)
        return y_predict
    
    
    def test(self, X, y):
        """
        Description:
        -
        Inputs:
        - X: (N, D): (test_num, dim)
        - y: (N,): number of test sample
        Returns:
        - test_acc: accuracy of test.
        """
        y_test = None
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape
        X1 = np.maximum(0,np.dot(X, W1) + b1)
        output = np.dot(X1, W2) + b2
        scores = output
        y_test = np.argmax(scores, axis=1)
        test_acc = (y_test == y).mean()
        return test_acc
        

def get_CIFAR10_data(num_training=49000, num_validate=1000, num_test=1000):
    """
    Description:
    - load data set
    """
    X_train, y_train, X_test, y_test = load_CIFAR10('../data/cifar-10-batches-py')
    mask = list(range(num_training, num_training+num_validate))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    
    mean = np.mean(X_train, axis=0)
    X_train -= mean
    X_val -= mean
    X_test -= mean
    #(49000,32,32,3) -> (49000,3072)
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validate, -1)
    X_test = X_test.reshape(num_test, -1)
    return X_train, y_train, X_val, y_val, X_test, y_test


def show_loss_accuracy_plot(result):
    plt.subplot(2, 1, 1)
    plt.plot(result['loss_history'])
    plt.title('loss history')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    
    plt.subplot(2, 1, 2)
    plt.plot(result['train_acc_history'], label='train')
    plt.plot(result['val_acc_history'], label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('epoch')
    plt.ylabel('accuracy/%')
    plt.show()
    
    
def show_net_weight(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()
    
    
def get_adjust_params():
    X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
    results = {}
    better_val = -1
    learning_rates = [1e-2, 1e-3, 1e-4]
    reg = [0.4, 0.5, 0.6]
    for lr in learning_rates:
        for r in reg:
            TNET = SimpleNeuralNetwork(32*32*3, 50, 10)
            stats = TNET.train(X_train, y_train, X_val, y_val,
                              num_iters=1000, batch_size=200,
                              learning_rate=lr, learning_rate_decay=0.95, reg=r)
            y_train_pred = TNET.predict(X_train)
            train_acc = (y_train_pred == y_train).mean()
            y_val_pred = TNET.predict(X_val)
            val_acc = (y_val_pred == y_val).mean()
            results[(lr, r)] = (train_acc, val_acc)
            if(better_val < val_acc):
                better_val = val_acc
    for lr, r in sorted(results):
        train_acc, val_acc = results[(lr, r)]
        print ("lr ", lr, " reg", r, " train acc: ", train_acc, " val acc ", val_acc)
    print 'best validation accuracy: ', better_val 
    
    
def main():
    X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
    net = SimpleNeuralNetwork(32*32*3, 50, 10)
    result = net.train(X_train, y_train, X_val, y_val,reg=0.6, learning_rate=1e-3, num_iters=2000, batch_size=200, isprint=True)
    print "=====test accuracy====="
    print net.test(X_test, y_test)
    show_loss_accuracy_plot(result)
    show_net_weight(net)
    

if __name__ == '__main__':
    main()
    #get_adjust_params()
    
    
    
    
