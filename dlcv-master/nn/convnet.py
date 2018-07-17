'''
Created on Jul 3, 2018

@author: mitom
'''

import numpy as np
import xlwt 
import time

from data_utils.data_utils import load_CIFAR10


def get_CIFAR10_data(num_training=49000, num_validate=1000, num_test=1000):
    """
    Description:
    - load data set
    Returns:
    - X_train: (N,C,H, W)
    - y_train: (y,)
    - X_val: (N,C,H,W)
    - y_val: (y,)
    - X_test:(N,C,H,W)
    - y_test: (y,)
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
    X_train = np.transpose(X_train,[0,3,1,2])
    X_val = np.transpose(X_val, [0,3,1,2])
    X_test = np.transpose(X_test, [0,3,1,2])
    return X_train, y_train, X_val, y_val, X_test, y_test


def save_data(data, filename):
    """
    Description:
    - data saved to excel file
    Inputs:
    - filename: /root/filename.ods
    """
    if not isinstance(data, np.ndarray):
        print "data type error, save failed!"
        return 
    xl_file = xlwt.Workbook()
    sheet = xl_file.add_sheet('data', cell_overwrite_ok=True)
    if len(data.shape)==4:
        # X(N, C, H, W)
        N, C, H, W = data.shape
        for n in range(N):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        #row = h +c*H   c = n*C + c -> row = h + (n*C+c)*H
                        sheet.write(h+(n*C+c)*H, 0, n)
                        sheet.write(h+(n*C+c)*H, 1, c)
                        sheet.write(h+(n*C+c)*H, w+2, data[n, c, h, w])
    if len(data.shape)==2:
        N, D = data.shape
        for d in range(D):
            for n in range(N):
                sheet.write(d, n, data[n, d])
    if len(data.shape)==1:
        D = data.shape[0]
        for d in range(D):
            sheet.write(d, 0, data[d])
    xl_file.save("/home/mitom/Desktop/save_data/" + filename)
    print "save " + filename + " done!"

    
class ConvNet(object):
    """
    a simple net for classifying CIFAR-10
    """
    def __init__(self):
        self.layers = []
        
    def addLayer(self, layer):
        self.layers.append(layer)
        
    def train(self, X, y, X_val, y_val, num_iters, batch_size):#100, 100
        train_num = X.shape[0]
        rand = batch_size*10
        for iter in range(num_iters):
            X_batch_data = None
            y_batch_data = None
            # 1000 -> (0~10000)
            batch_data_index = np.random.choice(range(train_num), rand)
            X_batch_data = X[batch_data_index]
            y_batch_data = y[batch_data_index]
            #learning_rate decrease
            #  to do...
            print("===== iter=" + str(iter) + " =====") 
            for batch_iter in range(0, rand, batch_size): # 0, 100, 200,...,900
                if(batch_iter + batch_size) < rand:
                    loss = self.train_inner(X_batch_data[batch_iter:batch_iter+batch_size], 
                                            y_batch_data[batch_iter:batch_iter+batch_size])
                else:
                    loss = self.train_inner(X_batch_data[batch_iter:rand], 
                                            y_batch_data[batch_iter:rand])
                print(str(time.clock()) + " ~ " + str(batch_iter) + "/" + str(rand) + "    loss: " + str(loss)) 
            # get evaluate accuracy
            eval_acc = self.evaluate(X_val, y_val)
            print(str(time.clock()) + " === evaluate accuracy: " + str(eval_acc))
    
    def train_inner(self, batch_data, batch_label):
        layer_num = len(self.layers)
        in_data = batch_data
        for i in range(layer_num):
            out_data = self.layers[i].forward(in_data)
            in_data = out_data
        N = out_data.shape[0]
        # softmax loss function
        loss = -np.sum(np.log(out_data[range(N), list(batch_label)]))
        loss /= N
        # compute residual
        residual_in = batch_label
        for i in range(0, layer_num):
            residual_out = self.layers[layer_num-i-1].backward(residual_in)
            residual_in = residual_out
        return loss
            
    def evaluate(self, eval_data, eval_label):
        layer_num = len(self.layers)
        in_data = eval_data
        for i in range(layer_num):
            out_data = self.layers[i].forward(in_data)
            in_data = out_data
        y_predict = np.argmax(out_data, axis=1)  
        eval_acc = (y_predict == eval_label).mean()
        return eval_acc 
    
    def test(self, test_data, test_label):
        layer_num = len(self.layers)
        in_data = test_data
        for i in range(layer_num):
            out_data = self.layers[i].forward(in_data)
            in_data = out_data
        y_predict = np.argmax(out_data, axis=1)  
        test_acc = (y_predict == test_label).mean()
        print "************ test result *************"
        print test_acc
        return test_acc


class ConvLayer(object):
    """
    Convolution operation
    """
    def __init__(self, input_channal, output_channal, kernel_size, stride=1, pad=1, learning_rate=1e-4, reg=0.75, name='Convolution-Layer'):
        self.input_channal = input_channal
        self.output_channal = output_channal
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.learning_rate = learning_rate
        self.reg = reg
        self.name = name
        # W -> (F, C, H, W)  filter_num  channel
        self.W = np.random.randn(output_channal, input_channal, kernel_size, kernel_size)
        self.b = np.random.randn(output_channal)
        self.prev_gradient_W = np.zeros_like(self.W)
        self.prev_gradient_b = np.zeros_like(self.b)
    
    def forward(self, X):
        # W(F,C,H,W)
        W_F, W_C, W_H, W_W = self.W.shape
        b_F = self.b.shape
        # X(N,C,H,W)
        X_N, X_C, X_H, X_W = X.shape
        pad, kernel_size, stride = self.pad, self.kernel_size, self.stride
        # out_H = 1 + (X_H + 2*pad - kernel_size)/stride
        out_H = int(1 + (X_H + 2*pad - kernel_size)/stride)
        out_W = int(1 + (X_W + 2*pad - kernel_size)/stride)
        # out(N, F, H, W)
        self.out = np.zeros((X_N, W_F, out_H, out_W,))
        """
        #slowly why? many for?
        for n in range(X_N):
            print "n = " + str(n)
            for f in range(W_F):
                # add bias
                conv_out_H_W = np.ones((out_H, out_W)) * self.b[f]
                for c in range(X_C):
                    X_peded = np.lib.pad(X[n, c, :, :],pad_width=self.pad, mode='constant', constant_values=0)
                    #print "X_peded: ",X_peded.shape 
                    for i in range(out_H):
                        for j in range(out_W):
                            conv_out_H_W[i, j] += np.sum(X_peded[i*self.stride : i*self.stride+W_H,  
                                                                 j*self.stride : j*self.stride+W_W] 
                                                                 * self.W[f, c, :, :])
                    self.out[n, f, :, :] =  conv_out_H_W
        """
        X_peded = np.pad(X, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
        for i in range(out_H):
            for j in range(out_W):
                in_data_pad_masked = X_peded[:, :, i*self.stride : i*self.stride+W_H,  
                                                   j*self.stride : j*self.stride+W_W]
                for f in range(W_F):
                    # X*W
                    self.out[:, f , i, j] = np.sum(in_data_pad_masked * self.W[f, :, :, :], axis=(1,2,3))
        #add bias
        for n in range(X_N):
            for f in range(W_F):
                self.out[n, f, :, :] += self.b[f]
        # for backward
        self.bottom_val = X
        return self.out
        
    def backward(self, residual):
        X_N, X_C, X_H, X_W = self.bottom_val.shape
        W_F, W_C, W_H, W_W = self.W.shape
        pad, kernel_size, stride = self.pad, self.kernel_size, self.stride
        out_H = int(1 + (X_H + 2*pad - kernel_size)/stride)
        out_W = int(1 + (X_W + 2*pad - kernel_size)/stride)
        # add 0
        X_ped = np.pad(self.bottom_val, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
        dx = np.zeros_like(self.bottom_val)
        dx_pad = np.zeros_like(X_ped)
        dw = np.zeros_like(self.W)
        db = np.sum(residual, axis=(0, 2, 3))
        for i in range(out_H):
            for j in range(out_W):
                X_pad_masked = X_ped[:, :, 
                                     i*stride : i*stride+W_H, 
                                     j*stride : j*stride+W_W]
                #compute dw
                for f in range(W_F):
                    dw[f, :, :, :] += np.sum(X_pad_masked * (residual[:, f, i, j])[:, None, None, None], axis=0)
                #compute dx_pad
                for n in range(X_N):
                    temp_W = np.rot90(self.W ,2, (2, 3)) #180 
                    dx_pad[n, :, i*stride : i*stride+W_H, j*stride : j*stride+W_W] += np.sum((self.W[:, :, :, :] * (residual[n, :, i, j])[:, None, None, None]), axis=0)
        dx[:, :, :, :] = dx_pad[:, :, pad:-pad, pad:-pad]
        self.W -= self.learning_rate * (dw + self.prev_gradient_W * self.reg)
        self.b -= self.learning_rate * db
        self.prev_gradient_W = self.W
        return dx
        
    
class MaxPooling(object):    
    """
    """
    def __init__(self, kernel_size, stride, name='MaxPooling-Layer'):
        self.kernel_size = kernel_size
        self.stride = stride
        self.name = name
    
    def forward(self, X):
        X_N, X_C, X_H, X_W = X.shape
        out_H = int((X_H - self.kernel_size)/self.stride + 1)
        out_W = int((X_W - self.kernel_size)/self.stride + 1)
        self.out = np.zeros((X_N, X_C, out_H, out_W))
        for n in range(X_N):
            for c in range(X_C):
                pooling_out_H_W = np.zeros((out_H, out_W))
                X_H_W_tensor = X[n, c, :, :]
                for i in range(out_H):
                    for j in range(out_W):
                        pooling_out_H_W[i, j] = np.max(X_H_W_tensor[i*self.stride : i*self.stride+self.kernel_size,
                                                                    j*self.stride : j*self.stride+self.kernel_size])
                self.out[n, c, :, :] = pooling_out_H_W
        self.bottom_val = X
        return self.out
    
    def backward(self, residual):
        X_N, X_C, X_H, X_W = residual.shape
        kernel_size, stride = self.kernel_size, self.stride
        out_H = int((X_H - kernel_size)/stride + 1)
        out_W = int((X_W - kernel_size)/stride + 1)
        dx = np.zeros_like(self.bottom_val)
        for i in range(out_H):
            for j in range(out_W):
                x_masked = self.bottom_val[:, :, i*stride : i*stride+kernel_size, 
                                           j*stride : j*stride+kernel_size]
                max_x_masked = np.max(x_masked, axis=(2, 3))
                temp_binary_mask = (x_masked == (max_x_masked)[:, :, None, None])
                dx[:, :, i*stride : i*stride+kernel_size, 
                                           j*stride : j*stride+kernel_size] += temp_binary_mask * (residual[:, :, i, j])[:, :, None, None]
        return dx
        
        

class ReLU(object):
    """
    activation function
    """
    def __init__(self, name='ReLU-Layer'):
        self.name = name
    
    def forward(self, X):
        self.out = np.maximum(X, 0)
        # for backward 
        self.top_val = X
        return self.out
        
    def backward(self, residual):
        return (self.top_val > 0) * residual   
        

class Flatten(object):
    """
    drop the four-dimensional data to two-dimensional input for the full connection layer
    """
    def __init__(self, name='Flatten-Layer'):
        self.name = name
    
    def forward(self, X):
        self.X_N, self.X_C, self.X_H, self.X_W = X.shape
        self.out = X.reshape(self.X_N, self.X_H*self.X_W*self.X_C)
        return self.out
    
    def backward(self, residual):
        return residual.reshape(self.X_N, self.X_C, self.X_H, self.X_W)
        

class FullConnection(object):
    """
    """
    def __init__(self, input_size, output_size, std=1e-4, learning_rate=1e-4, reg=0.75, name='FullConnection-Layer'):
        self.input_size = input_size
        self.output_size = output_size
        self.std = std
        self.learning_rate = learning_rate
        self.reg = reg
        self.W = std * np.random.randn(input_size, output_size)
        self.b = std * np.zeros(output_size)
        self.prev_grad_W = np.zeros_like(self.W)
        self.prev_grad_b = np.zeros_like(self.b)
        self.name = name
    
    def forward(self, X):
        self.out = X.dot(self.W) + self.b
        # for backward
        self.bottom_val = X
        return self.out
    
    def backward(self, loss):
        residual_x = loss.dot(self.W.T)
        self.W -= self.learning_rate * (self.bottom_val.T.dot(loss) + self.prev_grad_W * self.reg)
        self.b -= self.learning_rate * (np.sum(loss, axis=0))
        self.prev_grad_W = self.W
        self.prev_grad_b = self.b
        return residual_x
 
 
class Softmax(object):
    """
    """
    def __init__(self, name='Softmax-Layer'):
        self.name = name
        
    def forward(self, X):
        score = X - np.max(X, axis=1).reshape(-1, 1)
        self.out = np.exp(score)/np.sum(np.exp(score), axis=1).reshape(-1, 1)
        return self.out
    
    def backward(self, residual):
       X_N = residual.shape[0]
       dscores = self.out.copy()
       dscores[range(X_N), list(residual)] -= 1
       dscores /= X_N
       return dscores
   
   
def main():
    rate = 1e-5
    net = ConvNet()
    net.addLayer(ConvLayer(3, 32, 5, learning_rate=rate))       #(N, 3, 32, 32) -> (N, 32, 30, 30)
    net.addLayer(MaxPooling(2, 2))                              #(N, 32, 30, 30) -> (N, 32, 15, 15)
    net.addLayer(ReLU())
    net.addLayer(ConvLayer(32, 16, 3, learning_rate=rate))      #(N, 32, 15, 15) -> (N, 16, 15, 15)
    net.addLayer(MaxPooling(3, 2))                              #(N, 16, 15, 15) -> (N, 16, 7, 7)
    net.addLayer(ReLU())
    net.addLayer(Flatten())                                     #(N, 16, 7, 7) -> (N, 16*7*7) 
    net.addLayer(FullConnection(16*7*7, 100, learning_rate=rate))   #(N, 16*7*7) -> (N, 100) 
    net.addLayer(ReLU())
    net.addLayer(FullConnection(100, 10, learning_rate=rate))       #(N, 100) -> (N, 10)
    net.addLayer(Softmax())
    print "net init done!"
    X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
    net.train(X_train[0:10000], y_train[0:10000], X_val[0:1000], y_val[0:1000], 100, 100)
    net.test(X_test[0:1000], y_test[0:1000])

    
if __name__ == '__main__':
    main()

    