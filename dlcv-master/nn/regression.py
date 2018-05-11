'''
Created on May 4, 2018

@author: mitom
'''
import torch 
import torch.optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


"""
An example of simple curve regression
"""
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())
x, y = Variable(x), Variable(y)


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__() 
        self.hidden = nn.Linear(n_feature, n_hidden) 
        self.predict = nn.Linear(n_hidden, n_output) 
    def forward(self, x):
        x = F.relu(self.hidden(x)) 
        x = self.predict(x)
        return x

#print
net = Net(1, 10, 1)
print(net)
print(net)
optimizer = torch.optim.SGD(net.parameters(), lr=0.5) 
loss_function = nn.MSELoss() 
#plt.ion()  
#plt.show()
for t in range(3000):
    prediction = net(x)
    loss = loss_function(prediction, y) 
    optimizer.zero_grad()
    loss.backward() 
    optimizer.step() 
    if (t+1) % 10 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'L=%.4f' % loss.data[0], fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
        