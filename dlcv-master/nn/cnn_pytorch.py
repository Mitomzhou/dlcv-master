#encoding=utf-8
'''
Created on Apr 1, 2018

@author: mitom
'''

import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torch.optim
import torch.utils.data as data


class CNN(nn.Module):
    """
    CNN for MNIST
    """
    ""
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            #(in_channels, out_channels, kernel_size, stride, padding)                      
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),  
            nn.ReLU(),                  #activation function
            nn.MaxPool2d(2)             #(kernel_size)
        )
        self.conv2 = nn.Sequential(
            #(in_channels, out_channels, kernel_size, stride, padding)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),                  #activation function
            nn.MaxPool2d(2)             #(kernel_size)
        )
        self.out = nn.Linear(32*7*7, 10)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
    
cnn = CNN()
print cnn
#data
train_data = torchvision.datasets.MNIST(root='../data/mnist/', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.MNIST(root='../data/mnist/', train =False)
train_loader = data.DataLoader(dataset=train_data, batch_size=50, shuffle=True)

#test 2000  (2000,28,28) -> (2000,1,28,28) in range(0,1)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255
test_y = test_data.test_labels[:2000]

optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

print '======cnn.parameters======'
parms = list(cnn.parameters())
for name, parameters in cnn.named_parameters():
    print (name , " : " ,parameters.size(), type(parameters))
print '===========end============'

loss_func = nn.CrossEntropyLoss()

for epoch in range(1):
    for step, (x, y) in enumerate(train_loader):
        batch_x = Variable(x)
        batch_y = Variable(y)
        
        output = cnn(batch_x)
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y)*1.0/test_y.size(0)
            
            print('Epoch:', epoch, '|Step:', step,'|train loss:%.4f'%loss.data[0], '|test accuracy:%.4f'%accuracy)
'''       
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print (test_y[:10])
'''
            




