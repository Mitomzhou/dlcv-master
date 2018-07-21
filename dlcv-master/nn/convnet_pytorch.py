'''
Created on Jul 17, 2018

@author: mitom
'''


import torchvision as tv
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import time

from torch.autograd import Variable


def get_data():
    train_set = tv.datasets.CIFAR10(root='../data/cifar-10/', train=True, 
                                         transform=tv.transforms.ToTensor(), download=False)
    test_set = tv.datasets.CIFAR10(root='../data/cifar-10/', train=False, 
                                        transform=tv.transforms.ToTensor(), download=False)
    train_loader = data.DataLoader(dataset=train_set, batch_size=100, shuffle=True)
    test_loader = data.DataLoader(dataset=test_set, batch_size=1000, shuffle=False)
    for testdata in test_loader:
        test_X, test_y = testdata
        break
    return train_loader, test_loader, test_X, test_y


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU()
            )
        self.fullcon1 = nn.Linear(16*7*7, 100)
        self.fullcon2 = nn.Linear(100, 10)
         
    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = X.view(X.size(0), -1)
        X = self.fullcon1(X)
        X = F.relu(X)
        out = self.fullcon2(X)
        return out
    
    
def main():
    start_time = time.time()
    net = ConvNet()
    # print net and params
    print net
    for name, parameters in net.named_parameters():
        print (name , " : " ,parameters.size(), type(parameters))
    epoch = 10
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    #loss function
    loss_func = nn.CrossEntropyLoss()
    train_loader, test_loader, test_X, test_y = get_data()
    for iter in range(epoch):
        for i, (X, y) in enumerate(train_loader):
            batch_X = Variable(X)
            batch_y = Variable(y)
            output = net(batch_X)
            loss = loss_func(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct = 0
            total = 0
            if i % 100 == 0:
                outputs = net(Variable(test_X))
                _, pred_y = torch.max(outputs.data, 1)
                total += test_y.size(0)
                correct += (pred_y == test_y).sum()
                print str(iter) + ": train loss=%.4f"%loss.data[0] + " test_acc:%.4f"%(correct/float(total))
        # lr decay
        #if (epoch + 1) % 20 == 0:
        #    lr /= 3
        #    optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)
    total_correct = 0
    total_num = 0
    for testdata in test_loader:
        test_X, test_y = testdata
        outputs = net(Variable(test_X))
        _, pred_y = torch.max(outputs.data, 1)
        total_num += test_y.size(0)
        total_correct += (pred_y == test_y).sum()
    end_time = time.time()
    print "time: " + str((end_time-start_time)) +" ********* test_acc:%.4f"%(total_correct/float(total_num))
    
    
if __name__ == '__main__':
    main()

