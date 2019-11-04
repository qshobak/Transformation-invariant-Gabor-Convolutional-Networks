from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from GaborLayer import GaborConv, GaborConvSca, GaborYu
from utils import get_parameters_size

class Gabor_CNN(nn.Module):
    def __init__(self, channel=8):
        super(Gabor_CNN, self).__init__()
        self.channel = channel
        self.model = nn.Sequential( #input shape (1,32,32)
            GaborConv(in_channels=1, out_channels=10*channel, kernel_size=3, padding=0, stride=1, bias=False),
            #nn.BatchNorm2d(10*channel),
            #nn.Conv2d(in_channels=1, out_channels=10*channel, kernel_size=3, padding=0, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(in_channels=10*channel, out_channels=20*channel, kernel_size=3, padding=0, stride=1, bias=False),
            #nn.BatchNorm2d(20*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(in_channels=20*channel, out_channels=40*channel, kernel_size=3, padding=1, stride=1, bias=False),
            #nn.BatchNorm2d(40*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(in_channels=40*channel, out_channels=80*channel, kernel_size=3, padding=0, stride=1, bias=False),
            #nn.BatchNorm2d(80*channel),
            nn.ReLU(inplace=True),       
        )
        
        self.fc1 = nn.Linear(80*channel, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 10)
        
    def forward(self, x):
        x = self.model(x)
        #print(x.size())    
        #x = x.view(-1, 80, self.channel)
        #x = torch.max(x, 2)[0]
        x = x.view(-1, 80*self.channel)
        x = self.fc1(x)
        fea = x
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        out = x
        return out, fea

class Gabor_CNN5(nn.Module):
    def __init__(self, channel=8):
        super(Gabor_CNN5, self).__init__()
        self.channel = channel
        self.first = nn.Sequential( #input shape (1,32,32)
            GaborConv(in_channels=1, out_channels=10*channel, kernel_size=5, padding=2, stride=1, bias=False),
            nn.BatchNorm2d(10*channel),
            #nn.Conv2d(in_channels=1, out_channels=10*channel, kernel_size=3, padding=0, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.model = nn.Sequential( 
            GaborConv(in_channels=10*channel, out_channels=20*channel, kernel_size=5, padding=2, stride=1, bias=False),
            #nn.Conv2d(in_channels=10*channel, out_channels=20*channel, kernel_size=5, padding=2, stride=1, bias=False),
            nn.BatchNorm2d(20*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            GaborConv(in_channels=20*channel, out_channels=40*channel, kernel_size=5, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(40*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            GaborConv(in_channels=40*channel, out_channels=80*channel, kernel_size=5, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(80*channel),
            nn.ReLU(inplace=True), 
        )
        
        self.fc1 = nn.Linear(80*channel, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 10)
        
    def forward(self, x):
        x = self.model(self.first(x))
        #print(x.size())    
        #x = x.view(-1, 80, self.channel)
        #x = torch.max(x, 2)[0]
        x = x.view(-1, 80*self.channel)
        x = self.fc1(x)
        fea = x
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        out = x
        return out, fea

class Gabor_CNN7(nn.Module):
    def __init__(self, channel=8):
        super(Gabor_CNN7, self).__init__()
        self.channel = channel
        self.model = nn.Sequential( #input shape (1,32,32)
            GaborConv(in_channels=1, out_channels=10*channel, kernel_size=7, padding=3, stride=1, bias=False),
            #nn.BatchNorm2d(10*channel),
            #nn.Conv2d(in_channels=1, out_channels=10*channel, kernel_size=3, padding=0, stride=1, bias=False),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(2,2),

            nn.Conv2d(in_channels=10*channel, out_channels=20*channel, kernel_size=7, padding=3, stride=1, bias=False),
            #nn.BatchNorm2d(20*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(in_channels=20*channel, out_channels=40*channel, kernel_size=7, padding=3, stride=1, bias=False),
            #nn.BatchNorm2d(40*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(in_channels=40*channel, out_channels=80*channel, kernel_size=7, padding=0, stride=1, bias=False),
            #nn.BatchNorm2d(80*channel),
            nn.ReLU(inplace=True),       
        )
        
        self.fc1 = nn.Linear(80*channel, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 10)
        
    def forward(self, x):
        x = self.model(x)
        #print(x.size())    
        #x = x.view(-1, 80, self.channel)
        #x = torch.max(x, 2)[0]
        x = x.view(-1, 80*self.channel)
        x = self.fc1(x)
        fea = x
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        out = x
        return out, fea

class Gabor_CNN7New(nn.Module):
    def __init__(self, channel=8):
        super(Gabor_CNN7New, self).__init__()
        self.channel = channel
        self.model = nn.Sequential( #input shape (1,32,32)
            GaborConv(in_channels=1, out_channels=10*channel, kernel_size=7, padding=3, stride=1, bias=False),
            #nn.BatchNorm2d(10*channel),
            #nn.Conv2d(in_channels=1, out_channels=10*channel, kernel_size=3, padding=0, stride=1, bias=False),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(2,2),

            nn.Conv2d(in_channels=10*channel, out_channels=20*channel, kernel_size=7, padding=3, stride=1, bias=False),
            #nn.BatchNorm2d(20*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(in_channels=20*channel, out_channels=40*channel, kernel_size=7, padding=2, stride=1, bias=False),
            #nn.BatchNorm2d(40*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(in_channels=40*channel, out_channels=80*channel, kernel_size=7, padding=0, stride=1, bias=False),
            #nn.BatchNorm2d(80*channel),
            nn.ReLU(inplace=True),       
        )
        
        self.fc1 = nn.Linear(80*channel, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 10)
        
    def forward(self, x):
        x = self.model(x)
        #print(x.size())    
        #x = x.view(-1, 80, self.channel)
        #x = torch.max(x, 2)[0]
        x = x.view(-1, 80*self.channel)
        x = self.fc1(x)
        fea = x
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        out = x
        return out, fea

class Base_CNN(nn.Module):
    def __init__(self, channel=8):
        super(Base_CNN, self).__init__()
        self.channel = channel
        self.model = nn.Sequential( #input shape (1,32,32) for kernel = 3
            nn.Conv2d(in_channels=1, out_channels=10*channel, kernel_size=3, padding=0, stride=1, bias=False),
            #nn.BatchNorm2d(10*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(in_channels=10*channel, out_channels=20*channel, kernel_size=3, padding=0, stride=1, bias=False),
            #nn.BatchNorm2d(20*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(in_channels=20*channel, out_channels=40*channel, kernel_size=3, padding=1, stride=1, bias=False),
            #nn.BatchNorm2d(40*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(in_channels=40*channel, out_channels=80*channel, kernel_size=3, padding=0, stride=1, bias=False),
            #nn.BatchNorm2d(80*channel),
            nn.ReLU(inplace=True),      
        )
        
        self.fc1 = nn.Linear(80*channel, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 10)
        
    def forward(self, x):
        x = self.model(x)
        #print(x.size())    
        #x = x.view(-1, 80, self.channel)
        #x = torch.max(x, 2)[0]
        x = x.view(-1, 80*self.channel)
        x = self.fc1(x)
        fea = x
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        out = x
        return out, fea

class Gabor_Scale(nn.Module):
    def __init__(self, channel=8):
        super(Gabor_Scale, self).__init__()
        self.channel = channel
        self.model = nn.Sequential( #input shape (1,32,32)
            GaborConvSca(in_channels=1, out_channels=12, kernel_size=7, padding=3, stride=1, bias=False),
            nn.BatchNorm2d(12),
            #nn.Conv2d(in_channels=1, out_channels=10*channel, kernel_size=3, padding=0, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=7, padding=3, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=7, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(2,2),

            #nn.Conv2d(in_channels=40*channel, out_channels=80*channel, kernel_size=7, padding=0, stride=1, bias=False),
            #nn.BatchNorm2d(80*channel),
            #nn.ReLU(inplace=True),       
        )
        
        self.fc1 = nn.Linear(48, 256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.model(x)
        #print(x.size())    
        #x = x.view(-1, 80, self.channel)
        #x = torch.max(x, 2)[0]
        x = x.view(-1, 48)
        x = self.fc1(x)
        fea = x
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        out = x
        #print(x.size())
        return out, fea

class Gabor_Yu(nn.Module):
    def __init__(self, channel=2):
        super(Gabor_Yu, self).__init__()
        self.channel = channel
        self.gaborYu = nn.Sequential(
            GaborYu(in_channels=1, out_channels=4, kernel_size=3, padding=1, stride=1, bias=False), #128*8,1,32,32
            #nn.BatchNorm2d(1),
            #nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1, out_channels=10*channel, kernel_size=3, padding=0, stride=1, bias=False),  #128*8,20,30,30
            )
        self.model = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),  #128*8,20,15,15
            nn.Conv2d(in_channels=10*channel, out_channels=20*channel, kernel_size=3, padding=0, stride=1, bias=False),
            #nn.BatchNorm2d(20*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(in_channels=20*channel, out_channels=40*channel, kernel_size=3, padding=1, stride=1, bias=False),
            #nn.BatchNorm2d(40*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(in_channels=40*channel, out_channels=80*channel, kernel_size=3, padding=0, stride=1, bias=False),
            #nn.BatchNorm2d(80*channel),
            nn.ReLU(inplace=True),       
        )
        
        self.fc1 = nn.Linear(80*channel, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 10)
        
    def forward(self, x):   #128,1,32,32
        xGabor = self.gaborYu(x)    #128*8,20,30,30
        xConv = xGabor.view(x.size(0)*xGabor.size(1), -1, xGabor.size(2), xGabor.size(3)) #128*20,8,30,30
        xNew = torch.max(xConv, 1)[0]  #128*20,30,30
        xNew = torch.unsqueeze(xNew, 1) #128*20,1,30,30
        xNew = xNew.view(x.size(0), -1, xNew.size(2), xNew.size(3)) #128,20,30,30
        x = self.model(xNew)
        #print(x.size())    
        #x = x.view(-1, 80, self.channel)
        #x = torch.max(x, 2)[0]
        x = x.view(-1, 80*self.channel)
        x = self.fc1(x)
        fea = x
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        out = x
        return out, fea

def get_network_fn(name):
    networks_zoo = {
    'gaborCNN': Gabor_CNN(channel=8),
    'gaborCNN10': Gabor_CNN(channel=1),
    'gaborCNN20': Gabor_CNN(channel=2),
    'gaborCNN40': Gabor_CNN(channel=4),
    'gaborCNN51': Gabor_CNN5(channel=1),
    'gaborCNN52': Gabor_CNN5(channel=2),
    'gaborCNN54': Gabor_CNN5(channel=4),
    'gaborCNN72': Gabor_CNN7(channel=2),
    'gaborCNN74': Gabor_CNN7(channel=4),
    'gaborCNN72New': Gabor_CNN7New(channel=2),
    'gaborCNN74New': Gabor_CNN7New(channel=4),
    'gaborScale': Gabor_Scale(channel=2),
    'basecnn':Base_CNN(channel=8),
    'basecnn2':Base_CNN(channel=2),
    'basecnn4':Base_CNN(channel=4),
    'yuGabor': Gabor_Yu(channel=2),
    }
    if name is '':
        raise ValueError('Specify the network to train. All networks available:{}'.format(networks_zoo.keys()))
    elif name not in networks_zoo:
        raise ValueError('Name of network unknown {}. All networks available:{}'.format(name, networks_zoo.keys()))
    return networks_zoo[name]

def test():
    a = torch.randn(128,1,32,32).cuda()
    model = get_network_fn('yuGabor')
    model = model.cuda()
    #model = get_network_fn('gaborCNN')
    print(get_parameters_size(model)/1e6)
    #print(model)
    b = model(a)
    print(b[0].size())

if __name__ == '__main__':
    test()
