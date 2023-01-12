import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchinfo
import argparse
import numpy as np
from torch.autograd import Variable


class MLP_CNN_LSTM(nn.Module):
    def __init__(self, input_dim):
        super(MLP_CNN_LSTM, self).__init__()

        #mlp
        mlp_layers = [2048, 1024, 512, 256]
        self.mlp = nn.Sequential()
        for i in range(len(mlp_layers)):
            if i == 0:
                self.mlp.add_module('mlp_' + str(i+1), nn.Linear(input_dim, mlp_layers[i]))        
            else:
                self.mlp.add_module('mlp_' + str(i+1), nn.Linear(mlp_layers[i-1], mlp_layers[i]))
        
        #cnn
        self.cnn = nn.Sequential()
        cnn_kernel_size = [8, 8, 8]
        cnn_layers = [256, 128, 64]
        pooling_kernel_size = [2, 2, 2]
        for i in range(len(cnn_kernel_size)):
            if i == 0:  
                self.cnn.add_module('conv_' + str(i),
                                      nn.Conv1d(1, cnn_layers[i], kernel_size=cnn_kernel_size[i]))
            else:
                self.cnn.add_module('conv_' + str(i),
                                      nn.Conv1d(cnn_layers[i-1], cnn_layers[i],
                                      kernel_size=cnn_kernel_size[i]))
            
            self.cnn.add_module('maxpool_' + str(i), nn.MaxPool1d(kernel_size=pooling_kernel_size[i]))

            self.cnn.add_module('batchnorm_' + str(i), nn.BatchNorm1d(cnn_layers[i]))

            self.cnn.add_module('relu_' + str(i), nn.ReLU())

            self.cnn.add_module('dropout_' + str(i), nn.Dropout(0.2))
        
        #lstm
        self.lstm = nn.LSTM(cnn_layers[-1], 32, 1, batch_first=True, bidirectional=True)

        #last
        self.fc = nn.Sequential()
        self.fc.add_module('flatten', nn.Flatten())
        self.fc.add_module('fc_1', nn.Linear(1600, 256))
        self.fc.add_module('fc_2', nn.Linear(256, 64))
        self.fc.add_module('fc_3', nn.Linear(64, 1))
    
    def forward(self, x):
        h0 = Variable(torch.zeros(1 * 2, x.shape[0], 32))
        c0 = Variable(torch.zeros(1 * 2, x.shape[0], 32))
        x = self.mlp(x)
        x = x.reshape((x.shape[0], -1, x.shape[1]))
        x = self.cnn(x)
        x = x.transpose(1,2)
        x, _ = self.lstm(x, (h0, c0)) 
        x = self.fc(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()

        #mlp for A and D
        MLP_layers = [2048, 1024, 512, 256, 128, 64, 32, 16, 1]
        self.MLP = nn.Sequential()
        for i in range(len(MLP_layers)):
            if i == 0:
                self.MLP.add_module('mlp_' + str(i+1), nn.Linear(input_dim, MLP_layers[i]))
                self.MLP.add_module('relu_' + str(i+1), nn.ReLU())
            else:
                self.MLP.add_module('mlp_' + str(i+1), nn.Linear(MLP_layers[i-1], MLP_layers[i]))
                self.MLP.add_module('relu_' + str(i+1), nn.ReLU())


    def forward(self, x):
        x = self.MLP(x)
        return x



