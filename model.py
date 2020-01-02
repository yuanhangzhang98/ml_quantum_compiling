# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 10:27:16 2019

@author: Yuanhang Zhang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    '''
        input: the vector representation of a SU(2) matrix    size:4
        output: the estimated cost-to-go function (number of steps to the target)
        network structure: 2 fc layers, 4 residual blocks and 1 output layer
    '''
    def __init__(self, input_size=8, embedding_size=1000, hidden_size=200, output_size=1):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, embedding_size)
        self.bn1 = nn.BatchNorm1d(embedding_size)
        self.fc2 = nn.Linear(embedding_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.bn6 = nn.BatchNorm1d(hidden_size)
        self.fc7 = nn.Linear(hidden_size, hidden_size)
        self.bn7 = nn.BatchNorm1d(hidden_size)
        self.fc8 = nn.Linear(hidden_size, hidden_size)
        self.bn8 = nn.BatchNorm1d(hidden_size)
        self.fc9 = nn.Linear(hidden_size, hidden_size)
        self.bn9 = nn.BatchNorm1d(hidden_size)
        self.fc10 = nn.Linear(hidden_size, hidden_size)
        self.bn10 = nn.BatchNorm1d(hidden_size)   
        self.fc11 = nn.Linear(hidden_size, hidden_size)
        self.bn11 = nn.BatchNorm1d(hidden_size)   
        self.fc12 = nn.Linear(hidden_size, hidden_size)
        self.bn12 = nn.BatchNorm1d(hidden_size)   
        self.fc13 = nn.Linear(hidden_size, hidden_size)
        self.bn13 = nn.BatchNorm1d(hidden_size)   
        self.fc14 = nn.Linear(hidden_size, hidden_size)
        self.bn14 = nn.BatchNorm1d(hidden_size)   
        self.fc15 = nn.Linear(hidden_size, 1)       

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(x + self.bn4(self.fc4(F.leaky_relu(self.bn3(self.fc3(x))))))
        x = F.leaky_relu(x + self.bn6(self.fc6(F.leaky_relu(self.bn5(self.fc5(x))))))
        x = F.leaky_relu(x + self.bn8(self.fc8(F.leaky_relu(self.bn7(self.fc7(x))))))
        x = F.leaky_relu(x + self.bn10(self.fc10(F.leaky_relu(self.bn9(self.fc9(x))))))
        x = F.leaky_relu(x + self.bn12(self.fc12(F.leaky_relu(self.bn11(self.fc11(x))))))
        x = F.leaky_relu(x + self.bn14(self.fc14(F.leaky_relu(self.bn13(self.fc13(x))))))                
        x = self.fc15(x)
        return x
