# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:05:45 2019

@author: Yuanhang Zhang
"""

import numpy as np
import torch
from torch.utils import data
from dataGenerator import DataGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestDataset(data.Dataset):
    def __init__(self, env, cur_length, full_dataset_length, max_length, num_samples, epsilon):
        self.env = env
        self.cur_length = cur_length
        self.max_length = max_length
        self.num_samples = num_samples
        self.epsilon = epsilon
        self.full_dataset_length = full_dataset_length

        self.generator = DataGenerator(env, epsilon)
        self.states_full, self.actions, _, _ = self.generator.calc_data_full(self.full_dataset_length)
        self.reinitialize()
            
    def reinitialize(self):
        n = self.cur_length - self.full_dataset_length
        if n > 0:
            self.states_rand, _, _ = self.generator.calc_data_rand(self.states_full[-1].view(-1, 2, 2, 2), self.actions, n)
                
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, _):
        length = torch.randint(0, self.cur_length, ())
        if length < self.full_dataset_length:
            idx = torch.randint(0, len(self.states_full[length]), ())
            state = self.states_full[length][idx]
            return {'state': state, 'length': length+1}
        else:
            length = length - self.full_dataset_length
            idx = torch.randint(0, len(self.states_rand[length]), ())
            state = self.states_rand[length][idx]     
            return {'state': state, 'length': length+self.full_dataset_length+1}   
        