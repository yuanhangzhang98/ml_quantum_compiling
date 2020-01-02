# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:26:13 2019

@author: Yuanhang Zhang
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from tqdm import trange
import torch
from system import System

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataGenerator():
    def __init__(self, env, epsilon):
        self.env = env
        self.U = env.U
        self.scramble_table = self.env.scramble_table
        self.epsilon = epsilon
        self.init_batch = 2
        # using the quaternion distance measure, distance approximates theta/2
        # init_theta = 0.01 corresponds to a maximum initial distance of 0.005
        self.init_theta = epsilon 


    @torch.no_grad()
    def calc_data_full(self, n):
        states_list = []
        next_states_list = []
        masks_list = []
        
        states = self.env.randRotation(self.init_theta, self.init_batch)
        states = self.env.einsum('aij, bjk->baik', self.U, states).view(-1, 2, 2, 2)
        actions = torch.arange(self.env.n_actions, device=device)\
                        .expand(self.init_batch, self.env.n_actions).reshape(-1)
        next_states = self.env.einsum('aij, bjk->baik', self.U, states)
        distances = self.env.batch_distance_2(self.env.target, next_states)
        masks = distances > self.epsilon
        
        states_list.append(states.view(-1, self.env.state_size))
        # actions_list.append(actions)
        next_states_list.append(next_states.view(-1, self.env.n_actions, self.env.state_size))
        masks_list.append(masks.float().unsqueeze(-1))
        
        for i in range(1, n):
            next_indices = self.scramble_table[actions]
            states = self.env.einsum('abij, ajk->abik', self.U[next_indices], states).view(-1, 2, 2, 2)
            actions = next_indices.view(-1)
            next_states = self.env.einsum('aij, bjk->baik', self.U, states)
            distances = self.env.batch_distance_2(self.env.target, next_states)
            masks = distances > self.epsilon
            states_list.append(states.view(-1, self.env.state_size))
            next_states_list.append(next_states.view(-1, self.env.n_actions, self.env.state_size))
            masks_list.append(masks.float().unsqueeze(-1))
        return states_list, actions, next_states_list, masks_list

    @torch.no_grad()    
    def calc_data_rand(self, states, actions, n):
        states_list = []
        # actions_list = []
        next_states_list = []
        masks_list = []
        for i in range(n):
            next_indices = torch.gather(self.scramble_table[actions], 1, \
                            torch.randint(0, self.env.n_actions-1, (len(actions),1), device=device))
            states = self.env.einsum('abij, ajk->abik', self.U[next_indices], states).view(-1, 2, 2, 2)
            actions = next_indices.view(-1)
            next_states = self.env.einsum('aij, bjk->baik', self.U, states)
            distances = self.env.batch_distance_2(self.env.target, next_states)
            masks = distances > self.epsilon
            states_list.append(states.view(-1, self.env.state_size))
            next_states_list.append(next_states.view(-1, self.env.n_actions, self.env.state_size))
            masks_list.append(masks.float().unsqueeze(-1)) 
        return states_list, next_states_list, masks_list
