# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 14:38:22 2019

@author: Yuanhang Zhang
"""

import numpy as np
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import trange
        
class Agent():
    def __init__(self, policy_net, target_net, env, epsilon):
        self.policy_net = policy_net
        self.target_net = target_net
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = 1e-3
        # self.loss_func = torch.nn.SmoothL1Loss()
        self.loss_func = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        self.state_size = self.env.state_size
        self.n_actions = self.env.n_actions
        self.l = 1.0 # parameter lambda used in A* search
        self.decimal_punish = 400
        self.epsilon = epsilon # stop searching when distance less than epsilon
        self.action_inv_table = [1, 0, 3, 2]
    
    def search(self, init, brute_force_length, expand_size, keep_size, maximum_depth=200):
        with torch.no_grad():
            states = self.env.einsum('aij, jk->aik', self.env.U, init)
            actions = torch.arange(self.n_actions).to(self.device)
            action_sequence = actions.view(-1, 1)   # (batch, sequence_len)
            distances = self.env.batch_distance_2(self.env.target, states)
            min_dist, idx = torch.min(distances, 0)
            best_state = states[idx]
            best_sequence = action_sequence[idx]

            for i in trange(1, brute_force_length):
                next_indices = self.env.scramble_table[actions]     # (batch, n_actions-1)
                states = self.env.einsum('abij, ajk->abik', self.env.U[next_indices], states)\
                                    .view(-1, 2, 2, 2)
                # (batch, n_actions-1, seq_len)
                next_action_sequence = action_sequence.expand(self.n_actions-1,\
                                            action_sequence.shape[0], action_sequence.shape[1]).transpose(0,1)
                action_sequence = torch.cat((next_action_sequence, next_indices.unsqueeze(-1)), dim=-1)
                action_sequence = action_sequence.view(-1, action_sequence.shape[2])
                actions = next_indices.view(-1)
                distances = self.env.batch_distance_2(self.env.target, states)
                
                val, idx = torch.min(distances, 0)
                if val < min_dist:
                    min_dist = val
                    best_state = states[idx]
                    best_sequence = action_sequence[idx]

            # memory is not enough to evaluate all states, so chunk the states first
            states_flattened = states.view(-1, self.state_size)
            chunk_size = 100000
            if len(states) > chunk_size:
                n_chunk = len(states) // chunk_size + 1
                states_list = []
                actions_list = []
                action_sequence_list = []
                cost_to_go_list = []
                path_cost_list = []
                cost_decimal_list = []
                cost_list = []
                for i in trange(n_chunk):
                    cost_to_go = self.target_net(states_flattened[i*chunk_size:(i+1)*chunk_size]).view(-1) # (batch)
                    path_cost = brute_force_length * torch.ones_like(cost_to_go, device=self.device)
                    cost_decimal = (cost_to_go - torch.round(cost_to_go)) ** 2
                    cost = self.l * path_cost + cost_to_go + self.decimal_punish * cost_decimal / cost_to_go
                    keep_size_i = min(keep_size//(n_chunk-1), len(cost))
                    value, index = torch.topk(cost, keep_size_i, dim=0, largest=False, sorted=True)
                    states_list.append(states[index])
                    actions_list.append(actions[index])
                    action_sequence_list.append(action_sequence[index])
                    cost_to_go_list.append(cost_to_go[index])
                    path_cost_list.append(path_cost[index])
                    cost_decimal_list.append(cost_decimal[index])
                    cost_list.append(cost[index])
                states = torch.cat(states_list)
                actions = torch.cat(actions_list)
                action_sequence = torch.cat(action_sequence_list)
                cost_to_go = torch.cat(cost_to_go_list)
                path_cost = torch.cat(path_cost_list)
                cost_decimal = torch.cat(cost_decimal_list)
                cost = torch.cat(cost_list)
            else:
                cost_to_go = self.target_net(states_flattened).view(-1) # (batch)
                path_cost = brute_force_length * torch.ones_like(cost_to_go, device=self.device)
                cost_decimal = (cost_to_go - torch.round(cost_to_go)) ** 2
                cost = self.l * path_cost + cost_to_go + self.decimal_punish * cost_decimal / cost_to_go    
            if len(cost) > keep_size:           
                cost, index = torch.topk(cost, keep_size, dim=0, largest=False, sorted=True)
            else:
                cost, index = torch.sort(cost)
            states = states[index]
            actions = actions[index]
            action_sequence = action_sequence[index]
            cost_to_go = cost_to_go[index]
            cost_decimal = cost_decimal[index]
            path_cost = path_cost[index]
            for i in trange(maximum_depth):
                states_expand = states[:expand_size]
                actions_expand = actions[:expand_size]
                
                next_indices = self.env.scramble_table[actions_expand]
                next_states = self.env.einsum('abij, ajk->abik', self.env.U[next_indices], states_expand)\
                                        .view(-1, 2, 2, 2)
                next_actions = next_indices.view(-1)
                next_cost_to_go = self.target_net(next_states.view(-1, self.state_size)).view(-1)
                next_cost_decimal = (next_cost_to_go - torch.round(next_cost_to_go)) ** 2
                next_path_cost = (path_cost[:expand_size]+1).expand(self.n_actions-1, expand_size)\
                                                            .transpose(0, 1).reshape(-1)
                next_action_sequence = action_sequence[:expand_size].expand(self.n_actions-1,\
                  expand_size, action_sequence.shape[1]).transpose(0,1).reshape((self.n_actions-1)*expand_size, -1)
                next_action_sequence = torch.cat((next_action_sequence, next_actions.unsqueeze(-1)), dim=-1)
                # in order to keep all action_seqs of same length, use -1 to mean "no action"
                action_sequence = torch.cat((action_sequence, \
                    -1*torch.ones((len(action_sequence),1), dtype=torch.int64, device=self.device)), dim=-1)
                
                distances = self.env.batch_distance_2(self.env.target, next_states)
                
                val, idx = torch.min(distances, 0)
                if val < min_dist:
                    min_dist = val
                    best_state = next_states[idx]
                    best_sequence = next_action_sequence[idx]

                states = torch.cat((states[expand_size:], next_states), dim=0)
                actions = torch.cat((actions[expand_size:], next_actions), dim=0)
                action_sequence = torch.cat((action_sequence[expand_size:], next_action_sequence), dim=0)
                cost_to_go = torch.cat((cost_to_go[expand_size:], next_cost_to_go), dim=0)
                cost_decimal = torch.cat((cost_decimal[expand_size:], next_cost_decimal), dim=0)
                path_cost = torch.cat((path_cost[expand_size:], next_path_cost), dim=0)
                cost = torch.cat((cost[expand_size:], self.l*next_path_cost+next_cost_to_go\
                                        +self.decimal_punish*next_cost_decimal/next_cost_to_go), dim=0)
                if len(cost) > keep_size:           
                    cost, index = torch.topk(cost, keep_size, dim=0, largest=False, sorted=True)
                else:
                    cost, index = torch.sort(cost)
                states = states[index]
                actions = actions[index]
                action_sequence = action_sequence[index]
                cost_to_go = cost_to_go[index]
                cost_decimal = cost_decimal[index]
                path_cost = path_cost[index]
                # print('Cost-to-go:', cost_to_go[0].detach().cpu().numpy().item(),\
                #       'Total cost:', cost[0].detach().cpu().numpy().item())
            return min_dist.detach(), best_state.detach(), best_sequence.detach()  

    def update_model(self, data):
        states = data['state']
        next_states = data['next_states']
        mask = data['mask']
        batch_size = len(states)
        cost = self.policy_net(states)
        with torch.no_grad():
            cost_target = self.target_net(next_states.reshape(batch_size*self.n_actions, self.state_size))\
                                .reshape(batch_size, self.n_actions, 1)
            cost_target = cost_target * mask
            cost_target = torch.min(cost_target, 1)[0] + 1.0
        loss = self.loss_func(cost, cost_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach()
