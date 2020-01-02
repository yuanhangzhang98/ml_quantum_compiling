# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:50:10 2019

@author: Yuanhang Zhang
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import matplotlib.pyplot as plt
from tqdm import trange
import numpy as np
import torch
torch.multiprocessing.set_start_method("spawn", force=True)

from model import Model
from agent import Agent
from system import System
from testDataset import TestDataset

num_epoch = 1320
batch_size = 20000
min_length = 1
cur_length = 45
full_dataset_length = 11
max_length = cur_length
num_samples = batch_size
accuracy_tolerance = 0.001

ckpt_dir = 'ckpts/'
result_dir = 'results/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = Model(input_size=8, embedding_size=5000, hidden_size=1000).to(device)
target_net = Model(input_size=8, embedding_size=5000, hidden_size=1000).to(device)

policy_net.load_state_dict(torch.load(ckpt_dir + 'model_{}_{}.ckpt'.format(num_epoch, cur_length), map_location=device))

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

env = System(device)
agent = Agent(policy_net, target_net, env, accuracy_tolerance)

brute_force_length = 9
maximum_depth = 100
expand_size = 3000
keep_size = 100000
n_sample = 50
targets = env.randSU(n_sample)

min_dists = []
seq_lengths = []

for i in trange(len(targets)):
    state = targets[i]
    min_dist, best_state, best_seq = agent.search(state, brute_force_length, expand_size, keep_size, maximum_depth)
    state_np = best_state[0].detach().cpu().numpy() + 1j * best_state[1].detach().cpu().numpy()
    state_np /= (np.linalg.det(state_np)) ** (1.0/2.0) 
    min_dists.append(min_dist.detach().cpu().numpy().item())
    seq_lengths.append(torch.sum((best_seq != -1).float()).detach().cpu().numpy().item())
    
    print('min_dist:', min_dist)
    print('best_state:', state_np)
    print('best_seq:', best_seq)

print('average distance:', sum(min_dists)/n_sample)
print('average length:', sum(seq_lengths)/n_sample)
