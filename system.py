# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 10:43:02 2019

@author: Yuanhang Zhang
"""

import numpy as np
import torch

pi = np.pi

class System:
    @torch.no_grad()
    def __init__(self, device):
        self.state_size = 8
        self.n_actions = 4    
        gamma = np.exp(1j * pi / 5)
        kappa = (np.sqrt(5) - 1) / 2
        self.U_np = np.zeros((4, 2, 2), dtype=np.complex64)
        
        # here we ignored the global phase for 1-qubit systems, casting the U here into SU(2)
        self.U_np[0] = np.array([[gamma ** (-4), 0], [0, gamma ** 3]], dtype=np.complex64)
        self.U_np[0] = self.U_np[0] / np.sqrt(np.linalg.det(self.U_np[0]))
        self.U_np[1] = self.U_np[0].conj().T
        self.U_np[2] = np.array([[-kappa * gamma ** (-1), np.sqrt(kappa) * gamma ** (-3)],\
                              [np.sqrt(kappa) * gamma ** (-3), -kappa]], dtype=np.complex64)
        self.U_np[2] = self.U_np[2] / np.sqrt(np.linalg.det(self.U_np[2]))
        self.U_np[3] = self.U_np[2].conj().T
        I = np.identity(2, dtype=np.complex64)
        
        # (n_basic_operation, real and imag parts, 2-by-2 matrix)
        self.U = torch.zeros((self.n_actions, 2, 2, 2), dtype=torch.float32, device=device)
        self.U[:, 0, :, :] = torch.tensor(np.real(self.U_np), dtype=torch.float32, device=device)
        self.U[:, 1, :, :] = torch.tensor(np.imag(self.U_np), dtype=torch.float32, device=device)
        
        self.target = torch.zeros((2, 2, 2), dtype=torch.float32, device=device)
        self.target[0, :, :] = torch.tensor(np.real(I), dtype=torch.float32, device=device)
        # used when scrambling to avoid reverse actions
        self.scramble_table = torch.tensor([[0, 2, 3],\
                                            [1, 2, 3],\
                                            [0, 1, 2],\
                                            [0, 1, 3]], dtype=torch.int64, device=device)
        self.device = device

    
    @torch.no_grad()
    def mul(self, x1, x2):
        '''
            complex matrix multiplication
            x1: 2 * p * q    x2: 2 * q * r
            first dimension: real and imag parts
        '''
        real = torch.matmul(x1[0], x2[0]) - torch.matmul(x1[1], x2[1])
        imag = torch.matmul(x1[1], x2[0]) + torch.matmul(x1[0], x2[1])
        return torch.stack((real, imag))
    
    @torch.no_grad()
    def einsum(self, equation, U, states):
        '''
            einsum with customized complex number computation
            replacement for the old batch_mul functions for clarity and unification
            batch_mul(U, states) = einsum('ij, ajk->aik', U, states)
            # note that U and states are reversed in batch_mul_1; this caused some bugs
            batch_mul_1(states, U) = einsum('abij, ajk->abik', U, states) 
            batch_mul_2(U, states) = einsum('aij, bjk->baik', U, states)
        '''
        real = torch.einsum(equation, U[..., 0, :, :], states[..., 0, :, :])\
             - torch.einsum(equation, U[..., 1, :, :], states[..., 1, :, :])
        imag = torch.einsum(equation, U[..., 0, :, :], states[..., 1, :, :])\
             + torch.einsum(equation, U[..., 1, :, :], states[..., 0, :, :])
        return torch.stack((real, imag), dim=-3)
    
    
    @torch.no_grad()
    def batch_mul(self, x, batch):
        '''
            complex matrix batch multiplication
            x: 2 * p * q    batch: batch_size * 2 * q * r
        '''
        real = torch.matmul(x[0], batch[:, 0]) - torch.matmul(x[1], batch[:, 1])
        imag = torch.matmul(x[1], batch[:, 0]) + torch.matmul(x[0], batch[:, 1])
        return torch.stack((real, imag), dim=1)

    @torch.no_grad()
    def batch_mul_1(self, x, batch):
        '''
            complex matrix batch multiplication
            used when calculating Qs_next
            x: batch_size * 2 * p * q    batch: batch_size * n_operation * 2 * q * r
            output: batch_size * n_operation * 2 * p * q
        '''
        real = torch.einsum('abij,ajk->abik', batch[:, :, 0], x[:, 0])\
                 - torch.einsum('abij,ajk->abik',  batch[:, :, 1], x[:, 1])
        imag = torch.einsum('abij,ajk->abik', batch[:, :, 0], x[:, 1])\
                 + torch.einsum('abij,ajk->abik', batch[:, :, 1], x[:, 0])                 
        return torch.stack((real, imag), dim=2)

    @torch.no_grad()
    def batch_mul_2(self, x, batch):
        '''
            complex matrix batch multiplication
            used when calculating next_states
            x: 3 * 2 * p * q    batch: batch_size * 2 * q * r
            output: batch_size * 3 * 2 * p * q
        '''
        real = torch.einsum('aij,bjk->baik', x[:, 0], batch[:, 0])\
                 - torch.einsum('aij,bjk->baik', x[:, 1], batch[:, 1])
        imag = torch.einsum('aij,bjk->baik', x[:, 1], batch[:, 0])\
                 + torch.einsum('aij,bjk->baik', x[:, 0], batch[:, 1])                 
        return torch.stack((real, imag), dim=2)
      
    @torch.no_grad()
    def step(self, x, action):
        return self.mul(self.U[action], x)
    
    @torch.no_grad()
    def scramble(self, length):
        '''
            a function used during debugging
            low efficiency, don't use it
        '''
        state = self.target
        action_0 = torch.randint(0, 3, (), dtype=torch.int32)
        state = self.step(state, action_0)
        actions = torch.randint(0, 2, (length-1,))
        last_action = action_0
        scramble_seq = [last_action.item()]
        for i in range(length - 1):
            new_action = self.scramble_table[last_action, actions[i]]
            state = self.step(state, new_action)
            last_action = new_action
            scramble_seq.append(last_action.item())
        return state, scramble_seq
    
    @torch.no_grad()
    def distance(self, a, b):
        diff = a - b
        return torch.sum(diff * diff)
    
    @torch.no_grad()
    def batch_distance(self, target, batch):
        '''
            matrix distance measured with F-norm
            target: (2, 2, 2)
            batch: (batch_sizes, 2, 2, 2)
        '''
        batched_target = target.expand(batch.shape)
        diff = batched_target - batch
        return torch.sqrt(torch.sum(diff * diff, dim=[-1,-2,-3]))

    @torch.no_grad()
    def batch_distance_2(self, target, batch):
        '''
            the quaternion distance between two SU(2) matrices
            in SU(2), matrices differ by -1 corresponds to the same rotation
            the last function cannot deal with this; here we use another metric
            target: (2, 2, 2)
            batch: (batch_sizes, 2, 2, 2)
            equal to theta/2 when theta is small
        '''
        batched_target = target.expand(batch.shape)
        inner_prod = torch.sum(batched_target[..., 0] * batch[..., 0], dim=[-1, -2])
        return torch.sqrt(1 - inner_prod * inner_prod)

    @torch.no_grad()
    def randU(self, batch_size):
        '''
            generate random 2*2 unitary matrices
            shape: (batch_size, 2, 2, 2)
            U = exp(ia) * [ exp( ib)cos(phi) exp( ic)sin(phi)
                           -exp(-ic)sin(phi) exp(-ib)cos(phi)]
            
        '''
        abc = 2 * pi * torch.rand((3, batch_size), device=self.device)
        cosa, cosb, cosc = torch.cos(abc)
        sina, sinb, sinc = torch.sin(abc)
        sinphi = torch.sqrt(torch.rand(batch_size, device=self.device))
        cosphi = torch.sqrt(1 - sinphi*sinphi)
        real00 =  cosa * cosb * cosphi - sina * sinb * cosphi
        real01 =  cosa * cosc * sinphi - sina * sinc * sinphi
        real10 = -cosa * cosc * sinphi - sina * sinc * sinphi
        real11 =  cosa * cosb * cosphi + sina * sinb * cosphi
        imag00 =  cosa * sinb * cosphi + sina * cosb * cosphi
        imag01 =  cosa * sinc * sinphi + sina * cosc * sinphi
        imag10 =  cosa * sinc * sinphi - sina * cosc * sinphi
        imag11 = -cosa * sinb * cosphi + sina * cosb * cosphi
        U = torch.stack((real00, real01, real10, real11, imag00, imag01, imag10, imag11), dim=1)\
                    .view(batch_size, 2, 2, 2)
        return U
    
    @torch.no_grad()
    def randSU(self, batch_size):
        '''
            generate random 2*2 special unitary matrices
            shape: (batch_size, 2, 2, 2)
            U =           [ exp( ib)cos(phi) exp( ic)sin(phi)
                           -exp(-ic)sin(phi) exp(-ib)cos(phi)]
            
        '''
        bc = 2 * pi * torch.rand((2, batch_size), device=self.device)
        cosb, cosc = torch.cos(bc)
        sinb, sinc = torch.sin(bc)
        sinphi = torch.sqrt(torch.rand(batch_size, device=self.device))
        cosphi = torch.sqrt(1 - sinphi*sinphi)
        real00 =  cosb * cosphi
        real01 =  cosc * sinphi
        real10 = -cosc * sinphi
        real11 =  cosb * cosphi
        imag00 =  sinb * cosphi
        imag01 =  sinc * sinphi
        imag10 =  sinc * sinphi
        imag11 = -sinb * cosphi
        U = torch.stack((real00, real01, real10, real11, imag00, imag01, imag10, imag11), dim=1)\
                    .view(batch_size, 2, 2, 2)
        return U
    
    @torch.no_grad()    
    def randRotation(self, max_theta, batch_size):
        '''
            Rn(theta) = cos(theta/2) I - i sin(theta/2) (nx X + ny Y + nz Z)
            axis \hat{n} is randomly selected
            theta is uniformly selected between [-max_theta, max_theta]
        '''
        axis = torch.randn(3, batch_size, device=self.device)
        axis = axis / torch.sqrt(torch.sum(axis * axis, dim=0))
        a, b, c = axis
        theta = max_theta * (torch.rand(batch_size, device=self.device) - 0.5)
        sintheta = torch.sin(theta)
        costheta = torch.cos(theta)
        real00 =  costheta
        real01 =  b * sintheta
        real10 = -b * sintheta
        real11 =  costheta
        imag00 = -c * sintheta
        imag01 = -a * sintheta
        imag10 = -a * sintheta
        imag11 =  c * sintheta
        U = torch.stack((real00, real01, real10, real11, imag00, imag01, imag10, imag11), dim=1)\
                    .view(batch_size, 2, 2, 2)        
        return U
        