import math

import numpy as np
import torch
from torch import nn


class SineLayer(nn.Module):
    """Linear layer with sine activation. Adapted from Siren repo"""

    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # self.linear = nn.utils.weight_norm(nn.Linear(in_features, out_features, bias=bias))
        # self.layer_norm = nn.LayerNorm(out_features)
        # self.batch_norm = nn.BatchNorm1d(out_features)
        self.init_weights()

    def init_weights(self): # initialization function to initialize the weights in the first layer and later layers with normalized criterion
        with torch.no_grad():
            if self.is_first:
                # self.linear.weight.uniform_ - modifies the current values of the weights and they are set to a normal distribution (-1/n, 1/n)
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else: # The rest of layers are initialized with the following random values in the uniform distribution U()
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, x):
        # NOTE: Uncomment when using batch (or layer) normalization.
        # x = self.linear(x)
        # x = self.layer_norm(x)
        # x = self.batch_norm(x)
        # return torch.sin(self.omega_0 * x)

        return torch.sin(self.omega_0 * self.linear(x))
    

class Fourier_Enc(nn.Module):
    def __init__(self, c, L, initial_beta=10.0):
        
        super().__init__()
        self.L = L
        self.c = c
        
        # Trainable beta parameter
        self.beta = nn.Parameter(torch.tensor(initial_beta, dtype=torch.float32))
        # Is beta going to be trained??
        self.B = torch.FloatTensor(self.c , self.L).uniform_(-1, 1) 
        
        self.register_buffer('Bmatrix', self.B)
        self.scaled_B = self.B * self.beta
        
    def forward(self, coor):
        # Scale B matrix by the trainable beta
        
        # Compute Fourier angles: 2 * pi * (coor @( scaled_B)
        fourier_angle = (2 * np.pi * coor.unsqueeze(-1) * self.scaled_B)  # (n x 4) @ (4 x len_enc)

        # Apply cosine and sine, then concatenate along the last dimension
        cos_encoding = torch.cos(fourier_angle)  # (n x len_enc)
        sin_encoding = torch.sin(fourier_angle)  # (n x len_enc)

        # Concatenate both along the last dimension to get the desired shape
        encoded_fourier = torch.cat([cos_encoding, sin_encoding], dim=-1)  # (n x (2 * len_enc))
        return encoded_fourier       
        
                
class Siren(nn.Module):
    def __init__(self,      
            c: int=4 , 
            L: int=10, 
            hidden_dim: int=512, 
            n_layers: int=8, 
            out_dim: int=2, 
            omega_0: int=30,
            outermost_linear=False,
            ) -> None:
        
        super().__init__()
        
        # self.sine_layers = nn.ModuleList()
        self.L = L
        self.c = c
        fourier_dim = self.L*2*self.c
        
        self.sine_layers = nn.ModuleList()
        self.sine_layers.append(SineLayer(fourier_dim, hidden_dim, is_first=True, omega_0=omega_0))

        for i in range(n_layers-1):
            self.sine_layers.append(
                SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0)
            )
        
        # Regarding the last layer
        if outermost_linear:
            self.final_layer = nn.Linear(hidden_dim, out_dim)
            # For initialization purposes, don't keep track of the weights when you initialize them:
            with torch.no_grad():
                self.final_layer.weight.uniform_(-np.sqrt(6 / hidden_dim) / omega_0, np.sqrt(6 / hidden_dim) / omega_0)
            
        else:
            self.final_layer = SineLayer(hidden_dim, out_dim, is_first=False, omega_0=omega_0)
            
        self.sine_layers.append(self.final_layer)
        
        # self.sine_layers = nn.Sequential(self.sine_layers)
        # self.sine_layers = nn.ModuleList(self.sine_layers)
        # self.output_layer = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, coords):
        encoder = Fourier_Enc(self.c, self.L)
        x = encoder(coords)
        for layer in self.sine_layers:
            x = layer(x)
        return x