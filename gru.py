import torch
import numpy as np
import torch.nn as nn


#https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
#https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb
#https://medium.com/@yash9439/building-multi-layer-gru-from-scratch-305a03670fdd

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
                
        #each layer contains the weights for output, reset, and update. technically there should be 6 weight matrices, but they are condensed down to 2 and split up later
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights uniformly within [-1/sqrt(hidden_size), 1/sqrt(hidden_size)]
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):
        # Inputs:
        # input: of shape (batch_size, input_size)
        # hx: of shape (batch_size, hidden_size)
        # Output:
        # hy: of shape (batch_size, hidden_size)

        # Initialize hidden state with zeros if not provided
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size)

        # Compute linear transformations
        x_t = self.x2h(input)
        h_t = self.h2h(hx)

        # Split the transformations into reset, update, and new gate components
        ir, iz, in_ = x_t.chunk(3, 1)
        hr, hz, hn = h_t.chunk(3, 1)

        # Compute reset gate
        r = torch.sigmoid(ir + hr)
        
        # Compute update gate
        z = torch.sigmoid(iz + hz)
        
        # Compute candidate hidden state
        n = torch.tanh(in_ + (r * hn))

        # Compute final hidden state
        hy = (1 - z) * n + z * hx

        return n, hy
