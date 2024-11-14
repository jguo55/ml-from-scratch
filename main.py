import torch
import numpy as np
import torch.nn as nn


#https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
#https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.zt = nn.Linear(input_size, hidden_size, bias=bias)
        self.rt = nn.Linear(hidden_size, hidden_size, bias=bias)

'''
**GRU CELL PSUEDOCODE**
1. update gate z(t)
z(t) = sigmoid(Weights*inputs + bias + Weights(t-1)*hidden(t-1)+bias(t-1))
2. reset gate r(t)
r(t) = sigmoid(Weights*inputs + bias + Weights(t-1)*hidden(t-1)+bias(t-1))
3. output
output = tanh(Weights*inputs + r(t) hadamard hidden)
4. next hidden
h(t) = z(t) hadamard h(t-1) + (1-z(t)) hadamard output
'''
