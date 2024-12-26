import torch
import torch.nn as nn

import numpy as np

class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        self.fc1 = nn.Linear(3, 5)
        self.fc2 = nn.Linear(5,3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#lowkey the easiest way we can test if this is working is to just create 2 models with the same parameters and see if the function lines up with the pytorch one
#perhaps use mnist to test

def initializeAdam():
    #initialize moment vectors for each parameter
    m = {}
    v = {}
    for name, param in model.named_parameters():
        m[name] = np.zeros_like(param.data)
        v[name] = np.zeros_like(param.data)
    return m, v

def Adam(params, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=0.00000001, weight_decay=0, maximize=False): #amsgrad not implemented
    m, v = initializeAdam()
    print(m, v)
    #calculate gradient

    #moment 1 aka moving averages (m)
    #moment 2 (v)

    #m bias correction (mhat)
    #v bias correction (vhat)
    return 0

model = ExampleNet()
criterion = nn.CrossEntropyLoss
Adam(model)

