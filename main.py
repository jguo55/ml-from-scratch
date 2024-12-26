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
    
class AdamOptimizer():
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=0.00000001):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        #initialize moment vectors for each parameter
        self.m = {}
        self.v = {}
        for name, param in model.named_parameters():
            self.m[name] = np.zeros_like(param.data)
            self.v[name] = np.zeros_like(param.data)
        self.t = 1 #initialize timestep

    def updateParameters(self): #.step() in pytorch
        #here you're supposed to calculate gradients
        #conveniently, pytorch does that for you through loss.backward() and stores them, so you can just use those

        for name, param in model.named_parameters():
            #moment vector calculations (moving averages)
            mt = self.beta1*self.m[name]-(1-self.beta1)*param.grad
            vt = self.beta2*self.v[name]-(1-self.beta2)*param.grad**2

            #bias correction (because the vectors were initialized to 0)
            mhat = mt/(1-self.beta1**self.t)
            vhat = vt/(1-self.beta2**self.t)

            #update parameters
            param.data = param.data - self.lr*mhat/(np.sqrt(vhat)+self.eps)

            #update vectors
            self.m[name] = mt
            self.v[name] = vt

        self.t += 1

    


#lowkey the easiest way we can test if this is working is to just create 2 models with the same parameters and see if the function lines up with the pytorch one
#perhaps use mnist to test

model = ExampleNet()
criterion = nn.CrossEntropyLoss
optimizer = AdamOptimizer(model)

