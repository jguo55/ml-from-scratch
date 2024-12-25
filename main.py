import torch
import torch.nn as nn

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

model = ExampleNet()
print(list(model.parameters()))
criterion = nn.CrossEntropyLoss

def Adam(params, learning_rate):
    beta1 = 0.9
    beta2 = 0.999
    eps = 0.00000001 #1e-08
    return 0