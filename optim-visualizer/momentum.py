import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#change parameters here
torch.manual_seed(9)
model_momentum = 0.9
learning_rate = 0.1
weight_intial = 10

class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(1,3)
        self.fc3 = nn.Linear(3, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class SimpleDataset(Dataset):
    def __init__(self, points):
        self.X = torch.randn(points, 1)
        self.y = torch.randn(points, 1)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def getX(self):
        return self.X
    
    def gety(self):
        return self.y

SGDmodel = ExampleNet()

criterion = nn.MSELoss()

dataset = SimpleDataset(10)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

params = list(SGDmodel.parameters())
param1_values = np.linspace(-10.0, 10.0, 100) 

loss_values = []

for i, p1 in enumerate(param1_values):
    # modify each of the weights to each point in the graph
    params[0].data[0, 0] = p1

    predictions = SGDmodel(dataset.getX())
    loss = criterion(predictions, dataset.gety())  
        
    loss_values.append(loss.item())

plt.ion() #begin live plotting
figure, ax = plt.subplots(figsize=(10, 8))
ax.plot(param1_values, loss_values)
plt.xlabel('Parameter 1')
plt.ylabel('Loss')

#freeze everything except first 2 weights (because thats what we used to plot loss)
for name, param in SGDmodel.named_parameters():
    if not name == 'fc1.weight':
        param.requires_grad = False

params[0].data[0, 0] = weight_intial
SGDoptimizer = torch.optim.SGD(SGDmodel.parameters(), lr=learning_rate, momentum=model_momentum)
#SGDoptimizer = torch.optim.Adam(SGDmodel.parameters(), lr=0.1)
SGDweights = []
SGDlosses = []

SGDLine, = ax.plot(SGDweights, SGDlosses, '.-')

def step(X, y, model, optimizer, line, weights, losses):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y)

    loss.backward()

    optimizer.step()

    params = list(model.parameters())
    weights.append(params[0].data[0, 0].item())
    losses.append(loss.item())

    line.set_xdata(weights)
    line.set_ydata(losses)

def handle_close(evt):
    global plotopen
    plotopen = False

figure.canvas.mpl_connect('close_event', handle_close)

plotopen = True
while plotopen:
    step(dataset.getX(), dataset.gety(), SGDmodel, SGDoptimizer, SGDLine, SGDweights, SGDlosses)

    figure.canvas.draw()
        
    figure.canvas.flush_events()

    plt.pause(0.01)
