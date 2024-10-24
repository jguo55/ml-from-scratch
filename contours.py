import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import copy

#change parameters here
torch.manual_seed(0)

class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        self.fc1 = nn.Linear(2, 1)
        self.fc2 = nn.Linear(1,3)
        self.fc3 = nn.Linear(3, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class SimpleDataset(Dataset):
    def __init__(self, points):
        self.X = torch.randn(points, 2)
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

dataset = SimpleDataset(20)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

params = list(SGDmodel.parameters())
param1_values = np.linspace(-10.0, 10.0, 50) 
param2_values = np.linspace(-10.0, 10.0, 50)

loss_values = np.zeros((len(param1_values), len(param2_values)))

#find the highest loss point from -7.5 to 7.5 to start from
max_loss = -1
maxc = [0,0]

for i, p1 in enumerate(param1_values):
    for j, p2 in enumerate(param2_values):
        # modify each of the weights to each point in the graph
        params[0].data[0, 0] = p1
        params[0].data[0, 1] = p2 
        
        predictions = SGDmodel(dataset.getX())
        loss = criterion(predictions, dataset.gety())  
        
        loss_values[i, j] = loss
        if p1 > -7.5 and p1 < 7.5 and p2 > -7.5 and p2 < 7.5:
            if max_loss <= loss:
                max_loss = loss
                maxc = [p1, p2]

param1_grid, param2_grid = np.meshgrid(param1_values, param2_values)
plt.ion() #begin live plotting
figure, ax = plt.subplots(figsize=(10, 8))
plt.contourf(param1_grid, param2_grid, loss_values, 20, cmap='viridis')
plt.colorbar(label='Loss')
plt.xlabel('Parameter 1')
plt.ylabel('Parameter 2')
plt.title('Loss Contour Plot')

#freeze everything except first 2 weights (because thats what we used to plot loss)
for name, param in SGDmodel.named_parameters():
    if not name == 'fc1.weight':
        param.requires_grad = False

#start from the point with the most loss
params[0].data[0, 0] = maxc[0]
params[0].data[0, 1] = maxc[1]
SGDoptimizer = torch.optim.SGD(SGDmodel.parameters(), lr=0.1)
SGDlossX = [maxc[0]]
SGDlossY = [maxc[1]]

ADAMmodel = copy.deepcopy(SGDmodel)
ADAMoptimizer = torch.optim.Adam(ADAMmodel.parameters(), lr=0.1)
AdamlossX = [maxc[0]]
AdamlossY = [maxc[1]]

Mmodel = copy.deepcopy(SGDmodel)
Moptimizer = torch.optim.SGD(Mmodel.parameters(), lr=0.1, momentum=0.9)
MlossX = [maxc[0]]
MlossY = [maxc[1]]

MLine, = ax.plot(MlossX, MlossY, label = 'Momentum')
ADAMLine, = ax.plot(AdamlossX, AdamlossY, label='Adam')
SGDLine, = ax.plot(SGDlossX, SGDlossY, label='SGD')

plt.legend(loc="upper left")

def step(X, y, model, optimizer, line, lossesX, lossesY):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y)

    loss.backward()

    optimizer.step()

    params = list(model.parameters())
    lossesX.append(params[0].data[0, 0].item())
    lossesY.append(params[0].data[0, 1].item())

    line.set_xdata(lossesX)
    line.set_ydata(lossesY)

def handle_close(evt):
    global plotopen
    plotopen = False

figure.canvas.mpl_connect('close_event', handle_close)

plotopen = True
while plotopen:
    for num, (X, y) in enumerate(dataloader):
        step(X, y, SGDmodel, SGDoptimizer, SGDLine, SGDlossX, SGDlossY)
        step(X, y, ADAMmodel, ADAMoptimizer, ADAMLine, AdamlossX, AdamlossY)
        step(X, y, Mmodel, Moptimizer, MLine, MlossX, MlossY)

        figure.canvas.draw()
        
        figure.canvas.flush_events()

        plt.pause(0.01)

    


