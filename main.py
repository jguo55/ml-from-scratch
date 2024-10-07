import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import copy


class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        self.fc1 = nn.Linear(2, 1)
        self.fc2 = nn.Linear(1,1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
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

dataset = SimpleDataset(10)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

params = list(SGDmodel.parameters())
param1_values = np.linspace(-10.0, 10.0, 50) 
param2_values = np.linspace(-10.0, 10.0, 50)

loss_values = np.zeros((len(param1_values), len(param2_values)))

#store coordinates to place with min and max loss
max_loss = -1
min_loss = 100
maxc = [0,0]
minc = [0,0]

for i, p1 in enumerate(param1_values):
    for j, p2 in enumerate(param2_values):
        # modify each of the weights to each point in the graph
        params[0].data[0, 0] = p1
        params[0].data[0, 1] = p2 
        
        # Forward pass
        predictions = SGDmodel(dataset.getX())
        loss = criterion(predictions, dataset.gety())  # compute loss
        
        # Store the loss
        loss_values[i, j] = loss

        if max_loss <= loss:
            max_loss = loss
            maxc = [p1, p2]
        if min_loss >= loss:
            min_loss = loss
            minc = [p1, p2]


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


ADAMmodel = copy.deepcopy(SGDmodel)
ADAMparams = list(ADAMmodel.parameters())

#start from the point with the most loss
params[0].data[0, 0] = maxc[0]
params[0].data[0, 1] = maxc[1]
SGDoptimizer = torch.optim.SGD(SGDmodel.parameters(), lr=0.01)
SGDlossX = [maxc[0]]
SGDlossY = [maxc[1]]

ADAMparams[0].data[0, 0] = maxc[0]
ADAMparams[0].data[0, 1] = maxc[1]
ADAMoptimizer = torch.optim.Adam(ADAMmodel.parameters(), lr=0.01)
AdamlossX = [maxc[0]]
AdamlossY = [maxc[1]]

SGDLine, = ax.plot(SGDlossX, SGDlossY)
ADAMLine, = ax.plot(AdamlossX, AdamlossY)

while True:
    for num, (X, y) in enumerate(dataloader):
        SGDoptimizer.zero_grad()
        SGDpredictions = SGDmodel(X)
        SGDloss = criterion(SGDpredictions, y)

        SGDloss.backward()

        SGDoptimizer.step()

        SGDlossX.append(params[0].data[0, 0].item())
        SGDlossY.append(params[0].data[0, 1].item())

        SGDLine.set_xdata(SGDlossX)
        SGDLine.set_ydata(SGDlossY)

        ADAMoptimizer.zero_grad()
        ADAMpredictions = ADAMmodel(X)
        ADAMloss = criterion(ADAMpredictions, y)

        ADAMloss.backward()

        ADAMoptimizer.step()

        AdamlossX.append(ADAMparams[0].data[0, 0].item())
        AdamlossY.append(ADAMparams[0].data[0, 1].item())

        SGDLine.set_xdata(SGDlossX)
        SGDLine.set_ydata(SGDlossY)

        ADAMLine.set_xdata(AdamlossX)
        ADAMLine.set_ydata(AdamlossY)

        figure.canvas.draw()
        
        figure.canvas.flush_events()

        plt.pause(0.01)

    


