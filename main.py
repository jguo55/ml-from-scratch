import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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

model = ExampleNet()

criterion = nn.MSELoss()

dataset = SimpleDataset(10)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

params = list(model.parameters())
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
        predictions = model(dataset.getX())
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
plt.contourf(param1_grid, param2_grid, loss_values, 20, cmap='viridis')
plt.colorbar(label='Loss')
plt.xlabel('Parameter 1')
plt.ylabel('Parameter 2')
plt.title('Loss Contour Plot')

#freeze everything except first 2 weights (because thats what we used to plot loss)
for name, param in model.named_parameters():
    if not name == 'fc1.weight':
        param.requires_grad = False

epochs = 1000

#start from the point with the most loss
params[0].data[0, 0] = maxc[0]
params[0].data[0, 1] = maxc[1]
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
SGDlossX = [maxc[0]]
SGDlossY = [maxc[1]]

for epoch in range(epochs):
    for num, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)

        loss.backward()

        optimizer.step()

        SGDlossX.append(params[0].data[0, 0].item())
        SGDlossY.append(params[0].data[0, 1].item())

plt.plot(SGDlossX, SGDlossY)

#start from the point with the most loss
params[0].data[0, 0] = maxc[0]
params[0].data[0, 1] = maxc[1]
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
AdamlossX = [maxc[0]]
AdamlossY = [maxc[1]]

for epoch in range(epochs):
    for num, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)

        loss.backward()

        optimizer.step()

        AdamlossX.append(params[0].data[0, 0].item())
        AdamlossY.append(params[0].data[0, 1].item())

plt.plot(AdamlossX, AdamlossY)

plt.show()

    


