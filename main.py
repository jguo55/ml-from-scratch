import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = ExampleNet()

loss_fn = nn.MSELoss()

#random dataset of datapoints
X = torch.randn(10, 2)
y = torch.randn(10, 1)

params = list(model.parameters())
param1_values = np.linspace(-10.0, 10.0, 50) 
param2_values = np.linspace(-10.0, 10.0, 50)

loss_values = np.zeros((len(param1_values), len(param2_values)))

#store coordinates to place with min and max loss
max_loss = [0, 0, 0]
min_loss = [0, 0, 100]

for i, p1 in enumerate(param1_values):
    for j, p2 in enumerate(param2_values):
        # modify each of the weights to each point in the graph
        params[0].data[0, 0] = p1
        params[0].data[0, 1] = p2 
        
        # Forward pass
        predictions = model(X)
        loss = loss_fn(predictions, y).item()  # compute loss
        
        # Store the loss
        loss_values[i, j] = loss

        if max_loss[2] <= loss:
            max_loss = [p1, p2, loss]
        if min_loss[2] >= loss:
            min_loss = [p1, p2, loss]

param1_grid, param2_grid = np.meshgrid(param1_values, param2_values)
plt.contourf(param1_grid, param2_grid, loss_values, 20, cmap='viridis')
plt.colorbar(label='Loss')
plt.xlabel('Parameter 1')
plt.ylabel('Parameter 2')
plt.title('Loss Contour Plot')
#plt.plot([max_loss[0],min_loss[0]],[max_loss[1],min_loss[1]])
plt.show()




