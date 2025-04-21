import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Generate noisy data
X = np.linspace(-4, 4, 30).reshape(-1, 1)
y = np.sin(X) + np.random.normal(0, 0.4, X.shape)  # y = sin(x) + noise

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

#complex model that overfits easily
class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
    
    def forward(self, x):
        return self.fc(x)

model_overfit = ExampleNet()
model_regularized = ExampleNet()

criterion = nn.MSELoss()
optimizer_overfit = optim.Adam(model_overfit.parameters())
optimizer_regularized = optim.Adam(model_regularized.parameters(), weight_decay=0.01)  # L2 Regularization

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Overfitting Model
    optimizer_overfit.zero_grad()
    y_train_pred_overfit = model_overfit(X_train_torch)
    loss_overfit = criterion(y_train_pred_overfit, y_train_torch)
    loss_overfit.backward()
    optimizer_overfit.step()
    
    # Regularized Model
    optimizer_regularized.zero_grad()
    y_train_pred_regularized = model_regularized(X_train_torch)
    loss_regularized = criterion(y_train_pred_regularized, y_train_torch)
    loss_regularized.backward()
    optimizer_regularized.step()

# Evaluate models
y_test_pred_overfit = model_overfit(X_test_torch).detach().numpy()
y_test_pred_regularized = model_regularized(X_test_torch).detach().numpy()

# Compute errors
test_error_overfit = np.mean((y_test - y_test_pred_overfit) ** 2)
test_error_regularized = np.mean((y_test - y_test_pred_regularized) ** 2)

print(f"Test Error (Overfitting Model): {test_error_overfit:.4f}")
print(f"Test Error (Regularized Model): {test_error_regularized:.4f}")

# Visualize results
plt.scatter(X_train, y_train, label="Train Data", color='blue')
plt.scatter(X_test, y_test, label="Test Data", color='red')

# Create a smooth curve to visualize the fit
X_curve = np.linspace(-4, 4, 100).reshape(-1, 1)
X_curve_torch = torch.tensor(X_curve, dtype=torch.float32)
y_curve_overfit = model_overfit(X_curve_torch).detach().numpy()
y_curve_regularized = model_regularized(X_curve_torch).detach().numpy()

plt.plot(X_curve, y_curve_overfit, label="Overfitted Model", color='green')
plt.plot(X_curve, y_curve_regularized, label="Regularized Model", color='purple')
plt.legend()
plt.title("Overfitting vs Regularization in Neural Networks")
plt.show()