import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device.')


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        output = self.linear_relu_stack(x)
        return output

model = NeuralNetwork().to(device)
# print(model)


X = torch.rand(1, 28, 28, device=device)
output = model(X)
pred_probability = nn.Softmax(dim=1)(output)
y_pred = pred_probability.argmax(1)
print(f'X: {X}')
print(f'Output: {output}')
print(f'Predicted probabilities: {pred_probability}')
print(f'Predicted class: {y_pred}')



