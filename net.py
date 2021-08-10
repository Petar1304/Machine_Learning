import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# vars
batch_size = 16
epochs = 3
learning_rate = 1e-4


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.stack = nn.Sequential(
				nn.Flatten(),
				nn.Linear(28*28, 512),
				nn.ReLU(),
				nn.Linear(512, 512),
				nn.ReLU(),
				nn.Linear(512, 10),
				nn.Softmax()
			)

	def forward(self, x):
		output = self.stack(x)
		return output


model = Net()

# load data
training_data = datasets.FashionMNIST(
		root='data',
		train=True,
		download=False,
		transform=ToTensor()
	)

test_data = datasets.FashionMNIST(
		root='data',
		train=False,
		download=False,
		transform=ToTensor()
	)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
	for batch, (X, y) in enumerate(dataloader):
		# compute prediction and loss
		pred = model(X)
		loss = loss_fn(pred, y)

		# backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch % 100 == 0:
			loss, current = loss.item(), batch * len(X)
			print(f'Loss: {loss}  [{current}/{len(dataloader.dataset)}]')


def test_loop(dataloader, model, loss_fn):
	test_loss = 0
	correct = 0

	with torch.no_grad():
		for X, y in dataloader:
			pred = model(X)
			test_loss += loss_fn(pred, y).item() # .item() to transform from tensor to number
			correct += (pred.argmax() == y).sum().item()

	test_loss /= len(dataloader)
	correct /= len(dataloader.dataset)

	print(f'Testing:\n-> Accuracy: {(100*correct)}%, Avg loss: {test_loss} \n')



for i in range(epochs):
	print(f'Epoch: {i+1}')
	train_loop(train_dataloader, model, loss_fn, optimizer)
	test_loop(test_dataloader, model, loss_fn)