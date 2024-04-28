import torch
import torch.nn as nn

loss = nn.CrossEntropyLoss()

input = torch.randn(3, 5, requires_grad=True)
print(input)
target = torch.randn(3, 5).softmax(dim=1)
output = loss(input, target)
output.backward()
print(input.grad)