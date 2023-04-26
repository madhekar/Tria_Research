import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(1, 1)
        
    def forward(self, x):
        y = self.fc1(x)
        return [y, x]

model = MyModel()
x = torch.randn(1, 1)
outputs = model(x)
print(outputs)
