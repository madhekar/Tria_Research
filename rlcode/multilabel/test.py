import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(20, 5) # predict logits for 5 classes

x = torch.randn(1, 20)
y = torch.tensor([[1., 0., 1., 0., 0.]]) # get classA and classC as active
y1 = torch.tensor([[0., 1., 0., 1., 0.]]) #get classB and classD as active

print('x, y: ', x, y, y1)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-1)

for epoch in range(20):
    optimizer.zero_grad()
    output = model(x)
    print('output: ', output)
    loss1 = criterion(output, y)
    loss2 = criterion(output, y1)
    loss = loss1 + loss2
    loss.backward()
    optimizer.step()
    print('Loss: {:.3f}'.format(loss.item()))
