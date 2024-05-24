import torch
import torch.nn as nn

class TriaModule(nn.Module):
    def __init__(self, num_input, num_classes, droupout_prop=0):
        super(TriaModule, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(num_input, 5),
            nn.ReLU(),
            nn.Linear(5, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes),
            nn.Dropout(p=droupout_prop),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.pipe(x)

net = TriaModule(num_input=2, num_classes=3)
print(net)
v = torch.FloatTensor([[2, 3],[4,6]])
out = net(v)
print(out)
print("Cuda's availability is %s" % torch.cuda.is_available())
if torch.cuda.is_available():
    print("Data from cuda: %s" % out.to('cuda'))