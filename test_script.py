import torch
import torch.nn as nn
#import torch.cuda.profiler as profiler
N = 32
I = 128 
H = 256 
O = 1024
# 2 Layer MLP
model = torch.nn.Sequential(
            torch.nn.Linear(I, H), 
            torch.nn.Linear(H, O)
        ).cuda()
# Input and Label
x = torch.randn(N, I).cuda()
target = torch.empty(N, dtype=torch.long).random_(O).cuda()
# Loss and optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer.zero_grad()
with torch.autograd.profiler.emit_nvtx(record_shapes=True):
    ##profiler.start()
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
