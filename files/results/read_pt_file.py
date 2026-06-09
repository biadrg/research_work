import torch
state = torch.load('results/qsann_global.pt', map_location='cpu')
print(state.keys())   # lists every named parameter