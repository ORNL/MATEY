import torch.nn as nn

class leadtimeMLP(nn.Module):
    def __init__(self, hidden_dim, exp_factor=4.):
        super().__init__()
        self.fc1 = nn.Linear(1, int(hidden_dim * exp_factor))
        self.fc2 = nn.Linear(int(hidden_dim * exp_factor), hidden_dim)
        self.act = nn.GELU()
        
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
 
