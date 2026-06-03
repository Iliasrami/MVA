import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyScoreNetwork(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.fc1(x)
        x = self.softplus(x)
        x = self.fc2(x)
        x = self.softplus(x)
        x = self.fc3(x)
        return x
    
# conditional model for NCSN 
class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, num_levels):
        super().__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(num_levels, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.lin(x)
        gamma = self.embed(y)
        out = gamma.view(-1, self.num_out) * out
        return out
    
class ConditionalModel(nn.Module):
    def __init__(self, num_levels):
        super().__init__()
        self.lin1 = ConditionalLinear(2, 128, num_levels)
        self.lin2 = ConditionalLinear(128, 128, num_levels)
        self.lin3 = nn.Linear(128, 2)
    
    def forward(self, x, y):
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        return self.lin3(x)
    


    

    