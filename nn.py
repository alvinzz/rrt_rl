import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal

from device import DEVICE
import numpy as np

class Policy(nn.Module):
    def __init__(self, layer_sizes):
        super(Policy, self).__init__()
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            dim0 = layer_sizes[i]
            dim1 = layer_sizes[i+1]
            self.weights.append(nn.Parameter(nn.init.orthogonal_(torch.empty([dim0, dim1], dtype=torch.float, device=DEVICE))))
            self.biases.append(nn.Parameter(torch.zeros([1, dim1], dtype=torch.float, device=DEVICE)))

    def forward(self, state, goal):
        state = torch.tensor(state, dtype=torch.float, device=DEVICE)
        goal = torch.tensor(goal, dtype=torch.float, device=DEVICE)
        if state.shape[0] == 1:
            state = state.repeat([goal.shape[0], 1])
        if goal.shape[0] == 1:
            goal = goal.repeat([state.shape[0], 1])
        x = torch.cat([state, goal], dim=1)

        for (i, (w, b)) in enumerate(zip(self.weights, self.biases)):
            x = torch.matmul(x, w) + b
            if i < len(self.weights) - 1:
                x = F.relu(x)

        mean, log_std = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:]

        z = Normal(mean.view(-1), torch.exp(log_std.view(-1)))

        sample = z.sample()
        raw_action = sample.view(x.shape[0], -1)
        raw_log_prob = z.log_prob(sample).view(x.shape[0], -1).sum(dim=1)

        action = torch.tanh(raw_action)
        log_prob = raw_log_prob - (2*raw_action + np.log(4.0) - 2*F.softplus(2*raw_action)).sum(dim=1)

        return action, log_prob

    def __call__(self, state, goal):
        action, log_prob = self.forward(state, goal)
        return action.detach().cpu().numpy()

def 

if  __name__ == "__main__":
    p = Policy([4, 100, 100, 4])
    print(p(np.array([[-2, -2.]]), np.array([[2, 2.]])))