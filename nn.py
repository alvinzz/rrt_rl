import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

        self.optimizer = optim.Adam(self.weights + self.biases, lr=0.1)

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

    def get_log_prob(self, state, goal, action):
        raw_action = 0.5*torch.log(1e-8 + (1+action)/(1e-8 + 1-action))

        x = torch.cat([state, goal], dim=1)
        for (i, (w, b)) in enumerate(zip(self.weights, self.biases)):
            x = torch.matmul(x, w) + b
            if i < len(self.weights) - 1:
                x = F.relu(x)
        mean, log_std = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:]

        z = Normal(mean.view(-1), torch.exp(log_std.view(-1)))
        raw_log_prob = z.log_prob(raw_action.view(-1)).view(x.shape[0], -1).sum(dim=-1)
        log_prob = raw_log_prob - (2*raw_action + np.log(4.0) - 2*F.softplus(2*raw_action)).sum(dim=1)

        return log_prob

    def optimize(self, state, goal, action, action_weight):
        state = torch.tensor(state, dtype=torch.float, device=DEVICE)
        goal = torch.tensor(goal, dtype=torch.float, device=DEVICE)
        action = torch.tensor(state, dtype=torch.float, device=DEVICE)
        action_weight = torch.tensor(action_weight, dtype=torch.float, device=DEVICE)
        if state.shape[0] == 1:
            state = state.repeat([goal.shape[0], 1])
        if goal.shape[0] == 1:
            goal = goal.repeat([state.shape[0], 1])

        for epoch in range(10):
            optimizer.zero_grad()
            loss = -(self.get_log_prob(state, goal, action) * action_weight).mean()
            loss.backward()
            optimizer.step()

class ValueFn(nn.Module):
    def __init__(self, layer_sizes):
        assert layer_sizes[-1] == 1, "value_fn last layer size must be 1"

        super(ValueFn, self).__init__()
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            dim0 = layer_sizes[i]
            dim1 = layer_sizes[i+1]
            self.weights.append(nn.Parameter(nn.init.orthogonal_(torch.empty([dim0, dim1], dtype=torch.float, device=DEVICE))))
            self.biases.append(nn.Parameter(torch.zeros([1, dim1], dtype=torch.float, device=DEVICE)))

        self.optimizer = optim.Adam(self.weights + self.biases, lr=0.001)

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

        return torch.min(torch.zeros_like(x[:, 0]), x[:, 0])

    def __call__(self, state, goal):
        with torch.no_grad():
            x = self.forward(state, goal)
            return x.detach().cpu().numpy()

    def optimize(self, state, goal, target_value):
        state = torch.tensor(state, dtype=torch.float, device=DEVICE)
        goal = torch.tensor(goal, dtype=torch.float, device=DEVICE)
        target_value = torch.tensor(target_value, dtype=torch.float, device=DEVICE)
        if state.shape[0] == 1:
            state = state.repeat([goal.shape[0], 1])
        if goal.shape[0] == 1:
            goal = goal.repeat([state.shape[0], 1])

        # for epoch in range(10):
        self.optimizer.zero_grad()
        value_pred = self.forward(state, goal)
        loss = torch.mean((value_pred - target_value)**2)
        loss.backward()
        self.optimizer.step()

if  __name__ == "__main__":
    p = Policy([4, 100, 4])
    print(p(np.array([[-2, -2.]]), np.array([[2, 2.]])))
    v = ValueFn([4, 100, 1])
    print(v(np.array([[-2, -2.]]), np.array([[2, 2.]])))
