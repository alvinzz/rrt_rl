import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions.normal import Normal

from device import DEVICE
import numpy as np

class Policy(nn.Module):
    def __init__(self, layer_sizes, dotp=True):
        super(Policy, self).__init__()
        self.weights = []
        self.biases = []
        self.dotp = dotp

        if self.dotp:
            self.s_weights = []
            self.s_biases = []
            self.g_weights = []
            self.g_biases = []
            for i in range(len(layer_sizes) - 1):
                dim0 = layer_sizes[i]
                dim1 = layer_sizes[i+1]

                if i == 0:
                    dim0 = layer_sizes[0] + layer_sizes[-2]
                self.weights.append(nn.Parameter(nn.init.orthogonal_(torch.empty([dim0, dim1], dtype=torch.float, device=DEVICE))))
                self.biases.append(nn.Parameter(torch.zeros([1, dim1], dtype=torch.float, device=DEVICE)))

                if i == 0:
                    dim0 = layer_sizes[0] // 2
                if i == len(layer_sizes) - 2:
                    break
                self.s_weights.append(nn.Parameter(nn.init.orthogonal_(torch.empty([dim0, dim1], dtype=torch.float, device=DEVICE))))
                self.g_weights.append(nn.Parameter(nn.init.orthogonal_(torch.empty([dim0, dim1], dtype=torch.float, device=DEVICE))))
                self.s_biases.append(nn.Parameter(torch.zeros([1, dim1], dtype=torch.float, device=DEVICE)))
                self.g_biases.append(nn.Parameter(torch.zeros([1, dim1], dtype=torch.float, device=DEVICE)))
        else:
            for i in range(len(layer_sizes) - 1):
                dim0 = layer_sizes[i]
                dim1 = layer_sizes[i+1]
                self.weights.append(nn.Parameter(nn.init.orthogonal_(torch.empty([dim0, dim1], dtype=torch.float, device=DEVICE))))
                self.biases.append(nn.Parameter(torch.zeros([1, dim1], dtype=torch.float, device=DEVICE)))

        if self.dotp:
            for i in range(len(self.weights)):
                self.register_parameter("weights_{}".format(i), self.weights[i])
                self.register_parameter("biases_{}".format(i), self.biases[i])
            for i in range(len(self.s_weights)):
                self.register_parameter("s_weights_{}".format(i), self.s_weights[i])
                self.register_parameter("s_biases_{}".format(i), self.s_biases[i])
                self.register_parameter("g_weights_{}".format(i), self.g_weights[i])
                self.register_parameter("g_biases_{}".format(i), self.g_biases[i])
            self.optimizer = optim.Adam(self.weights + self.biases + self.s_weights + self.s_biases + self.g_weights + self.g_biases, lr=0.001)
        else:
            self.optimizer = optim.Adam(self.weights + self.biases, lr=0.001)
            for i in range(len(self.weights)):
                self.register_parameter("weights_{}".format(i), self.weights[i])
                self.register_parameter("biases_{}".format(i), self.biases[i])

    def forward(self, state, goal):
        state = torch.tensor(state, dtype=torch.float, device=DEVICE)
        goal = torch.tensor(goal, dtype=torch.float, device=DEVICE)
        if state.shape[0] == 1:
            state = state.repeat([goal.shape[0], 1])
        if goal.shape[0] == 1:
            goal = goal.repeat([state.shape[0], 1])

        if self.dotp:
            s = state.clone()
            g = goal.clone()
            for (i, (s_w, s_b, g_w, g_b)) in enumerate(zip(self.s_weights, self.s_biases, self.g_weights, self.g_biases)):
                s = torch.matmul(s, s_w) + s_b
                g = torch.matmul(g, g_w) + g_b
            x = torch.cat([state, goal, s*g], dim=1)
            for (i, (w, b)) in enumerate(zip(self.weights, self.biases)):
                x = torch.matmul(x, w) + b
                if i < len(self.weights) - 1:
                    x = F.relu(x)
        else:
            x = torch.cat([state, goal], dim=1)
            for (i, (w, b)) in enumerate(zip(self.weights, self.biases)):
                x = torch.matmul(x, w) + b
                if i < len(self.weights) - 1:
                    x = F.relu(x)

        mean, log_std = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:]

        z = Normal(mean.reshape(-1), torch.exp(log_std.reshape(-1)))

        sample = z.sample()
        raw_action = sample.view(x.shape[0], -1)
        raw_log_prob = z.log_prob(sample).view(x.shape[0], -1).sum(dim=1)

        action = torch.tanh(raw_action)
        log_prob = raw_log_prob - (2*raw_action + np.log(4.0) - 2*F.softplus(2*raw_action)).sum(dim=1)

        return action, log_prob

    def __call__(self, state, goal):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float, device=DEVICE)
            goal = torch.tensor(goal, dtype=torch.float, device=DEVICE)
            if state.shape[0] == 1:
                state = state.repeat([goal.shape[0], 1])
            if goal.shape[0] == 1:
                goal = goal.repeat([state.shape[0], 1])

            if self.dotp:
                s = state.clone()
                g = goal.clone()
                for (i, (s_w, s_b, g_w, g_b)) in enumerate(zip(self.s_weights, self.s_biases, self.g_weights, self.g_biases)):
                    s = torch.matmul(s, s_w) + s_b
                    g = torch.matmul(g, g_w) + g_b
                x = torch.cat([state, goal, s*g], dim=1)
                for (i, (w, b)) in enumerate(zip(self.weights, self.biases)):
                    x = torch.matmul(x, w) + b
                    if i < len(self.weights) - 1:
                        x = F.relu(x)
            else:
                x = torch.cat([state, goal], dim=1)
                for (i, (w, b)) in enumerate(zip(self.weights, self.biases)):
                    x = torch.matmul(x, w) + b
                    if i < len(self.weights) - 1:
                        x = F.relu(x)

            mean, log_std = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:]

            z = Normal(mean.reshape(-1), torch.exp(log_std.reshape(-1)))

            sample = z.sample()
            raw_action = sample.view(x.shape[0], -1)

            action = torch.tanh(raw_action)

        return action.detach().cpu().numpy()

    def get_log_prob(self, state, goal, action):
        raw_action = 0.5*torch.log(1+action + 1e-8) - 0.5*torch.log(1-action + 1e-8) # inverse tanh

        if self.dotp:
            s = state.clone()
            g = goal.clone()
            for (i, (s_w, s_b, g_w, g_b)) in enumerate(zip(self.s_weights, self.s_biases, self.g_weights, self.g_biases)):
                s = torch.matmul(s, s_w) + s_b
                g = torch.matmul(g, g_w) + g_b
            x = torch.cat([state, goal, s*g], dim=1)
            for (i, (w, b)) in enumerate(zip(self.weights, self.biases)):
                x = torch.matmul(x, w) + b
                if i < len(self.weights) - 1:
                    x = F.relu(x)
        else:
            x = torch.cat([state, goal], dim=1)
            for (i, (w, b)) in enumerate(zip(self.weights, self.biases)):
                x = torch.matmul(x, w) + b
                if i < len(self.weights) - 1:
                    x = F.relu(x)

        mean, log_std = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:]
        z = Normal(mean.reshape(-1), torch.exp(log_std.reshape(-1)))

        #sample = (raw_action - mean.detach()) / torch.exp(log_std.detach())
        #raw_action = sample * torch.exp(log_std) + mean
        raw_log_prob = z.log_prob(raw_action.view(-1)).view(x.shape[0], -1).sum(dim=-1)

        #action = torch.tanh(raw_action)
        log_prob = raw_log_prob - (2*raw_action + np.log(4.0) - 2*F.softplus(2*raw_action)).sum(dim=1)

        #return action, log_prob
        return None, log_prob

    #def optimize(self, state, goal, action, value_fn, dynamics, temperature, forward):
    def optimize(self, state, goal, action, action_weight):
        state = torch.tensor(state, dtype=torch.float, device=DEVICE)
        goal = torch.tensor(goal, dtype=torch.float, device=DEVICE)
        action = torch.tensor(action, dtype=torch.float, device=DEVICE)
        action_weight = torch.tensor(action_weight, dtype=torch.float, device=DEVICE)
        # if state.shape[0] == 1:
        #     state = state.repeat([goal.shape[0], 1])
        # if goal.shape[0] == 1:
        #     goal = goal.repeat([state.shape[0], 1])

        # for epoch in range(10):
        # with torch.autograd.detect_anomaly():
        self.optimizer.zero_grad()
        action, action_log_prob = self.get_log_prob(state, goal, action)
        #loss = -(action_log_prob * action_weight).mean()
        loss = -(action_log_prob * (action_weight - action_log_prob)).mean()
        #if forward:
        #    loss = (temperature * action_log_prob - value_fn.forward(dynamics.forward(state, action), goal)).mean()
        #else:
        #    loss = (temperature * action_log_prob - value_fn.forward(state, dynamics.forward(goal, action))).mean()
        loss.backward()
        self.optimizer.step()

class ValueFn(nn.Module):
    def __init__(self, layer_sizes, dotp=True):
        assert layer_sizes[-1] == 1, "value_fn last layer size must be 1"

        super(ValueFn, self).__init__()
        self.weights = []
        self.biases = []
        self.dotp = dotp

        if self.dotp:
            self.s_weights = []
            self.s_biases = []
            self.g_weights = []
            self.g_biases = []
            for i in range(len(layer_sizes) - 1):
                dim0 = layer_sizes[i]
                dim1 = layer_sizes[i+1]

                if i == 0:
                    dim0 = layer_sizes[0] + layer_sizes[-2]
                self.weights.append(nn.Parameter(nn.init.orthogonal_(torch.empty([dim0, dim1], dtype=torch.float, device=DEVICE))))
                self.biases.append(nn.Parameter(torch.zeros([1, dim1], dtype=torch.float, device=DEVICE)))

                if i == 0:
                    dim0 = layer_sizes[0] // 2
                if i == len(layer_sizes) - 2:
                    break
                self.s_weights.append(nn.Parameter(nn.init.orthogonal_(torch.empty([dim0, dim1], dtype=torch.float, device=DEVICE))))
                self.g_weights.append(nn.Parameter(nn.init.orthogonal_(torch.empty([dim0, dim1], dtype=torch.float, device=DEVICE))))
                self.s_biases.append(nn.Parameter(torch.zeros([1, dim1], dtype=torch.float, device=DEVICE)))
                self.g_biases.append(nn.Parameter(torch.zeros([1, dim1], dtype=torch.float, device=DEVICE)))
        else:
            for i in range(len(layer_sizes) - 1):
                dim0 = layer_sizes[i]
                dim1 = layer_sizes[i+1]
                self.weights.append(nn.Parameter(nn.init.orthogonal_(torch.empty([dim0, dim1], dtype=torch.float, device=DEVICE))))
                self.biases.append(nn.Parameter(torch.zeros([1, dim1], dtype=torch.float, device=DEVICE)))


        if self.dotp:
            for i in range(len(self.weights)):
                self.register_parameter("weights_{}".format(i), self.weights[i])
                self.register_parameter("biases_{}".format(i), self.biases[i])
            for i in range(len(self.s_weights)):
                self.register_parameter("s_weights_{}".format(i), self.s_weights[i])
                self.register_parameter("s_biases_{}".format(i), self.s_biases[i])
                self.register_parameter("g_weights_{}".format(i), self.g_weights[i])
                self.register_parameter("g_biases_{}".format(i), self.g_biases[i])
            self.optimizer = optim.Adam(self.weights + self.biases + self.s_weights + self.s_biases + self.g_weights + self.g_biases, lr=0.001)
        else:
            for i in range(len(self.weights)):
                self.register_parameter("weights_{}".format(i), self.weights[i])
                self.register_parameter("biases_{}".format(i), self.biases[i])
            self.optimizer = optim.Adam(self.weights + self.biases, lr=0.001)

    def forward(self, state, goal):
        state = torch.tensor(state, dtype=torch.float, device=DEVICE)
        goal = torch.tensor(goal, dtype=torch.float, device=DEVICE)
        if state.shape[0] == 1:
            state = state.repeat([goal.shape[0], 1])
        if goal.shape[0] == 1:
            goal = goal.repeat([state.shape[0], 1])

        if self.dotp:
            s = state.clone()
            g = goal.clone()
            for (i, (s_w, s_b, g_w, g_b)) in enumerate(zip(self.s_weights, self.s_biases, self.g_weights, self.g_biases)):
                s = torch.matmul(s, s_w) + s_b
                g = torch.matmul(g, g_w) + g_b
            x = torch.cat([state, goal, s*g], dim=1)
            for (i, (w, b)) in enumerate(zip(self.weights, self.biases)):
                x = torch.matmul(x, w) + b
                if i < len(self.weights) - 1:
                    x = F.relu(x)
        else:
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
        # if state.shape[0] == 1:
        #     state = state.repeat([goal.shape[0], 1])
        # if goal.shape[0] == 1:
        #     goal = goal.repeat([state.shape[0], 1])

        # for epoch in range(10):
        self.optimizer.zero_grad()
        value_pred = self.forward(state, goal)
        # loss = torch.mean((value_pred - target_value)**2)
        loss = torch.mean(torch.abs(value_pred - target_value))
        loss.backward()
        self.optimizer.step()

class Dynamics(nn.Module):
    def __init__(self, layer_sizes, dotp=True):
        super(Dynamics, self).__init__()
        self.weights = []
        self.biases = []
        self.dotp = dotp

        self.s_dim = layer_sizes[-1]
        self.a_dim = layer_sizes[0] - self.s_dim

        if self.dotp:
            self.s_weights = []
            self.s_biases = []
            self.a_weights = []
            self.a_biases = []
            for i in range(len(layer_sizes) - 1):
                dim0 = layer_sizes[i]
                dim1 = layer_sizes[i+1]

                if i == 0:
                    dim0 = layer_sizes[0] + layer_sizes[-2]
                self.weights.append(nn.Parameter(nn.init.orthogonal_(torch.empty([dim0, dim1], dtype=torch.float, device=DEVICE))))
                self.biases.append(nn.Parameter(torch.zeros([1, dim1], dtype=torch.float, device=DEVICE)))

                if i == len(layer_sizes) - 2:
                    break
                if i == 0:
                    dim0 = self.s_dim
                self.s_weights.append(nn.Parameter(nn.init.orthogonal_(torch.empty([dim0, dim1], dtype=torch.float, device=DEVICE))))
                if i == 0:
                    dim0 = self.a_dim
                self.a_weights.append(nn.Parameter(nn.init.orthogonal_(torch.empty([dim0, dim1], dtype=torch.float, device=DEVICE))))
                self.s_biases.append(nn.Parameter(torch.zeros([1, dim1], dtype=torch.float, device=DEVICE)))
                self.a_biases.append(nn.Parameter(torch.zeros([1, dim1], dtype=torch.float, device=DEVICE)))
        else:
            for i in range(len(layer_sizes) - 1):
                dim0 = layer_sizes[i]
                dim1 = layer_sizes[i+1]
                self.weights.append(nn.Parameter(nn.init.orthogonal_(torch.empty([dim0, dim1], dtype=torch.float, device=DEVICE))))
                self.biases.append(nn.Parameter(torch.zeros([1, dim1], dtype=torch.float, device=DEVICE)))


        if self.dotp:
            for i in range(len(self.weights)):
                self.register_parameter("weights_{}".format(i), self.weights[i])
                self.register_parameter("biases_{}".format(i), self.biases[i])
            for i in range(len(self.s_weights)):
                self.register_parameter("s_weights_{}".format(i), self.s_weights[i])
                self.register_parameter("s_biases_{}".format(i), self.s_biases[i])
                self.register_parameter("a_weights_{}".format(i), self.a_weights[i])
                self.register_parameter("a_biases_{}".format(i), self.a_biases[i])
            self.optimizer = optim.Adam(self.weights + self.biases + self.s_weights + self.s_biases + self.a_weights + self.a_biases, lr=0.001)
        else:
            for i in range(len(self.weights)):
                self.register_parameter("weights_{}".format(i), self.weights[i])
                self.register_parameter("biases_{}".format(i), self.biases[i])
            self.optimizer = optim.Adam(self.weights + self.biases, lr=0.001)

    def forward(self, state, action):
        state = torch.tensor(state, dtype=torch.float, device=DEVICE)
        action = torch.tensor(action, dtype=torch.float, device=DEVICE)
        if state.shape[0] == 1:
            state = state.repeat([action.shape[0], 1])
        if action.shape[0] == 1:
            action = action.repeat([state.shape[0], 1])

        if self.dotp:
            s = state.clone()
            a = action.clone()
            for (i, (s_w, s_b, a_w, a_b)) in enumerate(zip(self.s_weights, self.s_biases, self.a_weights, self.a_biases)):
                s = torch.matmul(s, s_w) + s_b
                a = torch.matmul(a, a_w) + a_b
            x = torch.cat([state, action, s*a], dim=1)
            for (i, (w, b)) in enumerate(zip(self.weights, self.biases)):
                x = torch.matmul(x, w) + b
                if i < len(self.weights) - 1:
                    x = F.relu(x)
        else:
            x = torch.cat([state, action], dim=1)
            for (i, (w, b)) in enumerate(zip(self.weights, self.biases)):
                x = torch.matmul(x, w) + b
                if i < len(self.weights) - 1:
                    x = F.relu(x)

        return x

    def __call__(self, state, action):
        with torch.no_grad():
            x = self.forward(state, action)
            return x.detach().cpu().numpy()

    def optimize(self, state, action, next_state):
        state = torch.tensor(state, dtype=torch.float, device=DEVICE)
        action = torch.tensor(action, dtype=torch.float, device=DEVICE)
        next_state = torch.tensor(next_state, dtype=torch.float, device=DEVICE)
        # if state.shape[0] == 1:
        #     state = state.repeat([goal.shape[0], 1])
        # if goal.shape[0] == 1:
        #     goal = goal.repeat([state.shape[0], 1])

        # for epoch in range(10):
        self.optimizer.zero_grad()
        state_pred = self.forward(state, action)
        # loss = torch.mean((state_pred - next_state)**2)
        loss = torch.mean(torch.abs(state_pred - next_state))
        loss.backward()
        self.optimizer.step()

if  __name__ == "__main__":
    p = Policy([4, 100, 4])
    print(p(np.array([[-2, -2.]]), np.array([[2, 2.]])))
    v = ValueFn([4, 100, 1])
    print(v(np.array([[-2, -2.]]), np.array([[2, 2.]])))
