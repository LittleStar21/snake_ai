import torch
import torch.nn as nn
import torch.optim as optim
import os

class DQN(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.fc1 = nn.Linear(in_size, 256)
        self.fc2 = nn.Linear(256, out_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def save(self):
        if not os.path.exists("./model"):
            os.mkdir("./model")
        torch.save(self.state_dict, "./model/model.pth")

class Trainer:
    def __init__(self, network, lr, gamma):
        self.network = network
        self.lr = lr
        self.gamma = gamma
        self.adam = optim.Adam(network.parameters(), lr = self.lr) 
        self.loss = nn.MSELoss()
    
    def step(self, states, actions, rewards, next_states, game_overs):
        states = torch.tensor(states, dtype = torch.float)
        actions = torch.tensor(actions, dtype = torch.long)
        rewards = torch.tensor(rewards, dtype = torch.float)
        next_states = torch.tensor(next_states, dtype = torch.float)
        
        if len(states.shape) == 1: #1-dimension
            states = states.unsqueeze(dim = 0)
            actions = actions.unsqueeze(dim = 0)
            rewards = rewards.unsqueeze(dim = 0)
            next_states = next_states.unsqueeze(dim = 0)
            game_overs = (game_overs, )

        predictions = self.network(states)
        targets = predictions.clone()
        
        for i in range(len(game_overs)):
            new_Q = rewards[i]
            if not game_overs[i]:
                new_Q = rewards[i] + self.gamma * torch.max(self.network(next_states[i]))
            targets[i][torch.argmax(actions[i]).item()] = new_Q
        
        self.adam.zero_grad()
        loss = self.loss(targets, predictions)
        loss.backward()
        self.adam.step()