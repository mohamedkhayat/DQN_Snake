from collections import deque
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import random

class DQN(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(DQN,self).__init__()
        self.fc1 = nn.Linear(input_dim,512)
        self.fc2 = nn.Linear(512,512)
        self.fc3= nn.Linear(512,output_dim)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 


class Agent:
    def __init__(self,state_dim,action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=50000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.train_start = 3000
        self.target_update_interval = 1000  
        self.steps_done = 0 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_dim,action_dim).to(self.device)
        self.target_model =DQN(state_dim,action_dim).to(self.device)
        self.optimizer = Adam(self.model.parameters(),lr = 0.001)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
    
    def act(self,state,train=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if np.random.rand() <= self.epsilon:
          return random.randrange(self.action_dim)  
          
        if train:
            act_values = self.model(state)
            
        else:
            with torch.no_grad():
                act_values = self.model(state)
                
        return torch.argmax(act_values[0]).item()

    def replay(self):
        
        if len(self.memory) < self.train_start:
            return
            
        mini_batch = random.sample(self.memory,self.batch_size)
        states = torch.FloatTensor(np.array([experience[0] for experience in mini_batch])).to(self.device)
        actions = torch.LongTensor(np.array([experience[1] for experience in mini_batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([experience[2] for experience in mini_batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([experience[3] for experience in mini_batch])).to(self.device)
        dones = torch.FloatTensor(np.array([experience[4] for experience in mini_batch])).to(self.device)
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.target_model(next_states).max(1)[0]

        targets = rewards + self.gamma * next_q_values * (1 - dones)
        self.model.train()
        self.optimizer.zero_grad()
        loss = F.mse_loss(current_q_values.squeeze(), targets)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.steps_done += 1
        
        if self.steps_done % self.target_update_interval == 0:
            self.update_target_model()
            print(f"Target model updated at step {self.steps_done}")

        
        
        
        
        
        
        