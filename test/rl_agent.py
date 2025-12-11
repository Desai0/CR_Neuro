import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Dict, Tuple, Any

# --- Configuration ---
GRID_W = 18
GRID_H = 15 # Output grid height (Playable Area)
OBS_GRID_H = 36 # Input observation grid height (36x18)
N_TILES = GRID_W * GRID_H
N_SLOTS = 4
N_ACTIONS = N_SLOTS * N_TILES # 4 * 270 = 1080

OBSERVATION_SIZE = 1 + 6 + 4 + (OBS_GRID_H * GRID_W) # 11 + 648 = 659

# Hyperparameters
BATCH_SIZE = 64
LR = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 10000 # Frames
TARGET_UPDATE = 1000
MEMORY_SIZE = 100000

class ActionWrapper:
    """
    Converts discrete action ID (0..1079) to actionable dictionary.
    """
    def __init__(self):
        self.grid_w = GRID_W
        self.grid_h = GRID_H # 15
        self.n_slots = N_SLOTS
    
    def decode_action(self, action_id: int) -> Dict[str, Any]:
        # action_id = slot * n_tiles + tile_index
        # tile_index = y * w + x
        
        slot_index = action_id // N_TILES
        tile_index = action_id % N_TILES
        
        tile_y = tile_index // self.grid_w
        tile_x = tile_index % self.grid_w
        
        return {
            'slot_index': slot_index,
            'tile': (tile_x, tile_y)
        }

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, device="cpu"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQNAgent] Using device: {self.device}")
        
        self.policy_net = DQN(OBSERVATION_SIZE, N_ACTIONS).to(self.device)
        self.target_net = DQN(OBSERVATION_SIZE, N_ACTIONS).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        
        self.action_wrapper = ActionWrapper()
        self.steps_done = 0
        
    def select_action(self, state_vector, eval_mode=False):
        # Epsilon-greedy
        sample = random.random()
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
            math.exp(-1. * self.steps_done / EPSILON_DECAY)
        
        if eval_mode:
            eps_threshold = 0.0
        
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                state_t = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                action_id = q_values.max(1)[1].item()
                return action_id
        else:
            return random.randrange(N_ACTIONS)

    def get_action_dict(self, action_id):
        return self.action_wrapper.decode_action(action_id)

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Q(s, a)
        current_q_values = self.policy_net(states_t).gather(1, actions_t)
        
        # V(s') = max_a Q(s', a)
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(1)[0].unsqueeze(1)
            expected_q_values = rewards_t + (GAMMA * next_q_values * (1 - dones_t))
            
        loss = nn.MSELoss()(current_q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print(f"[DQNAgent] Saved model to {path}")

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"[DQNAgent] Loaded model from {path}")

import math

