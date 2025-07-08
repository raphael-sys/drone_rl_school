from collections import deque
import json
import random
import numpy as np
import torch


class QNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, cfg):
        super().__init__()
        self.neuralnet = torch.nn.Sequential(
            torch.nn.Linear(state_dim, cfg.agent.layer_1_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(cfg.agent.layer_1_nodes, cfg.agent.layer_2_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(cfg.agent.layer_2_nodes, action_dim)
        )

    def forward(self, x):
        return self.neuralnet(x)


class ReplayBuffer:
    def __init__(self, cfg):
        self.buffer = deque(maxlen=cfg.agent.buffer_capacity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, sample_size):
        batch = random.sample(self.buffer, sample_size)
        o, a, r, o2, d = zip(*batch)
        return (
            torch.tensor(o, dtype=torch.float32, device=self.device),
            torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1),
            torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1),
            torch.tensor(o2, dtype=torch.float32, device=self.device),
            torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, cfg):
        # Definition of an observation:
        # vx, vy, vz, dx, dy, dz
        self.num_observations = 6

        # Define action space: 
        # 0:+x, 1:-x, 2:+y, 3:-y, 4:+z, 5:-z
        self.num_actions = 6

        # Store the config
        self.cfg = cfg

        # Initialize
        self.lr = self.cfg.agent.lr_start
        self.epsilon = self.cfg.agent.epsilon_start

        # Set a device to train on
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-table: (vel_x, vel_y, vel_z, delta_x, delta_y, delta_z, action) → Q-value
        self.q_net = QNetwork(self.num_observations, self.num_actions, cfg).to(device)

        # Target net
        self.target_net = QNetwork(self.num_observations, self.num_actions, cfg).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.cfg.agent.lr_start)
        

    def decay_epsilon(self, episodes):
        self.epsilon = max(self.cfg.agent.epsilon_min, self.cfg.agent.epsilon_start * self.cfg.agent.epsilon_decay ** episodes)


    def decay_alpha(self, episodes):
        self.lr = np.maximum(self.cfg.agent.lr_min, self.cfg.agent.lr_start * self.cfg.agent.lr_decay ** episodes)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr


    def choose_action(self, observation):
        # Exploration (epsilon greedy)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        
        # Exploitation
        with torch.no_grad():
            q_values = self.q_net(torch.tensor(observation, dtype=torch.float32))
            action = q_values.argmax().item()
        return action


    def update(self, buffer):
        states, actions, rewards, next_states, dones = buffer.sample(self.cfg.agent.batch_size)

        # Compute targets
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            targets = rewards + (1 - dones) * self.cfg.agent.gamma * max_next_q

        # Compute loss
        current_q = self.q_net(states).gather(1, actions)
        loss = torch.nn.functional.smooth_l1_loss(current_q, targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save_model(self, score, episode):
        raise NotImplementedError
        np.save(f"best_models/q_table_score_{score}_q_table.npy", self.q_table)
        with open(f"best_models/q_table_score_{score}_metadata.json", "w") as f:
            json.dump(metadata, f)


    def load_model(self, model_name):
        raise NotImplementedError
        self.q_table = np.load("q_table.npy")

        # Loading of the metadata not implemented
        with open("metadata.json", "r") as f:
            metadata = json.load(f)
        print('Loading of metadata not implemented. ' \
        'Metadata is shown here, but not stored or considered in the model.')
        print(metadata)
