from collections import deque
import json
import random
import numpy as np
import torch


class QNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.neuralnet = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.neuralnet(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
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
    def __init__(self, vel_bins=16, delta_bins=16, 
                 alpha=1.0, epsilon=1.0, gamma=0.99,
                 alpha_min=0.005, epsilon_min=0.01,
                 alpha_decay=0.9995, epsilon_decay=0.9995):
        # Definition of an observation:
        # vx, vy, vz, dx, dy, dz
        self.num_observations = 6

        # The number of bins per observation dimension
        self.vel_bins = vel_bins
        self.delta_bins = delta_bins

        # Define action space: 
        # 0:+x, 1:-x, 2:+y, 3:-y, 4:+z, 5:-z
        self.num_actions = 6

        # Learning hyperparameters
        self.alpha_0 = alpha
        self.alpha = alpha    # one global learning rate
        self.gamma = gamma  # discount factor
        self.epsilon_0 = epsilon  # exploration factor
        self.epsilon = epsilon
        self.alpha_min = alpha_min
        self.epsilon_min = epsilon_min
        self.alpha_decay = alpha_decay
        self.epsilon_decay = epsilon_decay

        # Q-table: (vel_x, vel_y, vel_z, delta_x, delta_y, delta_z, action) → Q-value
        self.q_net = QNetwork(self.num_observations, self.num_actions)
        

    def decay_epsilon(self, episodes):
        self.epsilon = max(self.epsilon_min, self.epsilon_0 * self.epsilon_decay ** episodes)


    def decay_global_alpha(self, episodes):
        self.alpha = np.maximum(self.alpha_min, self.alpha_0 * self.alpha_decay ** episodes)


    def choose_action(self, observation):
        # Exploration (epsilon greedy)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        
        # Exploitation
        with torch.no_grad():
            q_values = self.q_net(observation)
            action = q_values.argmax().item()
        return action


    def update(self, obs, action, reward, next_obs, alpha_individual_decay):
        discretized_obs = self.discretize_obs(obs)
        discretized_next_obs = self.discretize_obs(next_obs)

        current_q = self.q_table[*discretized_obs, action]
        max_next_q = np.max(self.q_table[*discretized_next_obs])

        # Q-learning update rule
        if self.alpha_per_state:
            self.q_table[*discretized_obs, action] = \
                current_q + self.alpha[*discretized_obs, action] * (reward + self.gamma * max_next_q - current_q)
        else:
            self.q_table[*discretized_obs, action] = \
                current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

        # Count the visits per state action pair
        self.visit_count[*discretized_obs, action] += 1

    
    def save_model(self, score, episode):
        np.save(f"best_models/q_table_score_{score}_q_table.npy", self.q_table)
        metadata = {
            "score": score,
            "episodes": episode,
            "delta_bins": self.delta_bins, 
            "vel_bins": self.vel_bins, 
            "agent": self.alpha, 
            "epsilon": self.epsilon, 
            "alpha_min": self.alpha_min, 
            "epsilon_min": self.epsilon_min, 
            "alpha_decay": self.alpha_decay, 
            "epsilon_decay": self.epsilon_decay,
            }
        with open(f"best_models/q_table_score_{score}_metadata.json", "w") as f:
            json.dump(metadata, f)


    def load_model(self, model_name):
        self.q_table = np.load("q_table.npy")

        # Loading of the metadata not implemented
        with open("metadata.json", "r") as f:
            metadata = json.load(f)
        print('Loading of metadata not implemented. ' \
        'Metadata is shown here, but not stored or considered in the model.')
        print(metadata)
