import json
import numpy as np

class QLearningAgent:
    def __init__(self, cfg):
        # Definition of an observation:
        # vx, vy, vz, dx, dy, dz

        # Define action space: 
        # 0:+x, 1:-x, 2:+y, 3:-y, 4:+z, 5:-z
        self.num_actions = 6

        # Store the config
        self.cfg = cfg

        # Learning rate and exploration factor
        self.lr = self.cfg.agent.lr_start
        self.epsilon = self.cfg.agent.epsilon_start

        # Q-table: (vel_x, vel_y, vel_z, delta_x, delta_y, delta_z, action) → Q-value
        self.q_table = np.zeros((self.cfg.agent.vel_bins, self.cfg.agent.vel_bins, self.cfg.agent.vel_bins, 
                                 self.cfg.agent.delta_bins, self.cfg.agent.delta_bins, self.cfg.agent.delta_bins, 
                                 self.num_actions))

        self.visit_count = np.zeros((self.cfg.agent.vel_bins, self.cfg.agent.vel_bins, self.cfg.agent.vel_bins, 
                                 self.cfg.agent.delta_bins, self.cfg.agent.delta_bins, self.cfg.agent.delta_bins, 
                                 self.num_actions))
        

    def decay_epsilon(self, episodes):
        self.epsilon = max(self.cfg.agent.epsilon_min, self.cfg.agent.epsilon_start * self.cfg.agent.epsilon_decay ** episodes)


    def decay_alpha(self, episodes):
        self.lr = np.maximum(self.cfg.agent.lr_min, self.cfg.agent.lr_start * self.cfg.agent.lr_decay ** episodes)


    def to_bin(_, val, bins, max_abs=10.0, method='log'):
        val = np.clip(val, -max_abs, max_abs)
        if method == 'linear':
            scaled = val / max_abs
        elif method == 'log':
            # To avoid log(0), use log1p for numerical stability
            scaled = np.sign(val) * np.log1p(abs(val)) / np.log1p(max_abs)
        elif method == 'sqrt':
            scaled = np.sign(val) * np.sqrt(abs(val)) / np.sqrt(max_abs)
        else:
            raise ValueError("Unknown method")
    
        # Convert from [-1, 1] to [0, 1], then scale to bins
        normed = (scaled + 1) / 2
        return int(np.clip(normed * bins, 0, bins - 1))


    def discretize_obs(self, obs):
        # Get the bin for each observation element
        return [
            self.to_bin(s, bin_count, method='linear')
            for s, bin_count in zip(obs, [self.cfg.agent.vel_bins] * 3 + [self.cfg.agent.delta_bins] * 3)
        ]


    def choose_action(self, obs):
        # Exploration
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        
        # Exploitation
        discretized_obs = self.discretize_obs(obs)
        return np.argmax(self.q_table[*discretized_obs, :])


    def update(self, obs, action, reward, next_obs):
        discretized_obs = self.discretize_obs(obs)
        discretized_next_obs = self.discretize_obs(next_obs)

        current_q = self.q_table[*discretized_obs, action]
        max_next_q = np.max(self.q_table[*discretized_next_obs])

        # Q-learning update rule
        if self.cfg.agent.alpha_per_state:
            self.q_table[*discretized_obs, action] = \
                current_q + self.lr[*discretized_obs, action] * (reward + self.cfg.agent.gamma * max_next_q - current_q)
        else:
            self.q_table[*discretized_obs, action] = \
                current_q + self.lr * (reward + self.cfg.agent.gamma * max_next_q - current_q)

        # Count the visits per state action pair
        self.visit_count[*discretized_obs, action] += 1

    
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
