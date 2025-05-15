import numpy as np

class QLearningAgent:
    def __init__(self, vel_bins=4, delta_bins=12, 
                 alpha=0.9, epsilon=1.0, gamma=0.99,
                 alpha_min=0.01, epsilon_min=0.05,
                 alpha_decay=0.9995, epsilon_decay=0.9995,
                 alpha_per_state=False):
        # Definition of an observation:
        # vx, vy, vz, dx, dy, dz

        # The number of bins per observation dimension
        self.vel_bins = vel_bins
        self.delta_bins = delta_bins

        # Define action space: 
        # 0:+x, 1:-x, 2:+y, 3:-y, 4:+z, 5:-z
        self.num_actions = 6

        # Learning hyperparameters
        self.alpha_0 = alpha
        self.alpha_per_state = alpha_per_state
        if self.alpha_per_state:
            self.alpha = np.full((self.vel_bins, self.vel_bins, self.vel_bins, 
                                    self.delta_bins, self.delta_bins, self.delta_bins, 
                                    self.num_actions), alpha)  # learning rate for each state action pair
        else:
            self.alpha = alpha    # one global learning rate
        self.gamma = gamma  # discount factor
        self.epsilon_0 = epsilon  # exploration factor
        self.epsilon = epsilon
        self.alpha_min = alpha_min
        self.epsilon_min = epsilon_min
        self.alpha_decay = alpha_decay
        self.epsilon_decay = epsilon_decay

        # Q-table: (vel_x, vel_y, vel_z, delta_x, delta_y, delta_z, action) → Q-value
        self.q_table = np.zeros((self.vel_bins, self.vel_bins, self.vel_bins, 
                                 self.delta_bins, self.delta_bins, self.delta_bins, 
                                 self.num_actions))

        self.visit_count = np.zeros((self.vel_bins, self.vel_bins, self.vel_bins, 
                                 self.delta_bins, self.delta_bins, self.delta_bins, 
                                 self.num_actions))
        

    def decay_epsilon(self, episodes):
        self.epsilon = max(self.epsilon_min, self.epsilon_0 * self.epsilon_decay ** episodes)


    def decay_individual_alpha(self):
        adapted_decay_rate = np.power(self.alpha_0, self.visit_count)
        self.alpha = np.maximum(self.alpha_min, self.alpha_0 * adapted_decay_rate)


    def decay_global_alpha(self, episodes):
        self.alpha = np.maximum(self.alpha_min, self.alpha_0 * self.alpha_decay ** episodes)


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
            self.to_bin(s, bin_count, method='sqrt')
            for s, bin_count in zip(obs, [self.vel_bins] * 3 + [self.delta_bins] * 3)
        ]


    def choose_action(self, obs):
        # Exploration
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        
        # Exploitation
        discretized_obs = self.discretize_obs(obs)
        return np.argmax(self.q_table[*discretized_obs, :])


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
