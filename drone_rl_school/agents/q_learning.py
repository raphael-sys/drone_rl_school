import numpy as np

class QLearningAgent:
    def __init__(self, vel_bins=4, delta_bins=16, 
                 alpha=0.2, epsilon=1.0, gamma=0.99,
                 alpha_min=0.01, epsilon_min=0.05,
                 alpha_decay=0.999, epsilon_decay=0.9997):
        # The number of bins per dimension
        self.vel_bins = vel_bins
        self.delta_bins = delta_bins

        # Define action space: 0:+x, 1:-x, 2:+y, 3:-y, 4:+z, 5:-z
        self.num_actions = 6

        # Learning hyperparameters
        self.alpha_0 = alpha
        self.alpha = np.full((self.vel_bins, self.vel_bins, self.vel_bins, 
                                 self.delta_bins, self.delta_bins, self.delta_bins, 
                                 self.num_actions), alpha)  # learning rate for each state
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


    def decay_alpha(self):
        adapted_decay_rate = np.power(self.alpha_decay, self.visit_count / np.sum(self.visit_count))
        self.alpha = np.maximum(self.alpha_min, self.alpha_0 * adapted_decay_rate)


    def discretize_state(self, state):
        # Assuming all values range from -10 to 10, map them into [0, bins-1]
        def to_bin(val, bins):
            return int(np.clip((val + 10) / 20 * bins, 0, bins - 1))

        return [to_bin(s, bin_count) for s, bin_count in zip(state, [self.vel_bins] * 3 + [self.delta_bins] * 3)]


    def choose_action(self, state):
        # Exploration
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        
        # Exploitation
        discretized_state = self.discretize_state(state)
        return np.argmax(self.q_table[*discretized_state, :])

    def update(self, state, action, reward, next_state, alpha_individual_decay):
        discretized_state = self.discretize_state(state)
        discretized_next_state = self.discretize_state(next_state)

        current_q = self.q_table[*discretized_state, action]
        max_next_q = np.max(self.q_table[*discretized_next_state])

        # Q-learning update rule
        self.q_table[*discretized_state, action] = \
            current_q + self.alpha[*discretized_state, action] * (reward + self.gamma * max_next_q - current_q)

        # Count the visit
        self.visit_count[*discretized_state, action] += 1

        # Decay alpha after each visit
        if alpha_individual_decay:
            self.alpha[*discretized_state, action] *= self.alpha_decay
