import numpy as np

class QLearningAgent:
    def __init__(self, pos_bins=2, vel_bins=2, delta_bins=4, alpha=0.05, gamma=0.99, epsilon=0.2):
        # The number of bins per dimension
        self.pos_bins = pos_bins
        self.vel_bins = vel_bins
        self.delta_bins = delta_bins

        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration factor

        # Define action space: 0:+x, 1:-x, 2:+y, 3:-y, 4:+z, 5:-z
        self.num_actions = 6

        # Q-table: (pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, delta_x, delta_y, delta_z, action) → Q-value
        self.q_table = np.zeros((self.pos_bins, self.pos_bins, self.pos_bins, 
                                 self.vel_bins, self.vel_bins, self.vel_bins, 
                                 self.delta_bins, self.delta_bins, self.delta_bins, 
                                 self.num_actions))

    def discretize_state(self, state):
        # Assuming all values range from -10 to 10, map them into [0, bins-1]
        def to_bin(val, bins):
            return int(np.clip((val + 10) / 20 * bins, 0, bins - 1))

        return [to_bin(s, self.pos_bins) for s in state]


    def choose_action(self, state):
        # Exploration
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        
        # Exploitation
        discretized_state = self.discretize_state(state)
        return np.argmax(self.q_table[*discretized_state, :])

    def update(self, state, action, reward, next_state):
        discretized_state = self.discretize_state(state)
        discretized_next_state = self.discretize_state(next_state)

        current_q = self.q_table[*discretized_state, action]
        max_next_q = np.max(self.q_table[*discretized_next_state])

        # Q-learning update rule
        self.q_table[*discretized_state, action] = \
            current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)