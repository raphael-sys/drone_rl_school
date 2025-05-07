import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np

class PointMassEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, dt=0.1, max_steps=400):
        super().__init__()
        self.dt = dt
        self.max_steps = max_steps

        # State: x,y,z
        # Velocity: vx,vy,vz
        # Delta (to goal): dx,dy,dz
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )

        # Actions: 0:+x, 1:-x, 2:+y, 3:-y, 4:+z, 5:-z
        self.action_space = spaces.Discrete(6)
        self.reset()

        # Visualization setup
        self.fig = None
        self.ax = None

    def reset(self, goal=(5,5,5)):
        # Set the goal
        self.goal = np.array(goal, dtype=np.float32)

        # Start at a random point around the origin and with zero velocity
        self.pos = self.goal + (np.random.rand(3) * 10 - 5)   # +5 to -5 from the goal in each dimension
        self.vel = np.zeros(3)

        # Reset the step count
        self.steps = 0

        return self._get_obs()

    def _get_obs(self):
        pos_delta = self.goal - self.pos
        return np.concatenate([self.vel, pos_delta]).astype(np.float32)

    def step(self, action):
        # Get the acceleration from the selected action
        acc = np.zeros(3)
        idx = action // 2
        sign = 1 if action % 2 == 0 else -1
        acc[idx] = sign * 1.0

        # Discrete physics of the movements
        self.vel += acc * self.dt
        self.pos += self.vel * self.dt
        self.steps += 1

        # Calculate the reward
        dist = np.linalg.norm(self.goal - self.pos)
        reward = -dist   # -dist**2 # - 0.1 * np.linalg.norm(self.vel)

        # End if goal is approximately reached or too many steps
        if dist < 0.2:   # Calculated to abort if distance is smaller than [0.1, 0.1, 0.1]
            reward = 10_000
            done = True
        elif dist > 18:   # Calculated to abort if distance is larger than [10, 10, 10]
            reward = -10_000
            done = True            
        elif self.steps >= self.max_steps:
            done = True
        else:
            done = False

        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        # Initialize plot
        if self.fig is None or self.ax is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlim(self.goal[0] - 10, self.goal[0] + 10)
            self.ax.set_ylim(self.goal[1] - 10, self.goal[1] + 10)
            self.ax.set_zlim(self.goal[2] - 10, self.goal[2] + 10)
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            plt.ion()

        # Clear and redraw
        self.ax.cla()
        self.ax.set_xlim(self.goal[0] - 10, self.goal[0] + 10)
        self.ax.set_ylim(self.goal[1] - 10, self.goal[1] + 10)
        self.ax.set_zlim(self.goal[2] - 10, self.goal[2] + 10)
        self.ax.scatter(*self.goal, color='red', s=100, label='Goal')
        self.ax.scatter(*self.pos, color='blue', s=50, label='Drone')
        self.ax.legend(loc='upper left')
        self.ax.set_title(f'Step {self.steps} - Distance: {np.linalg.norm(self.goal - self.pos):.2f}')
        plt.draw()
        plt.pause(0.05)
