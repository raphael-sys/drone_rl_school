import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np

class PointMassEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, goal=(5,5,5), dt=0.1, max_steps=400):
        super().__init__()
        self.dt = dt
        self.max_steps = max_steps
        self.goal = np.array(goal, dtype=np.float32)

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

    def reset(self):
        # Start at origin and with zero velocity
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)

        # Reset the step count
        self.steps = 0

        return self._get_obs()

    def _get_obs(self):
        delta = self.goal - self.pos
        return np.concatenate([self.pos, self.vel, delta]).astype(np.float32)

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

        # Set the reward
        dist = np.linalg.norm(self.goal - self.pos)
        reward = -np.sqrt(dist) - 0.1 * np.linalg.norm(self.vel)

        # End if goal is approximately reached or too many steps
        done = dist < 0.2 or self.steps >= self.max_steps
        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        # Initialize plot
        if self.fig is None or self.ax is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlim(-1, 6)
            self.ax.set_ylim(-1, 6)
            self.ax.set_zlim(-1, 6)
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            plt.ion()

        # Clear and redraw
        self.ax.cla()
        self.ax.set_xlim(-1, 6)
        self.ax.set_ylim(-1, 6)
        self.ax.set_zlim(-1, 6)
        self.ax.scatter(*self.goal, color='red', s=100, label='Goal')
        self.ax.scatter(*self.pos, color='blue', s=50, label='Drone')
        self.ax.legend(loc='upper left')
        self.ax.set_title(f'Step {self.steps}')
        plt.draw()
        plt.pause(0.05)
