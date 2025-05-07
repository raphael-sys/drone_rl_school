import gymnasium as gym
from gymnasium import spaces
import matplotlib.gridspec as gridspec
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
        self.ax3d = None

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
        if self.fig is None or self.ax3d is None:
            self.fig = plt.figure(figsize=(12, 8))
            
            # Define grid: 3 rows × 4 columns
            gs = gridspec.GridSpec(3, 4, width_ratios=[3, 1, 1, 1], height_ratios=[1, 1, 1])

            # 3D view
            self.ax3d = self.fig.add_subplot(gs[:, :3], projection='3d')
            self.ax3d.set_title('3D View')
            # XY plane
            self.ax_xy = self.fig.add_subplot(gs[0, 3])
            self.ax_xy.set_title('XY Projection')
            # XZ plane
            self.ax_xz = self.fig.add_subplot(gs[1, 3])
            self.ax_xz.set_title('XZ Projection')
            # YZ plane
            self.ax_yz = self.fig.add_subplot(gs[2, 3])
            self.ax_yz.set_title('YZ Projection')

            plt.tight_layout()
            plt.ion()

        # ---- 3D axis ----
        ax = self.ax3d
        ax.cla()
        ax.set_xlim(self.goal[0] - 10, self.goal[0] + 10)
        ax.set_ylim(self.goal[1] - 10, self.goal[1] + 10)
        ax.set_zlim(self.goal[2] - 10, self.goal[2] + 10)
        ax.scatter(*self.goal, color='red', s=100, label='Goal')
        ax.scatter(*self.pos, color='blue', s=50, label='Drone')
        ax.legend(loc='upper left')
        dist = np.linalg.norm(self.goal - self.pos)
        ax.set_title(f'3D  |  Step {self.steps:03d}  |  Dist {dist:.2f}')

        # ---- XY projection ----
        ax = self.ax_xy
        ax.cla()
        ax.set_xlim(self.goal[0] - 10, self.goal[0] + 10)
        ax.set_ylim(self.goal[1] - 10, self.goal[1] + 10)
        ax.scatter(self.goal[0], self.goal[1], color='red', s=80)
        ax.scatter(self.pos[0], self.pos[1], color='blue', s=40)
        ax.xaxis.grid(); ax.yaxis.grid()
        ax.set_title('XY view')
        ax.set_xlabel('X'); ax.set_ylabel('Y')

        # ---- XZ projection ----
        ax = self.ax_xz
        ax.cla()
        ax.set_xlim(self.goal[0] - 10, self.goal[0] + 10)
        ax.set_ylim(self.goal[2] - 10, self.goal[2] + 10)
        ax.scatter(self.goal[0], self.goal[2], color='red', s=80)
        ax.scatter(self.pos[0], self.pos[2], color='blue', s=40)
        ax.xaxis.grid(); ax.yaxis.grid()
        ax.set_title('XZ view')
        ax.set_xlabel('X'); ax.set_ylabel('Z')

        # ---- YZ projection ----
        ax = self.ax_yz
        ax.cla()
        ax.set_xlim(self.goal[1] - 10, self.goal[1] + 10)
        ax.set_ylim(self.goal[2] - 10, self.goal[2] + 10)
        ax.scatter(self.goal[1], self.goal[2], color='red', s=80)
        ax.scatter(self.pos[1], self.pos[2], color='blue', s=40)
        ax.xaxis.grid(); ax.yaxis.grid()
        ax.set_title('YZ view')
        ax.set_xlabel('Y'); ax.set_ylabel('Z')

        # redraw
        plt.draw()
        plt.pause(0.001)
