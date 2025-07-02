import gymnasium as gym
from gymnasium import spaces
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

class PointMassEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, dt=0.1, max_steps=400, disturbance_strength=0.1):
        super().__init__()
        self.dt = dt
        self.max_steps = max_steps

        self.disturbance_strength = disturbance_strength

        # Position: x,y,z
        self.pos = None
        # Velocity: vx,vy,vz
        self.vel = None

        # Position of the target
        self.goal = None

        # Actions: 0:+x, 1:-x, 2:+y, 3:-y, 4:+z, 5:-z

        # Reset the env, including state, goal and step counter
        self.reset()

        # Visualization setup
        self.fig = None
        self.ax3d = None
        self.plot_3d_pos = None
        self.plot_xy_pos = None
        self.plot_yz_pos = None
        self.plot_xz_pos = None
        self.stop_animation = False


    def on_key_press(self, event):
        if event.key == 'q':
            self.stop_animation = True


    def reset(self, goal=[5, 5, 5]):
        # Set the goal
        self.goal = np.array(goal, dtype=np.float32)

        # Start at a random point around the origin and with zero velocity
        self.pos = self.goal + (np.random.rand(3) * 10 - 5)   # +5 to -5 from the goal in each dimension
        self.vel = np.zeros(3)
        # self.vel = self.goal + (np.random.rand(3) * 2 - 1)   # +1 to -1 from 0 in each dimension

        # Reset the step count
        self.steps = 0

        return self._get_obs()

    def _get_obs(self):
        # Observation: vx, vy, vz, dx, dy, dz
        pos_delta = self.goal - self.pos
        return np.concatenate([self.vel, pos_delta]).astype(np.float32)

    def reward(self):
        # Returns the reward for the current state
        
        dist = np.linalg.norm(self.goal - self.pos)
        vel_norm = np.linalg.norm(self.vel)
        # reward = -dist**2 - 0.1 * np.linalg.norm(self.vel)

        # End if goal is approximately reached or too many steps
        if dist < 0.1 and vel_norm < 0.1:   # Calculated to abort if distance and velocity is smaller than e.g. [0.1, 0.1, 0.1]
            reward = 10_000
            done = True
        elif dist > 18:   # Calculated to abort if distance is larger than [10, 10, 10]
            reward = -10_000
            done = True            
        elif self.steps >= self.max_steps:
            reward = -dist 
            done = True
        else:
            reward = -dist 
            done = False

        return done, reward
    
    def metric(self):
        # Returns the metric (the value) of the current state
        
        dist = np.linalg.norm(self.goal - self.pos)
        metric = -dist

        return metric

    def disturbance_acc(self):
        return np.random.rand(3) * self.disturbance_strength

    def step(self, action):
        # Get the acceleration from the selected action
        acc = np.zeros(3)
        idx = action // 2
        sign = 1 if action % 2 == 0 else -1
        acc[idx] = sign * 1.0

        # Discrete physics of the movements
        self.vel += (acc * self.dt + self.disturbance_acc() * self.dt)
        self.pos += self.vel * self.dt
        self.steps += 1

        # Calculate the reward
        done, reward = self.reward()

        return self._get_obs(), reward, done, {}


    def render(self, mode='human'):
        # Initialize plot
        if self.fig is None or self.ax3d is None:
            self.fig = plt.figure(figsize=(12, 8))
            self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
            
            # Define grid: 3 rows × 4 columns
            gs = gridspec.GridSpec(3, 4, width_ratios=[3, 1, 1, 1], height_ratios=[1, 1, 1])

            # 3D view
            self.ax3d = self.fig.add_subplot(gs[:, :3], projection='3d')
            self.ax3d.set_title('3D View')
            self.ax3d.set_xlim(self.goal[0] - 10, self.goal[0] + 10)
            self.ax3d.set_ylim(self.goal[1] - 10, self.goal[1] + 10)
            self.ax3d.set_zlim(self.goal[2] - 10, self.goal[2] + 10)
            self.ax3d.scatter(*self.goal, color='red', s=100, label='Goal')
            self.plot_3d_pos = self.ax3d.scatter(*self.pos, color='blue', s=50, label='Drone')
            self.ax3d.legend(loc='upper left')
            # XY plane
            self.ax_xy = self.fig.add_subplot(gs[0, 3])
            self.ax_xy.set_title('XY Projection')
            self.ax_xy.set_xlim(self.goal[0] - 10, self.goal[0] + 10)
            self.ax_xy.set_ylim(self.goal[1] - 10, self.goal[1] + 10)
            self.ax_xy.scatter(self.goal[0], self.goal[1], color='red', s=80)
            self.plot_xy_pos = self.ax_xy.scatter(self.pos[0], self.pos[1], color='blue', s=40)
            self.ax_xy.xaxis.grid(); self.ax_xy.yaxis.grid()
            self.ax_xy.set_xlabel('X'); self.ax_xy.set_ylabel('Y')
            # XZ plane
            self.ax_xz = self.fig.add_subplot(gs[1, 3])
            self.ax_xz.set_title('XZ Projection')
            self.ax_xz.set_xlim(self.goal[0] - 10, self.goal[0] + 10)
            self.ax_xz.set_ylim(self.goal[2] - 10, self.goal[2] + 10)
            self.ax_xz.scatter(self.goal[0], self.goal[2], color='red', s=80)
            self.plot_xz_pos = self.ax_xz.scatter(self.pos[0], self.pos[2], color='blue', s=40)
            self.ax_xz.xaxis.grid(); self.ax_xz.yaxis.grid()
            self.ax_xz.set_xlabel('X'); self.ax_xz.set_ylabel('Z')
            # YZ plane
            self.ax_yz = self.fig.add_subplot(gs[2, 3])
            self.ax_yz.set_title('YZ Projection')
            self.ax_yz.set_xlim(self.goal[1] - 10, self.goal[1] + 10)
            self.ax_yz.set_ylim(self.goal[2] - 10, self.goal[2] + 10)
            self.ax_yz.scatter(self.goal[1], self.goal[2], color='red', s=80)
            self.plot_yz_pos = self.ax_yz.scatter(self.pos[1], self.pos[2], color='blue', s=40)
            self.ax_yz.xaxis.grid();self.ax_yz.yaxis.grid()
            self.ax_yz.set_xlabel('Y'); self.ax_yz.set_ylabel('Z')

            plt.tight_layout()
            plt.ion()

        # 3D axis
        self.plot_3d_pos._offsets3d = ([self.pos[0]], [self.pos[1]], [self.pos[2]])
        dist = np.linalg.norm(self.goal - self.pos)
        self.ax3d.set_title(f'3D  |  Step {self.steps:03d}  |  Dist {dist:.2f}')

        # XY projection
        self.plot_xy_pos.set_offsets([self.pos[0], self.pos[1]])

        # XZ projection
        self.plot_xz_pos.set_offsets([self.pos[0], self.pos[2]])

        # YZ projection
        self.plot_yz_pos.set_offsets([self.pos[1], self.pos[2]])

        # Redraw
        plt.draw()
        plt.pause(0.01)
