import gymnasium as gym
from IPython.display import display
from ipywidgets import Play, IntSlider, jslink, VBox
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

class PointMassEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, cfg, random_seed):
        super().__init__()

        # Store the config
        self.cfg = cfg

        # Create a random number generator
        self.rng = np.random.default_rng(random_seed)

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
        self.pos = self.goal + (self.rng.random(3) * 10 - 5)   # +5 to -5 from the goal in each dimension
        self.vel = np.zeros(3)
        # self.vel = self.goal + (self.rng.random(3) * 2 - 1)   # +1 to -1 from 0 in each dimension

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
        elif self.steps >= self.cfg.env.max_steps:
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
        # Generate random accelerations in each dimensions, this roughly simulates wind.
        return self.rng.random(3) * self.cfg.env.disturbance_strength

    def step(self, action):
        # Get the acceleration from the selected action
        acc = np.zeros(3)
        idx = action // 2
        sign = 1 if action % 2 == 0 else -1
        acc[idx] = sign * 1.0

        # Discrete physics of the movements
        self.vel += (acc * self.cfg.env.dt + self.disturbance_acc() * self.cfg.env.dt)
        self.pos += self.vel * self.cfg.env.dt
        self.steps += 1

        # Calculate the reward
        done, reward = self.reward()

        return self._get_obs(), reward, done, {}

    def simulate_and_plot(self, agent, episodes=1):
        for ep in range(episodes):
            obs = self.reset()
            done = False
            ep_reward = 0
            try:
                while not done:
                    action = agent.choose_action(obs)
                    obs, r, done, _ = self.step(action)
                    self.render()
                    ep_reward += r
                    # End simulation on 'q' key press
                    if self.stop_animation:
                        self.stop_animation = False
                        break
            except Exception as e:
                print("Simulation interrupted: ", e)
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            plt.ioff()
            print(f'Simulated Episode    Total Reward: {ep_reward}')

    def simulate(self, agent, episodes=1):
        trajectory = []
        obs = self.reset()
        done = False
        ep_reward = 0
        try:
            while not done:
                action = agent.choose_action(obs)
                obs, r, done, _ = self.step(action)
                ep_reward += r
                trajectory.append(self.pos.copy())
                # End simulation on 'q' key press
                if self.stop_animation:
                    self.stop_animation = False
                    break
        except Exception as e:
            print("Simulation interrupted: ", e)
        print(f'Simulated Episode    Total Reward: {ep_reward}')
        return trajectory

    def animate(self, trajectory, goal):
        max_frames = len(trajectory)

        # Set up the figure
        fig = plt.figure(figsize=(12, 8))
        fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Define grid: 3 rows × 4 columns
        gs = gridspec.GridSpec(3, 4, width_ratios=[3, 1, 1, 1], height_ratios=[1, 1, 1])

        # 3D view
        ax3d = fig.add_subplot(gs[:, :3], projection='3d')
        ax3d.set_title('3D View')
        ax3d.set_xlim(goal[0] - 10, goal[0] + 10)
        ax3d.set_ylim(goal[1] - 10, goal[1] + 10)
        ax3d.set_zlim(goal[2] - 10, goal[2] + 10)
        ax3d.scatter(*goal, color='red', s=100, label='Goal')
        plot_3d_pos = ax3d.scatter(*trajectory[0], color='blue', s=50, label='Drone')
        ax3d.legend(loc='upper left')
        # XY plane
        ax_xy = fig.add_subplot(gs[0, 3])
        ax_xy.set_title('XY Projection')
        ax_xy.set_xlim(goal[0] - 10, goal[0] + 10)
        ax_xy.set_ylim(goal[1] - 10, goal[1] + 10)
        ax_xy.scatter(goal[0], goal[1], color='red', s=80)
        plot_xy_pos = ax_xy.scatter(trajectory[0][0], trajectory[0][1], color='blue', s=40)
        ax_xy.xaxis.grid(); ax_xy.yaxis.grid()
        ax_xy.set_xlabel('X'); ax_xy.set_ylabel('Y')
        # XZ plane
        ax_xz = fig.add_subplot(gs[1, 3])
        ax_xz.set_title('XZ Projection')
        ax_xz.set_xlim(goal[0] - 10, goal[0] + 10)
        ax_xz.set_ylim(goal[2] - 10, goal[2] + 10)
        ax_xz.scatter(goal[0], goal[2], color='red', s=80)
        plot_xz_pos = ax_xz.scatter(trajectory[0][0], trajectory[0][2], color='blue', s=40)
        ax_xz.xaxis.grid(); ax_xz.yaxis.grid()
        ax_xz.set_xlabel('X'); ax_xz.set_ylabel('Z')
        # YZ plane
        ax_yz = fig.add_subplot(gs[2, 3])
        ax_yz.set_title('YZ Projection')
        ax_yz.set_xlim(goal[1] - 10, goal[1] + 10)
        ax_yz.set_ylim(goal[2] - 10, goal[2] + 10)
        ax_yz.scatter(goal[1], goal[2], color='red', s=80)
        plot_yz_pos = ax_yz.scatter(trajectory[0][1], trajectory[0][2], color='blue', s=40)
        ax_yz.xaxis.grid();ax_yz.yaxis.grid()
        ax_yz.set_xlabel('Y'); ax_yz.set_ylabel('Z')

        plt.tight_layout()
        plt.ion()

        # Update the view
        def update(frame):
            # 3D axis
            plot_3d_pos._offsets3d = ([trajectory[frame][0]], [trajectory[frame][1]], [trajectory[frame][2]])
            dist = np.linalg.norm(goal - trajectory[frame])
            ax3d.set_title(f'3D  |  Step {frame:03d}  |  Dist {dist:.2f}')

            # XY projection
            plot_xy_pos.set_offsets([trajectory[frame][0], trajectory[frame][1]])

            # XZ projection
            plot_xz_pos.set_offsets([trajectory[frame][0], trajectory[frame][2]])

            # YZ projection
            plot_yz_pos.set_offsets([trajectory[frame][1], trajectory[frame][2]])

        # Buttons
        play = Play(
            value=0,
            min=0,
            max=max_frames-1,
            step=1,
            interval=250,       # milliseconds between frames
            description="▶️",
            disabled=False
        )

        # Link the animation to the play button
        play.observe(lambda ev: update(ev['new']), names='value')

        # Show the viewer
        display(VBox([play]))
        # Trigger initial draw
        update(0)

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
            self.fig.show()

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
        self.fig.canvas.draw()
        plt.pause(0.01)
