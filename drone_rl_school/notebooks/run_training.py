import hydra
from drone_rl_school.agents.dqn import DQNAgent, ReplayBuffer
from drone_rl_school.agents.pid import PIDAgent
from drone_rl_school.envs.point_mass_env import PointMassEnv
from drone_rl_school.agents.q_learning import QLearningAgent
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from omegaconf import OmegaConf
import os
import subprocess
from torch.utils.tensorboard import SummaryWriter


def train(agent, env, writer, config,
        start_episode=0, best_score=float('-inf'),
        buffer=None, visualize=False, store_model=False):

    metrics = []
    rewards = []
    for episode in range(start_episode, start_episode + config.agent.episodes_to_train):
        obs = env.reset()
        ep_rewards = []
        ep_metrics = []
        done = False

        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done, _ = env.step(action)

            # Add the step to the experience buffer if one exists
            if buffer is not None:
                buffer.push(obs, action, reward, next_obs, done)

            # The training
            if len(buffer) >= config.agent.buffer_warmup_size:
                agent.update(buffer)
            
            obs = next_obs
            ep_rewards.append(reward)
            ep_metrics.append(env.metric())
            
            if visualize:
                env.render()

        # Store the average epoch metric and reward
        metrics.append(np.mean(ep_metrics))
        rewards.append(np.mean(ep_rewards))

        # Run the requested per episode decays
        if config.agent.epsilon_decay:
            agent.decay_epsilon(episode)
        if config.agent.lr_global_decay:
            agent.decay_global_alpha(episode)
        if config.agent.lr_individual_decay:
            agent.decay_individual_alpha()

        # Log values
        writer.add_scalar('metric', metrics[-1], episode)
        writer.add_scalar('reward', rewards[-1], episode)
        writer.add_scalar('epsilon', agent.epsilon, episode)
        writer.add_scalar('alpha_median', np.median(agent.lr), episode)
        writer.add_scalar('alpha_mean', np.mean(agent.lr), episode)
        writer.add_scalar('alpha_min', np.min(agent.lr), episode)
        writer.add_scalar('alpha_max', np.max(agent.lr), episode)
        writer.add_scalar('alpha_sum', np.sum(agent.lr), episode)

        # Update target net
        if episode % config.agent.target_update_freq == 0:
            agent.target_net.load_state_dict(agent.q_net.state_dict())

        # Print a training overview
        n = 100
        if (episode + 1) % n == 0:
            print(f"Episode {episode + 1 - n} to {episode + 1} \t \
                  Mean of Metric: {np.mean(metrics[-n:]):.2f}")
            if store_model and np.mean(metrics[-n:]) > best_score:
                best_score = np.mean(metrics[-n:])
                agent.save_model(best_score, episode)
                print(f"New best agent saved.")

    return episode, rewards, best_score


def simulate(agent, env, episodes=1):
    for ep in range(episodes):
        if hasattr(agent, 'alpha'):
            print(f'Current learning rate (mean): {np.mean(agent.lr)}')
        if hasattr(agent, 'epsilon'):
            print(f'Current exploration rate: {agent.epsilon}')
        obs = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = agent.choose_action(obs)
            obs, r, done, _ = env.step(action)
            env.render()
            ep_reward += r
            # End simulation on 'q' key press
            if env.stop_animation:
                env.stop_animation = False
                break
        plt.close(env.fig)
        env.fig = None
        env.ax = None
        plt.ioff()
        print(f'Simulated Episode    Total Reward: {ep_reward}')


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(config):
    # Prepare the logging directory
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name  = f"{timestamp}_{config.agent.type}_{commit[:7]}"
    log_dir   = os.path.join(config.run.log_root, exp_name)
    os.makedirs(log_dir, exist_ok=True)

    # Save a frozen copy of the config
    with open(os.path.join(log_dir, "config.yaml"), "w") as fp:
        fp.write(OmegaConf.to_yaml(config))

    # Create the actual tensorboard logger
    writer = SummaryWriter(log_dir)    # run in terminal: "tensorboard --logdir=runs", address: http://localhost:6006

    # Set up the environment
    env = PointMassEnv()   # TODO: Add the parameters from the config, some functions need to be implemented. Pay attention to externalize any randomnes into the config.

    if config.agent.type == 'dqn':
        agent = QLearningAgent(alpha_per_state=False)
    
        agent = DQNAgent(config)
        buffer = ReplayBuffer(config)

        episodes_trained = 0
        best_score = float('-inf')
        while True:
            env.disturbance_strength = 0
            
            ep_count, rewards, best_score = train(agent, env, writer, config,
                            start_episode=episodes_trained, best_score=best_score,
                            buffer=buffer)
            episodes_trained += ep_count

            # Run a demo with visualization
            env.disturbance_strength = 0
            simulate(agent, env)
            env.disturbance_strength = 0

    elif config.agent.type == 'pid':
        # Run the PID agent (no training needed)
        agent = PIDAgent()
        env.disturbance_strength = 0
        simulate(agent, env)
        env.disturbance_strength = 0


if __name__ == '__main__':
    main()
