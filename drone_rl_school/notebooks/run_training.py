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
            if buffer is None:
                agent.update(obs, action, reward, next_obs)
            elif len(buffer) >= config.agent.buffer_warmup_size:
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
        if config.agent.type == 'dqn' and episode % config.agent.target_update_freq == 0:
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
    random_seed = config.env.random_number_generator_seed
    env = PointMassEnv(config, random_seed)

    if config.agent.type == 'dqn':    
        agent = DQNAgent(config)
        buffer = ReplayBuffer(config)

        next_episode_to_train = 0
        best_score = float('-inf')
        while True:
            ep_count, rewards, best_score = train(agent, env, writer, config,
                            start_episode=next_episode_to_train, best_score=best_score,
                            buffer=buffer)
            next_episode_to_train += ep_count

            # Run a demo with visualization
            simulate(agent, env)

    elif config.agent.type == 'q_learning':
        agent = QLearningAgent(config)

        next_episode_to_train = 0
        best_score = float('-inf')
        while True:
            last_episode, rewards, best_score = train(agent, env, writer, config,
                            start_episode=next_episode_to_train, best_score=best_score)
            next_episode_to_train = last_episode + 1

            # Run a demo with visualization
            simulate(agent, env)

    elif config.agent.type == 'pid':
        # Run the PID agent (no training needed)
        agent = PIDAgent(config)
        env.disturbance_strength = 0
        simulate(agent, env)
        env.disturbance_strength = 0


if __name__ == '__main__':
    main()
