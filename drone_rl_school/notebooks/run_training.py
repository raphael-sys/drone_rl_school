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


def train(agent, env, writer, cfg,
        start_episode=0, best_score=float('-inf'),
        buffer=None, visualize=False, store_model=False):

    metrics = []
    rewards = []
    for current_episode in range(start_episode, start_episode + cfg.agent.episodes_to_train):
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
            elif len(buffer) >= cfg.agent.buffer_warmup_size:
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
        if cfg.agent.epsilon_decay:
            agent.decay_epsilon(current_episode)
        if cfg.agent.lr_global_decay:
            agent.decay_global_alpha(current_episode)
        if cfg.agent.lr_individual_decay:
            agent.decay_individual_alpha()

        # Log values
        writer.add_scalar('metric', metrics[-1], current_episode)
        writer.add_scalar('reward', rewards[-1], current_episode)
        writer.add_scalar('epsilon', agent.epsilon, current_episode)
        writer.add_scalar('alpha_median', np.median(agent.lr), current_episode)
        writer.add_scalar('alpha_mean', np.mean(agent.lr), current_episode)
        writer.add_scalar('alpha_min', np.min(agent.lr), current_episode)
        writer.add_scalar('alpha_max', np.max(agent.lr), current_episode)
        writer.add_scalar('alpha_sum', np.sum(agent.lr), current_episode)

        # Update target net (for dqn)
        if cfg.agent.type == 'dqn' and current_episode % cfg.agent.target_update_freq == 0:
            agent.target_net.load_state_dict(agent.q_net.state_dict())

        # Print a training overview
        n = 100
        if (current_episode + 1) % n == 0:
            print(f"Episode {current_episode + 1 - n} to {current_episode + 1} \t \
                  Mean of Metric: {np.mean(metrics[-n:]):.2f}")
            if store_model and np.mean(metrics[-n:]) > best_score:
                best_score = np.mean(metrics[-n:])
                agent.save_model(best_score, current_episode)
                print(f"New best agent saved.")

    return current_episode, rewards, best_score


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg):
    # Prepare the logging directory
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name  = f"{timestamp}_{cfg.agent.type}_{commit[:7]}"
    log_dir   = os.path.join(cfg.run.log_root, exp_name)
    os.makedirs(log_dir, exist_ok=True)

    # Save a frozen copy of the config
    with open(os.path.join(log_dir, "config.yaml"), "w") as fp:
        fp.write(OmegaConf.to_yaml(cfg))

    # Create the actual tensorboard logger
    writer = SummaryWriter(log_dir)    # run in terminal: "tensorboard --logdir=runs", address: http://localhost:6006

    # Set up the environment
    random_seed = cfg.env.random_number_generator_seed
    env = PointMassEnv(cfg, random_seed)

    if cfg.agent.type == 'dqn':    
        agent = DQNAgent(cfg)
        buffer = ReplayBuffer(cfg)

        next_episode_to_train = 0
        best_score = float('-inf')
        while True:
            ep_count, rewards, best_score = train(agent, env, writer, cfg,
                            start_episode=next_episode_to_train, best_score=best_score,
                            buffer=buffer)
            next_episode_to_train += ep_count

            # Run a demo with visualization
            env.simulate(agent)

    elif cfg.agent.type == 'q_learning':
        agent = QLearningAgent(cfg)

        next_episode_to_train = 0
        best_score = float('-inf')
        while True:
            last_episode, rewards, best_score = train(agent, env, writer, cfg,
                            start_episode=next_episode_to_train, best_score=best_score)
            next_episode_to_train = last_episode + 1

            # Run a demo with visualization
            env.simulate(agent)

    elif cfg.agent.type == 'pid':
        # Run the PID agent (no training needed)
        agent = PIDAgent(cfg)
        env.disturbance_strength = 0
        env.simulate(agent)
        env.disturbance_strength = 0


if __name__ == '__main__':
    main()
