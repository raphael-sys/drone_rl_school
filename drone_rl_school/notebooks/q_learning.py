import json
from drone_rl_school.envs.point_mass_env import PointMassEnv
from drone_rl_school.agents.q_learning import QLearningAgent
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def train(agent, env, episodes, epsilon_decay, alpha_global_decay, alpha_individual_decay, 
          writer, start_episode=0, best_score=float('-inf'), visualize=False):
    metrics = []
    rewards = []
    for episode in range(start_episode, start_episode + episodes):
        obs = env.reset()
        ep_rewards = []
        ep_metrics = []
        done = False

        while not done:
            action = agent.choose_action(obs)
            next_obs, r, done, _ = env.step(action)
            agent.update(obs, action, r, next_obs, alpha_individual_decay)
            obs = next_obs
            ep_rewards.append(r)
            ep_metrics.append(env.metric())
            if visualize:
                env.render()

        # Store the average epoch metric and reward
        metrics.append(np.mean(ep_metrics))
        rewards.append(np.mean(ep_rewards))

        # Run the requested per episode decays
        if epsilon_decay:
            agent.decay_epsilon(episode)
        if alpha_global_decay:
            agent.decay_global_alpha(episode)
        if alpha_individual_decay:
            agent.decay_individual_alpha()

        # Log values
        writer.add_scalar('metric', metrics[-1], episode)
        writer.add_scalar('reward', rewards[-1], episode)
        writer.add_scalar('epsilon', agent.epsilon, episode)
        writer.add_scalar('alpha_median', np.median(agent.alpha), episode)
        writer.add_scalar('alpha_mean', np.mean(agent.alpha), episode)
        writer.add_scalar('alpha_min', np.min(agent.alpha), episode)
        writer.add_scalar('alpha_max', np.max(agent.alpha), episode)
        writer.add_scalar('alpha_sum', np.sum(agent.alpha), episode)

        # Print a training overview
        n = 400
        if (episode + 1) % n == 0:
            print(f"Episode {episode + 1 - n} to {episode + 1} \t \
                  Mean of Metric: {np.mean(metrics[-n:]):.2f}")
            if np.mean(metrics[-n:]) > best_score:
                best_score = np.mean(metrics[-n:])
                np.save(f"best_models/q_table_score_{best_score}_q_table.npy", agent.q_table)
                metadata = {
                    "best_score": best_score,
                    "episodes": episode,
                    "delta_bins": agent.delta_bins, 
                    "vel_bins": agent.vel_bins, 
                    "agent": agent.alpha, 
                    "epsilon": agent.epsilon, 
                    "alpha_min": agent.alpha_min, 
                    "epsilon_min": agent.epsilon_min, 
                    "alpha_decay": agent.alpha_decay, 
                    "epsilon_decay": agent.epsilon_decay,
                    }
                with open(f"best_models/q_table_score_{best_score}_metadata.json", "w") as f:
                    json.dump(metadata, f)
                print(f"New best agent saved.")

    return episodes, rewards, best_score


def simulate(agent, env, episodes=1):
    for ep in range(episodes):
        print(f'Current learning rate (mean): {np.mean(agent.alpha)}')
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


if __name__ == '__main__':
    env = PointMassEnv()
    agent = QLearningAgent(alpha_per_state=False)
    writer = SummaryWriter()    # bash: tensorboard --logdir=runs, http://localhost:6006

    episodes_trained = 0
    best_score = float('-inf')
    while True:
        # Train without visualization
        episodes = 2_000
        
        epsilon_decay = True
        alpha_global_decay = True
        alpha_individual_decay = False

        ep_count, rewards, best_score = train(agent, env, episodes,
                        epsilon_decay, alpha_global_decay, alpha_individual_decay, 
                        writer, start_episode=episodes_trained, best_score=best_score)
        episodes_trained += ep_count

        # Run a demo with visualization
        simulate(agent, env)
