from drone_rl_school.envs.point_mass_env import PointMassEnv
from drone_rl_school.agents.q_learning import QLearningAgent
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def train(agent, env, episodes, epsilon_decay, alpha_global_decay, alpha_individual_decay, 
          writer, start_episode=0, visualize=False):
    rewards = []
    for episode in range(start_episode, start_episode + episodes):
        obs = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action = agent.choose_action(obs)
            next_obs, r, done, _ = env.step(action)
            agent.update(obs, action, r, next_obs, alpha_individual_decay)
            obs = next_obs
            ep_reward += r
            if visualize:
                env.render()
        rewards.append(ep_reward)

        # Log values
        writer.add_scalar('reward', ep_reward, episode)
        writer.add_scalar('alpha_median', np.median(agent.alpha), episode)
        writer.add_scalar('alpha_mean', np.mean(agent.alpha), episode)
        writer.add_scalar('epsilon', agent.epsilon, episode)

        if epsilon_decay:
            agent.decay_epsilon(episode)
        if alpha_global_decay:
            agent.decay_global_alpha(episode)

        # Print on training overview
        if (episode+1) % 200 == 0:
            print(f"Episode {episode+1-200} to {episode+1} \tMedian Reward: {np.median(rewards[-200:]):.2f}")

    return episodes, rewards


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
    agent = QLearningAgent()
    writer = SummaryWriter()    # bash: tensorboard --logdir=runs, http://localhost:6006

    episodes_trained = 0
    while True:
        # Train without visualization
        episodes = 10_000
        
        epsilon_decay = True
        alpha_global_decay = True
        alpha_individual_decay = False

        ep_count, rewards = train(agent, env, episodes,
                        epsilon_decay, alpha_global_decay, alpha_individual_decay, 
                        writer, start_episode=episodes_trained)
        episodes_trained += ep_count

        # Plot the rewards over time
        plt.figure()
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('3D Q‑Learning: Reward per Episode')
        plt.grid()
        plt.tight_layout()
        plt.show()

        # Run a demo with visualization
        simulate(agent, env)
