from drone_rl_school.envs.point_mass_env import PointMassEnv
from drone_rl_school.agents.q_learning import QLearningAgent
import numpy as np
import matplotlib.pyplot as plt


def train(agent, env, episodes, epsilon_decay, alpha_global_decay, alpha_individual_decay, visualize=False):
    rewards = []
    for ep in range(episodes):
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

        if epsilon_decay:
            agent.decay_epsilon()
        if alpha_global_decay:
            agent.decay_alpha()

        # Print on training overview
        if (ep+1) % 200 == 0:
            print(f"Episode {ep+1-200} to {ep+1} \tMedian Reward: {np.median(rewards[-200:]):.2f}")

    return rewards


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

    while True:
        # Train without visualization
        episodes = 2_000
        
        epsilon_decay = True
        alpha_global_decay = False
        alpha_individual_decay = False

        rewards = train(agent, env, episodes,
                        epsilon_decay, alpha_global_decay, alpha_individual_decay)

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
