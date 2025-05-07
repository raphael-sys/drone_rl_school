from drone_rl_school.envs.point_mass_env import PointMassEnv
from drone_rl_school.agents.q_learning import QLearningAgent
import numpy as np
import matplotlib.pyplot as plt


def train(agent, env, episodes, visualize=False):
    rewards = []
    for ep in range(episodes):
        obs = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action = agent.choose_action(obs)
            next_obs, r, done, _ = env.step(action)
            agent.update(obs, action, r, next_obs)
            obs = next_obs
            ep_reward += r
            if visualize:
                env.render()
        rewards.append(ep_reward)
        if (ep+1) % 200 == 0:
            print(f"Episode {ep+1-200} to {ep+1} \tMedian Reward: {np.median(rewards[-200:]):.2f}")

    return rewards

def simulate(agent, env, episodes=1):
    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = agent.choose_action(obs)
            obs, r, done, _ = env.step(action)
            env.render()
            ep_reward += r
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
        rewards = train(agent, env, episodes=10_000, visualize=False)

        # Plot the rewards over time
        plt.figure()
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('3D Q‑Learning: Reward per Episode')
        plt.show()
        # Run a demo with visualization
        simulate(agent, env)
