from drone_rl_school.envs.point_mass_env import PointMassEnv
from drone_rl_school.agents.q_learning import QLearningAgent
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
            print(f"Episode {ep+1}\tTotal Reward: {ep_reward:.2f}")

    # Plot the rewards over time
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('3D Q‑Learning: Reward per Episode')
    plt.show()

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
        print(f'Simulated Episode    Total Reward: {ep_reward}')
        env.close()


if __name__ == '__main__':
    env = PointMassEnv()
    agent = QLearningAgent()

    # Train without visualization
    train(agent, env, episodes=20_000, visualize=False)

    # Run a demo with visualization
    simulate(agent, env)
