import pytest
import torch
from drone_rl_school.agents.q_learning import QLearningAgent


@pytest.mark.parametrize("cfg", [['agent=q_learning', 'agent.episodes_to_train=1']], indirect=True)
def test_q_learning_full_stack(cfg):
    from drone_rl_school.train.train import train
    from drone_rl_school.agents.q_learning import QLearningAgent
    from drone_rl_school.envs.point_mass_env import PointMassEnv
    # Set up the elements
    agent = QLearningAgent(cfg)
    env = PointMassEnv(cfg, random_seed=2)
    writer = None
    # Run one training loop
    next_episode_to_train = 0
    best_score = float('-inf')
    last_episode, rewards, best_score = train(agent, env, writer, cfg,
                    start_episode=next_episode_to_train, best_score=best_score,
                    buffer=None)
    next_episode_to_train = last_episode + 1
    # Run a simulation
    trajectories = [env.simulate(agent) for _ in range(3)]
    env.animate(trajectories, env.goal, trajectory_names=['q_learning'] * 3)
