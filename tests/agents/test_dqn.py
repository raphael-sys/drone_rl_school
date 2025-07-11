import pytest
import torch
from drone_rl_school.agents.dqn import QNetwork


def test_qnetwork_forward_shape(cfg):
    # state_dim=6, action_dim=6
    nn = QNetwork(state_dim=6, action_dim=6, cfg=cfg)
    x = torch.randn(3, 6)          # batch of 3 states
    out = nn(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 6)     # should map to action_dim

@pytest.mark.parametrize("cfg", [['agent=dqn', 'agent.episodes_to_train=1']], indirect=True)
def test_dqn_full_stack(cfg):
    from drone_rl_school.train.train import train
    from drone_rl_school.agents.dqn import DQNAgent, ReplayBuffer
    from drone_rl_school.envs.point_mass_env import PointMassEnv
    # Set up the elements
    agent = DQNAgent(cfg)
    buffer = ReplayBuffer(cfg)
    env = PointMassEnv(cfg, random_seed=2)
    writer = None
    # Run one training loop
    next_episode_to_train = 0
    best_score = float('-inf')
    last_episode, rewards, best_score = train(agent, env, writer, cfg,
                    start_episode=next_episode_to_train, best_score=best_score,
                    buffer=buffer)
    next_episode_to_train = last_episode + 1
    # Run a simulation
    trajectories = [env.simulate(agent) for _ in range(3)]
    env.animate(trajectories, env.goal, trajectory_names=['dqn'] * 3)
