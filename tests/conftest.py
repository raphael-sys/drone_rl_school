import pytest


class AgentCfg:
    # minimal dummy cfg for testing
    class Agent:
        layer_1_nodes = 16
        layer_2_nodes = 8

        buffer_capacity = 100
        batch_size = 4

        lr_start = 0.01
        lr_min = 0.001
        lr_decay = 0.9

        epsilon_start = 1.0
        epsilon_min = 0.1
        epsilon_decay = 0.95

        gamma = 0.99

    agent = Agent()

@pytest.fixture
def cfg():
    return AgentCfg()

@pytest.fixture
def small_replay(cfg):
    from drone_rl_school.dqn import ReplayBuffer
    buf = ReplayBuffer(cfg)
    # populate with some dummy transitions
    for _ in range(10):
        obs = [0.1]*6
        action = 2
        reward = 1.0
        next_obs = [0.2]*6
        done = False
        buf.push(obs, action, reward, next_obs, done)
    return buf
