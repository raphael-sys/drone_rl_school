import hydra
import pytest


@pytest.fixture
def cfg(request):
    overrides = request.param if hasattr(request, "param") else []
    if isinstance(overrides, str):
        overrides = [overrides]

    # Initialize Hydra and get the config
    with hydra.initialize(config_path="../src/drone_rl_school/configs", version_base=None):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    return cfg

@pytest.fixture
def small_replay(cfg):
    from drone_rl_school.agents.dqn import ReplayBuffer
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
