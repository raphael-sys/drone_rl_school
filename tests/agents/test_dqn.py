import torch
from drone_rl_school.agents.dqn import QNetwork


def test_qnetwork_forward_shape(cfg):
    # state_dim=6, action_dim=6
    nn = QNetwork(state_dim=6, action_dim=6, cfg=cfg)
    x = torch.randn(3, 6)          # batch of 3 states
    out = nn(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 6)     # should map to action_dim
