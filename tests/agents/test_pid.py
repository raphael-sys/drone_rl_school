import pytest

@pytest.mark.parametrize("cfg", ["agent=pid"], indirect=True)
def test_pid_full_stack(cfg):
    from drone_rl_school.train.train import train
    from drone_rl_school.agents.pid import PIDAgent
    from drone_rl_school.envs.point_mass_env import PointMassEnv
    # Set up the elements
    agent = PIDAgent(cfg)
    env = PointMassEnv(cfg, random_seed=2)
    # Run a simulation
    trajectories = [env.simulate(agent) for _ in range(3)]
    env.animate(trajectories, env.goal, trajectory_names=['pid'] * 3)
