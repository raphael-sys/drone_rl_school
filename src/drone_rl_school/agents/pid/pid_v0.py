import numpy as np

from drone_rl_school.registry import agent_registry

__version__ = "0.1.0"

class PIDAgent:
    def __init__(self, cfg):
        # Save the config
        self.cfg = cfg

        self.integral = np.zeros(3)
        self.last_error = None

    def choose_action(self, obs):
        error = obs[3:]

        # I 
        self.integral += error * self.cfg.env.dt

        # D
        if self.last_error is None:
            derivative = np.zeros_like(error)
        else:
            derivative = (error - self.last_error) / self.cfg.env.dt

        self.last_error = error.copy()

        # Calculate the resulting PID controller signal
        signals = (self.cfg.agent.p * error
                   + self.cfg.agent.i * self.integral
                   + self.cfg.agent.d * derivative)
        # Get the action key for the action with the highest amplitude
        idx = np.argmax(np.abs(signals))
        action = idx * 2 + (1 if signals[idx] < 0 else 0)

        return action

agent_registry.register("pid_v0", "drone_rl_school.agents.pid.pid_v0:PIDAgent")
