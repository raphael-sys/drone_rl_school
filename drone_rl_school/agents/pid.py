import numpy as np


class PIDAgent:
    def __init__(self, p=0.1, i=0, d=0.1, dt=0.1):
        self.p = p
        self.i = i
        self.d = d

        self.dt = dt

        self.integral = np.zeros(3)
        self.last_error = None

    def choose_action(self, obs):
        error = obs[3:]

        # I 
        self.integral += error * self.dt

        # D
        if self.last_error is None:
            derivative = np.zeros_like(error)
        else:
            derivative = (error - self.last_error) / self.dt

        self.last_error = error.copy()

        # Calculate the resulting PID controller signal
        signals = (self.p * error
                   + self.i * self.integral
                   + self.d * derivative)
        # Get the action key for the action with the highest amplitude
        idx = np.argmax(np.abs(signals))
        action = idx * 2 + (1 if signals[idx] < 0 else 0)

        return action
    