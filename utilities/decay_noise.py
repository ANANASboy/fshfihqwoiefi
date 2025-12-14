import torch as th
import numpy as np


class DecayGaussianNoise:
    """
    Simple decaying Gaussian noise generator.
    current_std decays multiplicatively each call to decay() but never below end.
    """
    def __init__(self, start: float, end: float, decay: float):
        self.start = float(start)
        self.end = float(end)
        self.decay = float(decay)
        self.current_std = self.start

    def sample(self, shape, device=None):
        """Return noise tensor with current std."""
        return th.randn(size=shape, device=device) * self.current_std

    def sample_numpy(self, shape):
        """Return noise as numpy array."""
        return np.random.randn(*shape) * self.current_std

    def decay_step(self):
        """Decay std one step."""
        self.current_std = max(self.end, self.current_std * self.decay)

    def reset(self):
        """Reset to start std."""
        self.current_std = self.start
