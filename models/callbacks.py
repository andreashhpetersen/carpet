import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class DataLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(DataLoggerCallback, self).__init__(verbose)
        self.data = []

    def _on_step(self) -> bool:
        done = self.locals['dones'][0]
        if done:
            new_obs = self.locals['infos'][0]['terminal_observation']
        else:
            new_obs = self.locals['new_obs'][0]

        self.data.append(np.hstack((
            self.locals['obs_tensor'].cpu().numpy()[0],
            self.locals['actions'],
            self.locals['rewards'],
            new_obs,
            done
        )))
        return True
