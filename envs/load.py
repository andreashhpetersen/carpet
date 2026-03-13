import uppaal_gym
import gymnasium as gym


def load_env(env_id, **env_kwargs):
    if env_id == 'BouncingBall-v0':
        env_kwargs = {
            'ts_size': env_kwargs.get('ts_size', 0.3),
            'max_n_steps': env_kwargs.get('max_n_steps', 400),
        }
        return gym.make(env_id, **env_kwargs)
    elif env_id == 'RandomWalk-v0':
        return gym.make(env_id, **env_kwargs)
    else:
        raise ValueError(f'env "{env_id}" not supported')
