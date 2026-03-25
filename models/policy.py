import numpy as np

from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO
from models.callbacks import DataLoggerCallback


def init_model(env, model_type=PPO, **model_kwargs):
    if model_type == MaskablePPO:
        policy = MaskableActorCriticPolicy
    else:
        policy = 'MlpPolicy'

    model = model_type(
        policy, env,
        device='cpu', **model_kwargs
    )
    return model


def train_model(model, n_timesteps, path=None, replay_path=None, callbacks=[], progress_bar=True):
    data_logger = DataLoggerCallback()
    callbacks.append(data_logger)
    callback = CallbackList(callbacks)

    model.learn(n_timesteps, callback=callback, progress_bar=progress_bar)
    rew, std = evaluate_policy(model, model.env, n_eval_episodes=25)
    print(rew, std)

    if replay_path is not None:
        np.save(replay_path, np.array(data_logger.data))

    if path is not None:
        model.save(path.replace('.zip', ''))

    return model


def load_or_train_model(
        env, path, model_type=PPO, n_timesteps=250_000,
        replay_path='', continue_training=False, callbacks=[], **model_kwargs
):
    try:
        model = model_type.load(path, device='cpu', **model_kwargs)
        model.set_env(env)
    except FileNotFoundError:
        model = init_model(env, model_type=model_type, **model_kwargs)
        continue_training = True

    if continue_training:
        train_model(model, n_timesteps, path=path, replay_path=replay_path, callbacks=callbacks)

    return model
