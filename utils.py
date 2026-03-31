import csv
import json
import numpy as np
import os
from datetime import datetime
from glob import glob


def pad_to_array(list_of_arrays, n=None, pad_value=0, return_mask=False):
    """
    Convert a list of NumPy arrays into a padded numeric array.
    Optionally also return a boolean mask marking valid (unpadded) entries.

    Args:
        list_of_arrays (list of np.ndarray): Input arrays.
        n (int or None): Desired length along the first axis.
                         If None, the maximum length among inputs is used.
        pad_value (scalar, optional): Value used for padding. Default is 0.
        return_mask (bool, optional): Whether to return a mask of valid entries.

    Returns:
        np.ndarray or (np.ndarray, np.ndarray):
            If return_mask is False → padded array.
            If return_mask is True → (padded array, mask) where mask is boolean.
    """
    arrays = [np.atleast_1d(np.array(a)) for a in list_of_arrays]

    # Decide first-axis length
    if n is None:
        n = max(a.shape[0] for a in arrays)

    # Find maximum shape for all axes beyond the first
    max_inner_shape = ()
    if any(len(a.shape) > 1 for a in arrays):
        max_rank = max(len(a.shape) for a in arrays)
        shapes = [a.shape[1:] + (0,) * (max_rank - 1 - (len(a.shape) - 1))
                  for a in arrays]
        max_inner_shape = tuple(map(max, zip(*shapes)))

    padded, masks = [], []
    for a in arrays:
        target_shape = (n, *max_inner_shape)
        trimmed = a[:n]  # truncate if longer
        pad_width = [(0, ts - s) for s, ts in zip(trimmed.shape, target_shape)]
        if len(pad_width) < len(target_shape):
            pad_width.extend((0, ts) for ts in target_shape[len(pad_width):])

        a_padded = np.pad(trimmed, pad_width, constant_values=pad_value)
        padded.append(a_padded)

        if return_mask:
            mask = np.pad(
                np.ones_like(trimmed, dtype=bool),
                pad_width,
                constant_values=False
            )
            masks.append(mask)

    result = np.stack(padded, axis=0)
    if return_mask:
        return result, np.stack(masks, axis=0)
    else:
        return result


def normalize(x, mean, std):
    return (x - mean) / std


def normalize_to_prob(P, axis=-1, eps=1e-12):
    P = np.array(P, dtype=float)
    sums = P.sum(axis=axis, keepdims=True)
    return P / (sums + eps)


class ResultsLogger:
    """
    Writes results to a per-model markdown file under data/results/ while
    also printing to stdout. Use log() for result lines and section() for
    headings.
    """
    def __init__(self, model_dir, model_name):
        path = f'./data/results/{model_dir}.md'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._file = open(path, 'a')
        self.log(f'\n# {model_name} — {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    def section(self, title):
        self.log(f'\n## {title}')

    def log(self, msg):
        print(msg)
        self._file.write(msg + '\n')
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class CSVLogger:
    """
    Writes per-round metrics to a CSV file and a companion JSON metadata file.

    Files are saved to data/results/csv/{env_name}_{run_id}.{csv,json}.
    The description string should explain what is new or special about this run.
    """
    COLUMNS = ['round', 'n_regions', 'n_splits', 'het_max', 'het_mean',
               'll', 'perplexity', 'n_zero', 'n_total', 'prec_1step', 'prec_2step']

    def __init__(self, env_name, config_dict, description, run_id=None):
        if run_id is None:
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_id = run_id
        os.makedirs('./data/results/csv', exist_ok=True)
        base = f'./data/results/csv/{env_name}_{run_id}'
        with open(base + '.json', 'w') as f:
            json.dump({'run_id': run_id, 'env': env_name,
                       'config': config_dict, 'description': description}, f, indent=2)
        self._file = open(base + '.csv', 'w', newline='')
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.COLUMNS)

    def log_round(self, round_num, n_regions, n_splits=None, het_max=None, het_mean=None,
                  ll=None, perplexity=None, n_zero=None, n_total=None,
                  prec_1step=None, prec_2step=None):
        self._writer.writerow([round_num, n_regions, n_splits, het_max, het_mean,
                                ll, perplexity, n_zero, n_total, prec_1step, prec_2step])
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


### Agent data generation and saving/loading utilities


def generate_agent_data(model, env, n_runs=500, eps=0.0, pad_to_size=None, include_terminal=False):
    print('generate agent data')
    all_obs, all_acts, all_rews = [], [], []

    for _ in range(n_runs):
        episode_obs = []
        episode_acts = []
        episode_rews = []

        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        done = False
        while not done:
            nobs, reward, done, _, _ = env.step(action)

            episode_obs.append(obs)
            episode_acts.append(action)
            episode_rews.append(reward)

            if action.ndim == 0:
                action = action.item()

            # choose next action
            if np.random.random() < eps:
                action = model.action_space.sample()
            else:
                action, _ = model.predict(nobs, deterministic=True)

            if done and include_terminal:
                episode_obs.append(nobs)
                episode_acts.append(action)
                episode_rews.append(0)

            obs = nobs

        all_obs.append(np.array(episode_obs))
        all_acts.append(np.array(episode_acts))
        all_rews.append(np.array(episode_rews))

    if pad_to_size is not None:
        obs = pad_to_array(all_obs, n=pad_to_size)
        acts = pad_to_array(all_acts, n=pad_to_size)
        rews, mask = pad_to_array(all_rews, n=pad_to_size, return_mask=True)
        return obs, acts, rews, mask
    else:
        return np.array(all_obs), np.array(all_acts), np.array(all_rews), None


def save_training_data(arrays, model_dir, path=None):
    """
    Save training data arrays for a given model directory. If path is None,
    saves to a new file in the directory with an incremented name.
    """
    directory = f'./data/training/{model_dir}/'
    if path is None:
        existing_files = glob(directory + '*.npz')
        i = len(existing_files) + 1
        path = f'{directory}run_{i}.npz'
    np.savez(path, *arrays)


def load_training_data(model_dir, path=None):
    """
    Load training data arrays for a given model directory. If path is None,
    randomly selects a file from the directory.
    """
    if path is None:
        directory = f'./data/training/{model_dir}/'
        existing_files = glob(directory + '*.npz')
        if not existing_files:
            raise FileNotFoundError(f'No files found in {directory}')
        path = np.random.choice(existing_files)
    data = np.load(path)
    return data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
