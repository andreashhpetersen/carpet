"""
Ensemble of CARPET tree partitionings.

Usage
-----
Build an ensemble of k trees, each trained independently via run_carpet:

    trees = build_ensemble(
        k=5,
        make_tree=lambda: my_initialized_tree_factory(),
        env=env, model=model, logger=logger, model_dir=model_dir,
        env_name='Random Walk',
        **carpet_kwargs,
    )

Each member tree gets its own RunLogger and subfolder under data/results/runs/.
An ensemble manifest is saved to data/results/ensembles/{env}_{id}.json.

Scoring-based transition model
-------------------------------
All inference uses score-weighted sampling over a fixed point set (either the
training data or a mesh grid) rather than enumerating intersection regions.
Given a current intersection region (r1, ..., rk) and a set of candidate points
with precomputed label matrix L of shape (n_points, k):

    score(x_j) = prod_i  T_i[r_i, L[j, i]]

Working in log space (sum of log probs) avoids underflow.  The score of a point
is the joint probability that every tree assigns to that point's region, given
the current region in that tree.  Phantom intersection regions — combinations
that exist in the product of marginals but have no geometric area — naturally
receive zero score and fall out without any special handling.

Evaluation (training points)
-----------------------------
For LL and precision evaluation, the candidate set is the flattened training
data.  Precompute once with build_label_matrix and reuse:

    all_obs, label_matrix = build_label_matrix(trees, obs, mask)
    per_tree, joint = evaluate_ensemble(trees, all_obs, label_matrix, obs, mask)
    in_supp, top1, *eucl = estimate_precision_ensemble(
        trees, all_obs, label_matrix, env, model)

Trajectory sampling (mesh grid)
---------------------------------
For simulation, the candidate set is a dense mesh over the state space bounds.
Using a mesh rather than training points avoids repeated reuse of the same
observed states and gives a geometrically faithful sample from the predicted
next region.  At each step:

  1. Score all mesh points from the current intersection region.
  2. Sample the next state proportionally to the scores (softmax).
  3. Update the current intersection region and repeat.

    mesh, mesh_labels = build_mesh(trees, bounds, resolution=50)
    trajectory = sample_ensemble_trajectory(
        trees, mesh, mesh_labels, start_state, n_steps=100)

Assessment
----------
Compare sampled trajectories against real environment trajectories using:
  - Region sequence similarity: fraction of timesteps where sampled and real
    trajectories share the same intersection region tuple.
  - State distribution similarity: per-timestep Euclidean distance between
    the mean of sampled trajectories and the mean of real trajectories, or
    a KS test per dimension.
"""

import json
import os
from collections import defaultdict

import numpy as np
from datetime import datetime

from analysis.metrics import evaluate
from pipeline import run_carpet
from utils import RunLogger


def _git_snapshot():
    """Return a dict with the current git commit hash and message, or empty strings on failure."""
    import subprocess
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
        commit_msg = subprocess.check_output(
            ['git', 'log', '-1', '--pretty=%s'], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit_hash, commit_msg = '', ''
    return {'git_commit': commit_hash, 'git_message': commit_msg}


def build_ensemble(k, make_tree, env, model, logger, model_dir, env_name,
                   description='', **carpet_kwargs):
    """
    Train k independent trees using run_carpet.

    Parameters
    ----------
    k : int
        Number of ensemble members.
    make_tree : callable
        Zero-argument function returning a freshly initialised TreeObserver
        (including initial action split). Called once per member.
    env, model, logger, model_dir
        Passed through to run_carpet.
    env_name : str
        Used for RunLogger subfolder naming and ensemble manifest.
    description : str, optional
        Free-text note recorded in the manifest (e.g. what changed in this run).
        The current git commit hash and message are captured automatically.
    **carpet_kwargs
        Passed through to run_carpet and stored in the manifest.

    Returns
    -------
    trees : list of TreeObserver
    manifest_path : str
    """
    ensemble_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    member_run_ids = []
    trees = []

    for i in range(k):
        logger.section(f'Ensemble member {i + 1}/{k}')

        tree = make_tree()

        run_logger = RunLogger(
            env_name=env_name,
            config_dict={**carpet_kwargs, 'ensemble_id': ensemble_id, 'member_index': i},
            description=f'Ensemble {ensemble_id} — member {i + 1}/{k}',
        )

        run_carpet(tree, env, model, logger, model_dir,
                   run_logger=run_logger, **carpet_kwargs)
        run_logger.close()

        member_run_ids.append(run_logger.run_id)
        trees.append(tree)

        tree_path = os.path.join(run_logger.run_dir, 'tree.joblib')
        tree.save(tree_path)

    os.makedirs('./data/results/ensembles', exist_ok=True)
    manifest = {
        'ensemble_id': ensemble_id,
        'env': env_name,
        'k': k,
        'members': [{'index': i, 'run_id': rid} for i, rid in enumerate(member_run_ids)],
        'config': carpet_kwargs,
        'description': description,
        **_git_snapshot(),
    }
    manifest_path = f'./data/results/ensembles/{env_name}_{ensemble_id}.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.log(f'Ensemble manifest saved to {manifest_path}')
    return trees, manifest_path


def load_ensemble(manifest_path):
    """
    Load a previously built ensemble from disk.

    Parameters
    ----------
    manifest_path : str
        Path to the ensemble manifest JSON.

    Returns
    -------
    trees : list of TreeObserver
    manifest : dict
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    from models.tree import TreeObserver

    trees = []
    for member in manifest['members']:
        run_dir = f"./data/results/runs/{manifest['env']}_{member['run_id']}"
        tree_path = os.path.join(run_dir, 'tree.joblib')
        trees.append(TreeObserver.load(tree_path))

    return trees, manifest


def ensemble_region(trees, state):
    """
    Return the intersection region tuple for a single state across all trees.

    Parameters
    ----------
    trees : list of TreeObserver
    state : np.ndarray, shape (n_dims,)

    Returns
    -------
    tuple of int — (r1, r2, ..., rk)
    """
    return tuple(tree.get(state).label for tree in trees)


def build_label_matrix(trees, obs, mask):
    """
    Flatten training data and precompute region labels for every point in
    every tree.

    Parameters
    ----------
    trees : list of TreeObserver
    obs   : np.ndarray, shape (n_runs, T, n_dims)
    mask  : np.ndarray, shape (n_runs, T), bool

    Returns
    -------
    all_obs      : np.ndarray, shape (n_points, n_dims)
    label_matrix : np.ndarray, shape (n_points, k), dtype int
        label_matrix[j, i] = region label of all_obs[j] in tree i
    """
    all_obs = np.concatenate([run[run_mask] for run, run_mask in zip(obs, mask)])
    label_matrix = np.stack(
        [tree.get_labels(all_obs) for tree in trees], axis=1
    ).astype(int)
    return all_obs, label_matrix


def _log_score_points(trees, label_matrix, current_tuple, cache=None):
    """
    Compute log-scores for every training point as a candidate next state.

    score(x_j) = sum_i  log T_i[r_i,  label_matrix[j, i]]

    Points unreachable under any single tree get -inf.

    Parameters
    ----------
    trees        : list of TreeObserver
    label_matrix : np.ndarray, shape (n_points, k)
    current_tuple : tuple of int — current intersection region (r1, ..., rk)
    cache : dict or None — if provided, results are memoised by current_tuple

    Returns
    -------
    log_scores : np.ndarray, shape (n_points,)
    """
    if cache is not None and current_tuple in cache:
        return cache[current_tuple]

    log_scores = np.zeros(len(label_matrix), dtype=float)
    for i, (tree, r_cur) in enumerate(zip(trees, current_tuple)):
        probs = tree.T[r_cur, label_matrix[:, i]]
        with np.errstate(divide='ignore'):
            log_scores += np.log(probs)

    if cache is not None:
        cache[current_tuple] = log_scores

    return log_scores


def _logsumexp(log_scores):
    """Numerically stable log-sum-exp, ignoring -inf entries."""
    finite = log_scores[np.isfinite(log_scores)]
    if len(finite) == 0:
        return -np.inf
    m = np.max(finite)
    return m + np.log(np.sum(np.exp(finite - m)))


def evaluate_ensemble(trees, all_obs, label_matrix, obs, mask):
    """
    Evaluate the ensemble on training data.

    Per-tree: runs `evaluate` independently on each tree.
    Joint: computes LL by scoring training points as candidate next states,
    summing scores within each intersection region, and normalising.

    Parameters
    ----------
    trees        : list of TreeObserver
    all_obs      : np.ndarray, shape (n_points, n_dims)  — from build_label_matrix
    label_matrix : np.ndarray, shape (n_points, k)       — from build_label_matrix
    obs          : np.ndarray, shape (n_runs, T, n_dims)
    mask         : np.ndarray, shape (n_runs, T), bool

    Returns
    -------
    per_tree : list of (ll, perplexity, n_zero, n_total) — one per tree
    joint    : (ll, perplexity, n_zero, n_total)
    """
    per_tree = [evaluate(tree, obs, mask) for tree in trees]

    # Map intersection tuple → indices into all_obs / label_matrix
    tuple_to_indices = defaultdict(list)
    for j, row in enumerate(label_matrix):
        tuple_to_indices[tuple(row)].append(j)

    total_ll = 0.0
    n_transitions = 0
    n_zero = 0
    cache = {}

    offset = 0
    for run, run_mask in zip(obs, mask):
        run_obs = run[run_mask]
        n = len(run_obs)
        run_L = label_matrix[offset:offset + n]
        offset += n

        for i in range(n - 1):
            current_tuple = tuple(run_L[i])
            next_tuple = tuple(run_L[i + 1])

            log_scores = _log_score_points(trees, label_matrix, current_tuple, cache)

            next_indices = tuple_to_indices.get(next_tuple, [])
            if not next_indices:
                n_zero += 1
                continue

            log_next = _logsumexp(log_scores[next_indices])
            if not np.isfinite(log_next):
                n_zero += 1
                continue

            log_total = _logsumexp(log_scores)
            if not np.isfinite(log_total):
                n_zero += 1
                continue

            total_ll += log_next - log_total
            n_transitions += 1

    total = n_transitions + n_zero
    avg_ll = total_ll / n_transitions if n_transitions > 0 else float('-inf')
    perplexity = np.exp(-avg_ll)

    return per_tree, (avg_ll, perplexity, n_zero, total)


def estimate_precision_ensemble(trees, all_obs, label_matrix, env, model, n_runs=100):
    """
    Estimate ensemble precision and euclidean error when the real model is acting.

    For each environment step:
      - Scores all training points as candidate next states.
      - Predicted next intersection tuple = label_matrix row of the argmax point.
      - Checks top-1 (predicted == actual) and in-support (actual tuple has any
        training point with finite score).
      - Euclidean error: weighted mean distance from all training points to the
        actual next state, using normalised scores as weights.

    Parameters
    ----------
    trees        : list of TreeObserver
    all_obs      : np.ndarray, shape (n_points, n_dims)
    label_matrix : np.ndarray, shape (n_points, k)
    env          : gym environment
    model        : RL model with predict()
    n_runs       : int

    Returns
    -------
    in_support_prec  : float
    top1_prec        : float
    euclidean_error  : float — score-weighted mean distance to actual next state
    euclidean_true   : float — uniform mean distance within actual next region (baseline)
    euclidean_ratio  : float — euclidean_error / euclidean_true (1.0 = perfect region prediction)
    """
    tuple_to_indices = defaultdict(list)
    for j, row in enumerate(label_matrix):
        tuple_to_indices[tuple(row)].append(j)

    n_top1 = 0
    n_support = 0
    total_euclidean = 0.0
    total_euclidean_true = 0.0
    n_euclidean_true = 0
    n_total = 0
    cache = {}

    for _ in range(n_runs):
        obs_env, _ = env.reset()
        done = False

        while not done:
            current_tuple = ensemble_region(trees, obs_env)
            log_scores = _log_score_points(trees, label_matrix, current_tuple, cache)

            action, _ = model.predict(obs_env, deterministic=True)
            nobs, _, done, _, _ = env.step(action)

            next_tuple = ensemble_region(trees, nobs)

            finite_mask = np.isfinite(log_scores)
            if finite_mask.any():
                predicted_tuple = tuple(label_matrix[np.argmax(log_scores)])
                if predicted_tuple == next_tuple:
                    n_top1 += 1

                next_indices = tuple_to_indices.get(next_tuple, [])
                if next_indices and np.any(finite_mask[next_indices]):
                    n_support += 1

                # Normalise scores to weights and compute weighted mean distance
                finite_log = log_scores[finite_mask]
                m = np.max(finite_log)
                weights = np.zeros(len(all_obs))
                weights[finite_mask] = np.exp(finite_log - m)
                weights /= weights.sum()

                dists = np.linalg.norm(all_obs - nobs, axis=1)
                total_euclidean += float(np.dot(weights, dists))

                # True error: uniform sample from actual next intersection region
                next_indices = tuple_to_indices.get(next_tuple, [])
                if next_indices:
                    true_pts = all_obs[next_indices]
                    idx = np.random.choice(len(true_pts), size=min(20, len(true_pts)), replace=True)
                    total_euclidean_true += float(np.mean(np.linalg.norm(true_pts[idx] - nobs, axis=1)))
                    n_euclidean_true += 1

            n_total += 1
            obs_env = nobs

    top1 = n_top1 / n_total if n_total > 0 else 0.0
    in_support = n_support / n_total if n_total > 0 else 0.0
    euclidean_error = total_euclidean / n_total if n_total > 0 else float('inf')
    euclidean_true = total_euclidean_true / n_euclidean_true if n_euclidean_true > 0 else float('inf')
    euclidean_ratio = euclidean_error / euclidean_true if euclidean_true > 0 else float('nan')
    return in_support, top1, euclidean_error, euclidean_true, euclidean_ratio


def build_mesh(trees, bounds, resolution=50):
    """
    Build a dense mesh grid over the state space and precompute region labels
    for every mesh point in every tree.

    Parameters
    ----------
    trees      : list of TreeObserver
    bounds     : np.ndarray, shape (n_dims, 2) — [[lo, hi], ...] per dimension
    resolution : int
        Number of points per dimension.  Total mesh size = resolution ** n_dims.
        For 2D this is resolution^2; be cautious for n_dims > 3.

    Returns
    -------
    mesh        : np.ndarray, shape (n_points, n_dims)
    mesh_labels : np.ndarray, shape (n_points, k), dtype int
    """
    axes = [np.linspace(bounds[d, 0], bounds[d, 1], resolution)
            for d in range(bounds.shape[0])]
    grid = np.meshgrid(*axes, indexing='ij')
    mesh = np.stack([g.ravel() for g in grid], axis=1)
    mesh_labels = np.stack(
        [tree.get_labels(mesh) for tree in trees], axis=1
    ).astype(int)
    return mesh, mesh_labels


def sample_ensemble_trajectory(trees, mesh, mesh_labels, start_state,
                                n_steps=100, temperature=1.0):
    """
    Sample a trajectory from the ensemble transition model using a mesh grid.

    At each step the mesh is scored from the current intersection region and
    the next state is drawn proportionally to the scores (softmax with optional
    temperature).  Using a mesh rather than training points avoids repeated
    reuse of observed states and gives a geometrically faithful sample from the
    predicted next region.  Phantom intersection regions receive zero score and
    are excluded automatically.

    Parameters
    ----------
    trees       : list of TreeObserver
    mesh        : np.ndarray, shape (n_points, n_dims) — from build_mesh
    mesh_labels : np.ndarray, shape (n_points, k)     — from build_mesh
    start_state : np.ndarray, shape (n_dims,)
    n_steps     : int
    temperature : float
        Scales the log-scores before softmax.  temperature=1.0 samples
        proportionally to the raw scores; lower values concentrate mass on
        high-scoring points (approaching argmax); higher values flatten the
        distribution.

    Returns
    -------
    trajectory : np.ndarray, shape (n_steps + 1, n_dims)
        Row 0 is start_state; rows 1..n_steps are sampled next states.
    """
    trajectory = [start_state]
    current = start_state
    cache = {}

    for _ in range(n_steps):
        current_tuple = ensemble_region(trees, current)
        log_scores = _log_score_points(trees, mesh_labels, current_tuple, cache)

        finite_mask = np.isfinite(log_scores)
        if not finite_mask.any():
            # No reachable mesh points — stop early
            break

        scaled = log_scores[finite_mask] / temperature
        scaled -= np.max(scaled)          # numerical stability
        weights = np.exp(scaled)
        weights /= weights.sum()

        finite_indices = np.where(finite_mask)[0]
        chosen = np.random.choice(finite_indices, p=weights)
        current = mesh[chosen]
        trajectory.append(current)

    return np.array(trajectory)


def assess_trajectories(trees, sampled_trajs, real_trajs):
    """
    Compare sampled ensemble trajectories against real environment trajectories.

    Two complementary metrics are computed:

    Region sequence similarity
        For each pair of (sampled, real) trajectories truncated to the same
        length, the fraction of timesteps where both share the same intersection
        region tuple.  A value of 1.0 means perfect region-level agreement at
        every step.

    Mean state distance
        At each timestep t, the Euclidean distance between the mean of all
        sampled states and the mean of all real states.  Returned as an array
        of length min(T_sampled, T_real) so the caller can plot it over time.

    Parameters
    ----------
    trees        : list of TreeObserver
    sampled_trajs : list of np.ndarray, each shape (T_s, n_dims)
    real_trajs    : list of np.ndarray, each shape (T_r, n_dims)

    Returns
    -------
    region_similarity : float  — mean fraction of matching region tuples
    mean_state_dist   : np.ndarray — per-timestep distance between trajectory means
    """
    # Region sequence similarity
    similarities = []
    for s_traj, r_traj in zip(sampled_trajs, real_trajs):
        T = min(len(s_traj), len(r_traj))
        matches = 0
        for t in range(T):
            if ensemble_region(trees, s_traj[t]) == ensemble_region(trees, r_traj[t]):
                matches += 1
        similarities.append(matches / T if T > 0 else 0.0)
    region_similarity = float(np.mean(similarities)) if similarities else 0.0

    # Per-timestep mean state distance
    T_max = min(
        min(len(t) for t in sampled_trajs),
        min(len(t) for t in real_trajs),
    )
    sampled_means = np.mean([t[:T_max] for t in sampled_trajs], axis=0)
    real_means    = np.mean([t[:T_max] for t in real_trajs],    axis=0)
    mean_state_dist = np.linalg.norm(sampled_means - real_means, axis=1)

    return region_similarity, mean_state_dist
