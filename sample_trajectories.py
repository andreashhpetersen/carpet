"""
Sample and assess ensemble trajectories.

Usage
-----
    python sample_trajectories.py --manifest PATH [options]

Examples
--------
    # 20 trajectory pairs, plot results, save to default location
    python sample_trajectories.py --manifest data/results/ensembles/Random\ Walk_20260414_151801.json

    # 50 pairs, mesh resolution 80, temperature 0.5, save plot to custom path
    python sample_trajectories.py \\
        --manifest data/results/ensembles/Random\ Walk_20260414_151801.json \\
        --n-traj 50 --resolution 80 --temperature 0.5 \\
        --save-plot results/trajectories.png

    # Don't plot, just print assessment metrics
    python sample_trajectories.py --manifest ... --no-plot

For each run, a start state is drawn from the real environment (env.reset()),
a real trajectory is collected by running the RL model, and a sampled
trajectory of the same length is generated from the ensemble.  Assessment
metrics (region similarity, per-timestep mean state distance) are printed and
optionally plotted.
"""

import argparse
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from configs import load_config
from ensemble import (load_ensemble, build_mesh,
                      sample_ensemble_trajectory, assess_trajectories)
from envs.load import load_env
from models.policy import load_or_train_model


def collect_real_trajectory(env, model, start_obs):
    """Roll out the RL model from start_obs until done. Returns state array."""
    trajectory = [start_obs]
    obs = start_obs
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        trajectory.append(obs)
    return np.array(trajectory)


def plot_results(sampled_trajs, real_trajs, mean_state_dist,
                 region_similarity, env_name, n_dims, save_path):
    """
    Two-panel plot:
      Left  — overlay of sampled (dashed) and real (solid) trajectories.
              For n_dims > 2, plots dimension 0 vs dimension 1 only.
      Right — per-timestep mean state distance between trajectory sets.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_traj, ax_dist = axes

    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Trajectory overlay (up to 10 pairs to keep the plot readable)
    n_show = min(10, len(sampled_trajs))
    for i in range(n_show):
        color = prop_cycle[i % len(prop_cycle)]
        r = real_trajs[i]
        s = sampled_trajs[i]
        if n_dims >= 2:
            ax_traj.plot(r[:, 0], r[:, 1], color=color, alpha=0.7,
                         linewidth=1.2, label=f'real {i+1}' if i == 0 else '_')
            ax_traj.plot(s[:, 0], s[:, 1], color=color, alpha=0.7,
                         linewidth=1.2, linestyle='--',
                         label=f'sampled {i+1}' if i == 0 else '_')
            ax_traj.plot(r[0, 0], r[0, 1], 'o', color=color, markersize=5)
        else:
            xs = np.arange(len(r))
            ax_traj.plot(xs, r[:, 0], color=color, alpha=0.7, linewidth=1.2)
            ax_traj.plot(np.arange(len(s)), s[:, 0], color=color, alpha=0.7,
                         linewidth=1.2, linestyle='--')

    ax_traj.set_title(f'{env_name} — trajectories (solid=real, dashed=sampled)')
    if n_dims >= 2:
        ax_traj.set_xlabel('dim 0')
        ax_traj.set_ylabel('dim 1')
    else:
        ax_traj.set_xlabel('step')
        ax_traj.set_ylabel('dim 0')
    ax_traj.legend(fontsize=8)

    # Per-timestep mean state distance
    ax_dist.plot(mean_state_dist, marker='o', markersize=3)
    ax_dist.set_xlabel('Timestep')
    ax_dist.set_ylabel('Distance between means')
    ax_dist.set_title(
        f'{env_name} — mean state distance\n'
        f'region similarity = {region_similarity:.3f}'
    )
    ax_dist.axhline(np.mean(mean_state_dist), color='grey',
                    linewidth=0.8, linestyle='--', label='mean')
    ax_dist.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Plot saved to {save_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Sample and assess ensemble trajectories.')
    parser.add_argument('--manifest', required=True,
                        help='Path to ensemble manifest JSON.')
    parser.add_argument('--n-traj', type=int, default=20,
                        help='Number of trajectory pairs to sample (default: 20).')
    parser.add_argument('--n-steps', type=int, default=None,
                        help='Max steps per sampled trajectory. Defaults to the '
                             'length of the corresponding real trajectory.')
    parser.add_argument('--resolution', type=int, default=50,
                        help='Mesh grid resolution per dimension (default: 50).')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (default: 1.0). Lower = more '
                             'deterministic, higher = more exploratory.')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting; only print metrics.')
    parser.add_argument('--save-plot', default=None,
                        help='Path to save the plot (default: next to manifest).')
    args = parser.parse_args()

    # Load manifest and infer config from env name
    with open(args.manifest) as f:
        manifest = json.load(f)
    env_name = manifest['env']

    env_name_to_config = {
        'Random Walk':   'random_walk',
        'Bouncing Ball': 'bouncing_ball',
        'Cruise Control': 'cruise_control',
    }
    config_name = env_name_to_config.get(env_name)
    if config_name is None:
        print(f'Unknown env "{env_name}" in manifest. '
              f'Known: {list(env_name_to_config)}', file=sys.stderr)
        sys.exit(1)

    config = load_config(config_name)
    bounds = np.array(config['bounds'])
    n_dims = config['n_dims']

    print(f'Loading ensemble from {args.manifest} ...')
    trees, _ = load_ensemble(args.manifest)
    print(f'  {len(trees)} member trees loaded.')

    env = load_env(config['env_id'])
    model = load_or_train_model(env, config['model_path'],
                                n_timesteps=config['n_timesteps'])

    print(f'Building mesh (resolution={args.resolution}) ...')
    mesh, mesh_labels = build_mesh(trees, bounds, resolution=args.resolution)
    print(f'  Mesh: {len(mesh)} points.')

    print(f'Sampling {args.n_traj} trajectory pairs ...')
    sampled_trajs = []
    real_trajs = []

    for i in range(args.n_traj):
        start_obs, _ = env.reset()

        # Real trajectory
        real_traj = collect_real_trajectory(env, model, start_obs)
        real_trajs.append(real_traj)

        # Sampled trajectory — same length as real unless overridden
        n_steps = args.n_steps if args.n_steps is not None else len(real_traj) - 1
        sampled_traj = sample_ensemble_trajectory(
            trees, mesh, mesh_labels, start_obs,
            n_steps=n_steps, temperature=args.temperature
        )
        sampled_trajs.append(sampled_traj)

        if (i + 1) % 5 == 0 or (i + 1) == args.n_traj:
            print(f'  {i + 1}/{args.n_traj} done')

    print('Assessing trajectories ...')
    region_similarity, mean_state_dist = assess_trajectories(
        trees, sampled_trajs, real_trajs)

    print(f'\n=== Results ({env_name}) ===')
    print(f'  Region sequence similarity : {region_similarity:.4f}  '
          f'(1.0 = perfect region match at every step)')
    print(f'  Mean state distance        : {np.mean(mean_state_dist):.4f}  '
          f'(avg over timesteps)')
    print(f'  Max state distance         : {np.max(mean_state_dist):.4f}  '
          f'(worst timestep)')

    if not args.no_plot:
        if args.save_plot is not None:
            save_path = args.save_plot
        else:
            save_path = args.manifest.replace('.json', '_trajectories.png')

        plot_results(sampled_trajs, real_trajs, mean_state_dist,
                     region_similarity, env_name, n_dims, save_path)


if __name__ == '__main__':
    main()
