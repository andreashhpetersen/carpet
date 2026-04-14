import numpy as np

from analysis.metrics import estimate_precision_model, evaluate
from learning.splitting import split_on_transition, split_on_transition_guided
from models.tree import TreeObserver, State
from utils import load_training_data
from viz.plotting import plot_tree_partition


def sample_next_states(tree: TreeObserver, env, obs, mask, n_samples=16):
    """
    Return a list of size `n_regions` where each element is a list of `State`
    objects generated from observed points in `obs` and annotated with their
    next states as sampled `n_samples` times from `env`.
    """
    print('sample next states\n')

    n_regions = tree.n_leaves
    region2states = [[] for _ in range(n_regions)]
    leaves = tree.leaf_dict

    valid_obs = obs[mask]
    region_labels = tree.get_labels(valid_obs)  # single batched traversal

    for x, region_label in zip(valid_obs, region_labels):
        leaf = leaves[region_label]
        state = State(x, region=region_label)

        next_states = []
        ps = None
        for i in range(n_samples):
            ns, _, _ = env.unwrapped.step_from(x, leaf.action)
            next_states.append(ns)

            if np.array_equal(ns, ps):
                break
            else:
                ps = ns

        next_states = np.array(next_states)
        state.next_states = next_states
        state.next_regions = tree.get_labels(next_states)
        state.make_transition_probability_vector(n_regions)

        region2states[region_label].append(state)

    return region2states


def run_carpet_fixed(tree, env, model, logger, model_dir,
                     rounds=10, n_samples=32, estimation_runs=25,
                     laplace=0.0, n_dims=2, model_name='',
                     save_dir='./data/figs/', csv_logger=None):
    """
    Fixed-round CARPET refinement loop (original version).

    Runs exactly `rounds` iterations, splitting all regions each round
    using split_on_transition (entropy-gated, deterministic groupings).
    """
    obs, _, _, mask = load_training_data(model_dir)

    for round_num in range(1, rounds + 1):
        logger.section(f'Round {round_num}')

        region2states = sample_next_states(tree, env, obs, mask, n_samples=n_samples)

        split_on_transition(region2states, tree)
        logger.log(f'Regions: {tree.n_leaves}')

        obs, _, _, mask = load_training_data(model_dir)

        tree.set_transition_scores(obs, mask, n_step=1, laplace=laplace)
        ll, perp, n_zero, n_total = evaluate(tree, obs, mask)
        logger.log(f'Log likelihood: {ll:.4f} | Perplexity: {perp:.4f} | Zero-prob transitions: {n_zero}/{n_total}')

        _, prec_1step = estimate_precision_model(
            tree.T, tree, env, model, n_runs=estimation_runs
        )
        logger.log(f'Precision (1 step, model acting): {prec_1step:.4f}')

        tree.set_transition_scores(obs, mask, n_step=2, laplace=laplace)
        _, prec_2step = estimate_precision_model(
            tree.T, tree, env, model, n_step=2, n_runs=estimation_runs
        )
        logger.log(f'Precision (2 step, model acting): {prec_2step:.4f}')

        if csv_logger is not None:
            csv_logger.log_round(round_num, n_regions=tree.n_leaves,
                                 ll=ll, perplexity=perp, n_zero=n_zero, n_total=n_total,
                                 prec_1step=prec_1step, prec_2step=prec_2step)

        if n_dims == 2:
            plot_tree_partition(
                tree, draw_boundaries=False,
                title=f'{model_name} Round {round_num}',
                save_dir=save_dir
            )


def run_carpet(tree, env, model, logger, model_dir,
               het_thresh=0.1, n_samples=32, estimation_runs=25,
               laplace=0.0, n_dims=2, model_name='',
               save_dir='./data/figs/', max_rounds=50, propagate=False,
               max_regions=200, min_ll_improvement=0.01, ll_patience=3,
               run_logger=None):
    """
    Heterogeneity-guided CARPET refinement loop.

    Iterates until no region exceeds het_thresh or max_rounds is reached.
    Each round:
      1. Samples next states from the environment for each region.
      2. Computes within-region heterogeneity and logs max/mean.
      3. Splits regions above het_thresh in order of decreasing heterogeneity.
      4. Reloads training data and updates transition scores.
      5. Evaluates log likelihood, perplexity, and 1- and 2-step precision.
      6. Saves a partition plot (2D only).

    Parameters
    ----------
    tree : TreeObserver
    env : gym environment
    model : RL model with a predict() method
    logger : ResultsLogger
    model_dir : str
        Used to reload training data each round.
    het_thresh : float
        Minimum within-region heterogeneity (mean TV distance from region mean)
        required to attempt a split. Regions below this are considered converged.
    n_samples : int
        Number of next-state samples per point in sample_next_states.
    estimation_runs : int
        Number of episodes for precision estimation.
    laplace : float
        Laplace smoothing for set_transition_scores.
    n_dims : int
        State space dimensionality. Plotting is skipped for n_dims != 2.
    model_name : str
        Used for plot titles.
    save_dir : str
        Directory to save partition plots.
    max_rounds : int
        Hard upper limit on number of rounds.
    max_regions : int
        Hard upper bound on the number of leaves. Passed to split_on_transition_guided.
    min_ll_improvement : float
        Minimum LL gain to count as an improvement. Used with ll_patience.
    ll_patience : int
        Stop early if LL has not improved by at least min_ll_improvement for
        this many consecutive rounds.
    """
    obs, _, _, mask = load_training_data(model_dir)
    best_ll = None
    rounds_without_improvement = 0

    for round_num in range(1, max_rounds + 1):
        logger.section(f'Round {round_num}')

        region2states = sample_next_states(tree, env, obs, mask, n_samples=n_samples)

        n_splits, het_scores, gate_counts = split_on_transition_guided(
            region2states, tree, het_thresh=het_thresh, propagate=propagate,
            max_regions=max_regions
        )

        het_values = [v for v in het_scores.values() if v > 0]
        if het_values:
            logger.log(f'Heterogeneity — max: {max(het_values):.4f}, mean: {np.mean(het_values):.4f}')

        blocked = {k: v for k, v in gate_counts.items() if k != 'split' and v > 0}
        if blocked:
            logger.log('Gates: ' + ', '.join(f'{k}={v}' for k, v in blocked.items()))

        if n_splits == 0:
            logger.log('Converged — no regions above heterogeneity threshold.')
            break

        if tree.n_leaves >= max_regions:
            logger.log(f'Converged — region budget of {max_regions} reached.')

        logger.log(f'Regions: {tree.n_leaves} (+{n_splits} splits this round)')

        obs, _, _, mask = load_training_data(model_dir)

        tree.set_transition_scores(obs, mask, n_step=1, laplace=laplace)
        ll, perp, n_zero, n_total = evaluate(tree, obs, mask)
        logger.log(f'Log likelihood: {ll:.4f} | Perplexity: {perp:.4f} | Zero-prob transitions: {n_zero}/{n_total}')

        if best_ll is None or ll - best_ll >= min_ll_improvement:
            best_ll = ll
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
            logger.log(f'No LL improvement ({rounds_without_improvement}/{ll_patience} rounds patience used)')
            if rounds_without_improvement >= ll_patience:
                logger.log(f'Converged — no LL improvement for {ll_patience} consecutive rounds.')
                break

        _, prec_1step = estimate_precision_model(
            tree.T, tree, env, model, n_runs=estimation_runs
        )
        logger.log(f'Precision (1 step, model acting): {prec_1step:.4f}')

        tree.set_transition_scores(obs, mask, n_step=2, laplace=laplace)
        _, prec_2step = estimate_precision_model(
            tree.T, tree, env, model, n_step=2, n_runs=estimation_runs
        )
        logger.log(f'Precision (2 step, model acting): {prec_2step:.4f}')

        if run_logger is not None:
            run_logger.log_round(round_num, n_regions=tree.n_leaves,
                                 n_splits=n_splits, het_max=max(het_values) if het_values else None,
                                 het_mean=float(np.mean(het_values)) if het_values else None,
                                 ll=ll, perplexity=perp, n_zero=n_zero, n_total=n_total,
                                 prec_1step=prec_1step, prec_2step=prec_2step)
            if n_dims == 2:
                plot_tree_partition(
                    tree, draw_boundaries=False,
                    title=f'round_{round_num:02d}',
                    save_dir=run_logger.figs_dir
                )

    # Always leave tree with a 1-step T so it's in a consistent state after training
    tree.set_transition_scores(obs, mask, n_step=1, laplace=laplace)
