import numpy as np


def estimate_precision_model(T, tree, env, model, n_runs=100, n_step=1):
    """
    Estimate action and region prediction precision when the real model is acting.
    The environment is driven by model.predict at every step. act_precision measures
    how often the tree's action for a region agrees with the model's action at that
    state. reg_precision measures how often the tree's n-step region prediction
    matches the region the model actually ends up in.
    """
    data = []
    leaves = tree.leaf_dict

    for _ in range(n_runs):
        predictor_leaves = []
        obs, _ = env.reset()

        done = False
        for i in range(n_step):
            predictor_leaves.append(tree.get(obs))

            if (i + 1) < n_step:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, _ = env.step(action)

        while not done:
            predictors = tuple(l.label for l in predictor_leaves[-n_step:])
            predicted_reg = np.argmax(T[predictors]).flatten()[0]
            tree_act = leaves[predictors[-1]].action

            actual_act, _ = model.predict(obs, deterministic=True)
            nobs, _, done, _, _ = env.step(actual_act)
            actual_reg = tree.get(nobs).label

            predictor_leaves.append(leaves[actual_reg])
            data.append((tree_act, actual_act, predicted_reg, actual_reg))
            obs = nobs

    data = np.array(data)
    act_precision = sum(data[:,0] == data[:,1]) / len(data)
    reg_precision = sum(data[:,2] == data[:,3]) / len(data)
    return act_precision, reg_precision


def estimate_precision_tree(T, tree, env, model, n_runs=100, n_step=1):
    """
    Estimate action and region prediction precision when the tree is acting.
    The environment is driven by the tree's leaf action at every step. act_precision
    measures how often the tree's action agrees with what the model would have done.
    reg_precision measures how often the tree's n-step region prediction matches
    the region actually reached under the tree's policy.
    """
    data = []
    leaves = tree.leaf_dict

    for _ in range(n_runs):
        predictor_leaves = []
        obs, _ = env.reset()

        done = False
        for i in range(n_step):
            leaf = tree.get(obs)
            predictor_leaves.append(leaf)

            if (i + 1) < n_step:
                obs, _, done, _, _ = env.step(leaf.action)

        while not done:
            predictors = tuple(l.label for l in predictor_leaves[-n_step:])
            predicted_reg = np.argmax(T[predictors]).flatten()[0]
            tree_act = leaves[predictors[-1]].action

            nobs, _, done, _, _ = env.step(tree_act)
            actual_act, _ = model.predict(obs, deterministic=True)
            actual_reg = tree.get(nobs).label

            predictor_leaves.append(leaves[actual_reg])
            data.append((tree_act, actual_act, predicted_reg, actual_reg))
            obs = nobs

    data = np.array(data)
    act_precision = sum(data[:,0] == data[:,1]) / len(data)
    reg_precision = sum(data[:,2] == data[:,3]) / len(data)
    return act_precision, reg_precision


def evaluate(tree, obs, mask):
    """
    Evaluate the learned tree by computing the average log likelihood and
    perplexity of the observed transitions under the tree's transition
    probabilities.

    For each observed transition (s_t, s_{t+1}), we find the corresponding
    leaves in the tree and compute the log probability of transitioning from
    the current leaf to the next leaf according to the tree's transition
    probabilities. Zero-probability transitions are excluded from the
    likelihood computation and reported separately.

    Perplexity = exp(-avg_log_likelihood) and can be interpreted as the
    effective number of equally likely next regions per transition.
    Perplexity of 1.0 means perfect prediction.
    """
    leaves = tree.leaf_dict

    total_log_likelihood = 0
    n_transitions = 0
    n_zero = 0
    for run, run_mask in zip(obs, mask):
        run_obs = run[run_mask]
        labels = tree.get_labels(run_obs)

        for i in range(len(labels)-1):
            current_leaf = leaves[labels[i]]
            next_leaf = labels[i+1]
            prob = tree.T[current_leaf.label, next_leaf]
            if prob == 0:
                n_zero += 1
            else:
                total_log_likelihood += np.log(prob)
                n_transitions += 1

    avg_log_likelihood = total_log_likelihood / n_transitions if n_transitions > 0 else float('-inf')
    perplexity = np.exp(-avg_log_likelihood)
    return avg_log_likelihood, perplexity, n_zero, n_transitions + n_zero


def estimate_euclidean_error(tree, obs, mask, n_samples=20):
    """
    Estimate predicted and true Euclidean errors for the top-1 region prediction.

    For each observed transition (s_t -> s_{t+1}):
      - Predicted error: sample n_samples points from the predicted next region
        (argmax of T[r_t]) and compute mean distance to s_{t+1}.
      - True error: sample n_samples points from the actual next region
        (the region s_{t+1} falls in) and compute mean distance to s_{t+1}.
        This is the irreducible error from the partition's spatial resolution —
        the best any method can do at this granularity.
      - Ratio: predicted / true. A ratio of 1.0 means the correct region was
        predicted; higher means the prediction was spatially further off.

    Parameters
    ----------
    tree : TreeObserver
    obs  : np.ndarray, shape (n_runs, T, n_dims)
    mask : np.ndarray, shape (n_runs, T), bool
    n_samples : int

    Returns
    -------
    pred_error : float
    true_error : float
    ratio      : float  (pred_error / true_error)
    """
    all_obs = np.concatenate([run[run_mask] for run, run_mask in zip(obs, mask)], axis=0)
    all_labels = tree.get_labels(all_obs)

    leaf_states = {}
    for label in np.unique(all_labels):
        leaf_states[label] = all_obs[all_labels == label]

    total_pred = 0.0
    total_true = 0.0
    n_transitions = 0

    for run, run_mask in zip(obs, mask):
        run_obs = run[run_mask]
        labels = tree.get_labels(run_obs)

        for i in range(len(labels) - 1):
            r_i = int(labels[i])
            s_next = run_obs[i + 1]
            actual_r_j = int(labels[i + 1])

            predicted_r_j = int(np.argmax(tree.T[r_i]))

            states_in_pred = leaf_states.get(predicted_r_j)
            states_in_true = leaf_states.get(actual_r_j)
            if (states_in_pred is None or len(states_in_pred) == 0 or
                    states_in_true is None or len(states_in_true) == 0):
                continue

            idx_pred = np.random.choice(len(states_in_pred), size=n_samples, replace=True)
            idx_true = np.random.choice(len(states_in_true), size=n_samples, replace=True)

            total_pred += np.mean(np.linalg.norm(states_in_pred[idx_pred] - s_next, axis=1))
            total_true += np.mean(np.linalg.norm(states_in_true[idx_true] - s_next, axis=1))
            n_transitions += 1

    if n_transitions == 0:
        return float('inf'), float('inf'), float('nan')

    pred_error = total_pred / n_transitions
    true_error = total_true / n_transitions
    ratio = pred_error / true_error if true_error > 0 else float('nan')
    return pred_error, true_error, ratio


def _run_labels_from_obs(tree, obs, mask):
    """Extract per-run label sequences from observation arrays."""
    run_labels = []
    for run, run_mask in zip(obs, mask):
        valid = run[run_mask]
        if len(valid) > 1:
            run_labels.append(tree.get_labels(valid))
    return run_labels


def _precompute_T_powers(T, n_regions, max_k):
    """Return list of T^1, T^2, ..., T^max_k."""
    Tk = np.eye(n_regions)
    T_powers = []
    for _ in range(max_k):
        Tk = Tk @ T
        T_powers.append(Tk.copy())
    return T_powers


def _build_emp_Tk(run_labels, n_regions, k, laplace):
    """
    Build and row-normalise the empirical k-step count matrix, and collect
    all observed (i, j) pairs at lag k.
    """
    emp_counts = np.full((n_regions, n_regions), laplace)
    pairs = []
    for labels in run_labels:
        for t in range(len(labels) - k):
            i, j = int(labels[t]), int(labels[t + k])
            emp_counts[i, j] += 1
            pairs.append((i, j))
    row_sums = emp_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return emp_counts / row_sums, pairs


def kstep_precision(tree, obs, mask, max_k=5, top_x=1, laplace=0.0):
    """
    Evaluate k-step region prediction precision for k = 1 .. max_k.

    Two curves are returned for each value of top_x:

    matrix_power_curve : list of float, length max_k
        Fraction of observed k-step transitions where the true next region
        is among the top-x most likely regions under T^k[alpha(s_t)].

    empirical_curve : list of float, length max_k
        Same, but using the empirical k-step count matrix built directly
        from the trajectory data.  This is the best achievable precision
        at horizon k given the partition, without any Markov assumption.

    Comparing the two curves shows whether the 1-step Markov model composes
    correctly across steps.

    Parameters
    ----------
    tree    : TreeObserver with tree.T set (1-step transition matrix)
    obs     : np.ndarray, shape (n_runs, T, n_dims)
    mask    : np.ndarray, shape (n_runs, T), bool
    max_k   : int
    top_x   : int
        Number of most-likely regions to consider a prediction correct.
        top_x=1 is standard top-1 precision (argmax).
    laplace : float
        Laplace smoothing for the empirical k-step matrices.

    Returns
    -------
    matrix_power_curve : list of float, length max_k
    empirical_curve    : list of float, length max_k
    n_transitions      : list of int, length max_k  (number of pairs at each k)
    """
    n_regions = tree.n_leaves
    run_labels = _run_labels_from_obs(tree, obs, mask)
    T_powers = _precompute_T_powers(tree.T, n_regions, max_k)

    matrix_power_curve = []
    empirical_curve    = []
    n_transitions_list = []

    for k in range(1, max_k + 1):
        emp_T_k, pairs = _build_emp_Tk(run_labels, n_regions, k, laplace)

        if not pairs:
            matrix_power_curve.append(float('nan'))
            empirical_curve.append(float('nan'))
            n_transitions_list.append(0)
            continue

        # Top-x indices per row (argsort ascending → take last top_x)
        mat_topx = np.argsort(T_powers[k - 1], axis=1)[:, -top_x:]  # (n_regions, top_x)
        emp_topx = np.argsort(emp_T_k,          axis=1)[:, -top_x:]

        n_correct_mat = sum(1 for i, j in pairs if j in mat_topx[i])
        n_correct_emp = sum(1 for i, j in pairs if j in emp_topx[i])

        n = len(pairs)
        matrix_power_curve.append(n_correct_mat / n)
        empirical_curve.append(n_correct_emp / n)
        n_transitions_list.append(n)

    return matrix_power_curve, empirical_curve, n_transitions_list


def kstep_coverage(tree, obs, mask, max_k=5, prob=0.9, laplace=0.0):
    """
    For each k, compute the expected number of regions needed to cover at
    least `prob` of the probability mass after k steps, averaged over
    observed k-step transitions (weighted by visit frequency).

    This answers the question: "How many regions do I need to name to be
    confident that the system will be in one of them with probability >= p
    after k steps?"

    Two curves are returned:

    mat_coverage : list of float, length max_k
        Mean coverage size under T^k (matrix power of the learned 1-step model).

    emp_coverage : list of float, length max_k
        Mean coverage size under the empirical k-step count matrix.

    The coverage size for a given starting region i is:
        min |S| such that sum_{j in S} T^k[i,j] >= prob,
        where S is built greedily by taking the highest-probability regions first.

    Parameters
    ----------
    tree    : TreeObserver with tree.T set (1-step transition matrix)
    obs     : np.ndarray, shape (n_runs, T, n_dims)
    mask    : np.ndarray, shape (n_runs, T), bool
    max_k   : int
    prob    : float in (0, 1]
        Probability mass threshold.
    laplace : float
        Laplace smoothing for the empirical k-step matrices.

    Returns
    -------
    mat_coverage : list of float, length max_k
    emp_coverage : list of float, length max_k
    n_transitions : list of int, length max_k
    """
    n_regions = tree.n_leaves
    run_labels = _run_labels_from_obs(tree, obs, mask)
    T_powers = _precompute_T_powers(tree.T, n_regions, max_k)

    def _coverage_sizes(prob_matrix):
        """
        For each row of prob_matrix, return the number of regions needed
        to reach `prob` cumulative probability (greedy, highest-first).
        Returns an array of shape (n_regions,).
        """
        sorted_desc = np.sort(prob_matrix, axis=1)[:, ::-1]  # (n_regions, n_regions)
        cumsum = np.cumsum(sorted_desc, axis=1)
        # First column index where cumsum >= prob; clamp to n_regions
        sizes = np.argmax(cumsum >= prob, axis=1) + 1
        # argmax returns 0 if never True (prob > 1.0 or all-zero row) → set to n_regions
        sizes = np.where(cumsum[:, -1] >= prob, sizes, n_regions)
        return sizes

    mat_coverage  = []
    emp_coverage  = []
    n_transitions_list = []

    for k in range(1, max_k + 1):
        emp_T_k, pairs = _build_emp_Tk(run_labels, n_regions, k, laplace)

        if not pairs:
            mat_coverage.append(float('nan'))
            emp_coverage.append(float('nan'))
            n_transitions_list.append(0)
            continue

        mat_sizes = _coverage_sizes(T_powers[k - 1])  # (n_regions,)
        emp_sizes = _coverage_sizes(emp_T_k)           # (n_regions,)

        # Average over observed transitions (weighted by visit frequency)
        mat_coverage.append(float(np.mean([mat_sizes[i] for i, _ in pairs])))
        emp_coverage.append(float(np.mean([emp_sizes[i] for i, _ in pairs])))
        n_transitions_list.append(len(pairs))

    return mat_coverage, emp_coverage, n_transitions_list


def simulate(tree, env, n_sims=5):
    """
    Simulate trajectories from the tree by starting at the initial state and
    repeatedly sampling the next leaf according to the transition probabilities
    until reaching a terminal leaf. For each trajectory, we record the sequence
    of regions and actions taken. We return these sequences as arrays, padded
    to the same length.
    """
    leaves = tree.leaf_dict

    all_regions = []
    all_actions = []
    for i in range(n_sims):
        regions, actions = [], []

        obs, _ = env.reset()
        leaf = tree.get(obs)
        while not leaf.terminal:
            regions.append(leaf.label)
            actions.append(leaf.action)
            next_leaf = np.random.choice(len(leaves), p=tree.T[leaf.label])
            leaf = leaves[next_leaf]

        regions.append(leaf.label)
        actions.append(leaf.action)
        all_regions.append(regions)
        all_actions.append(actions)

    return pad_to_array(all_regions), pad_to_array(all_actions)
