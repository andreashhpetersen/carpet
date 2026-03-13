import numpy as np


def estimate_precision_n_step(T, tree, env, model, n_runs=100, n_step=1):
    """

    """
    data = []
    leaves = tree.leaf_dict

    for _ in range(n_runs):
        predictor_leaves = []
        obs, _ = env.reset()

        done = False
        for i in range(n_step):
            leaf = tree.get(obs)
            action = leaf.action

            predictor_leaves.append(leaf)

            if (i + 1) < n_step:
                obs, _, done, _, _ = env.step(action)

        while not done:

            predictors = tuple(l.label for l in predictor_leaves[-n_step:])
            predicted_reg = np.argmax(T[predictors]).flatten()[0]
            action = leaves[predictors[-1]].action

            nobs, _, done, _, _ = env.step(action)

            actual_act, _ = model.predict(obs, deterministic=True)
            actual_reg = tree.get(nobs).label

            predictor_leaves.append(leaves[actual_reg])

            data.append((
                action, actual_act,
                predicted_reg, actual_reg
            ))
            obs = nobs

    data = np.array(data)
    act_precision = sum(data[:,0] == data[:,1]) / len(data)
    reg_precision = sum(data[:,2] == data[:,3]) / len(data)
    return act_precision, reg_precision


def evaluate(tree, obs, mask):
    """
    Evaluate the learned tree by computing the average log likelihood of the
    observed transitions under the tree's transition probabilities.

    For each observed transition (s_t, s_{t+1}), we find the corresponding
    leaves in the tree and compute the log probability of transitioning from
    the current leaf to the next leaf according to the tree's transition
    probabilities. We then average these log probabilities over all observed
    transitions to get the average log likelihood.
    """
    leaves = tree.leaf_dict

    # compute the total log likelihood of the observed transitions under the tree's transition probabilities
    total_log_likelihood = 0
    for run, run_mask in zip(obs, mask):
        run_obs = run[run_mask]
        labels = tree.get_labels(run_obs)

        for i in range(len(labels)-1):
            current_leaf = leaves[labels[i]]
            next_leaf = labels[i+1]
            prob = current_leaf.T[next_leaf]
            total_log_likelihood += np.log(prob + 1e-10)  # add small value to avoid log(0)

    avg_log_likelihood = total_log_likelihood / (np.sum(mask) - len(mask))
    print(f"Average log likelihood of observed transitions: {avg_log_likelihood}")


def simulate(tree, env, n_sims=5):
    """
    Simulate trajectories from the tree by starting at the initial state and
    repeatedly sampling the next leaf according to the transition probabilities
    until reaching a terminal leaf. For each trajectory, we record the sequence
    of regions and actions taken. We return these sequences as arrays, padded
    to the same length.
    """
    leaves = tree.leaf_dict
    for leaf in leaves.values():
        leaf.T = normalize_to_prob(leaf.T)

    all_regions = []
    all_actions = []
    for i in range(n_sims):
        regions, actions = [], []

        obs, _ = env.reset()
        leaf = tree.get(obs)
        while not leaf.terminal:
            regions.append(leaf.label)
            actions.append(leaf.action)
            next_leaf = np.random.choice(len(leaves), p=leaf.T)
            leaf = leaves[next_leaf]

        regions.append(leaf.label)
        actions.append(leaf.action)
        all_regions.append(regions)
        all_actions.append(actions)

    return pad_to_array(all_regions), pad_to_array(all_actions)
