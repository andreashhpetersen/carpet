import heapq
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, Normalizer


def _entropy(p):
    """Shannon entropy of a probability vector in nats, ignoring zero entries."""
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def characterize_clusters(points, clustering):

    # simple mean operation
    clust_data = []
    for cluster in np.unique(clustering):
        clust_data.append(np.mean(points[clustering == cluster], axis=0))
    return np.array(clust_data)


def get_deterministic_args(probs):
    """
    Find the elements in `probs' that has a deterministic behavior and return a
    a list of the arguments of these grouped according to their transition
    """

    ds = np.argwhere(np.isclose(np.max(probs, axis=1), 1))
    out = []

    reg2index = {}
    for a in ds:
        reg = np.argmax(probs[a])
        if reg not in reg2index:
            reg2index[reg] = len(out)
            out.append([])

        out[reg2index[reg]].append(a)

    return out


def poly_log_reg(X, y, degree=1, plot=False, max_iter=200, thresh=0.95, max_degree=5, min_improvement=0.01):
    """
    Fit polynomial-logistic pipelines for degrees degree..max_degree and return the best pipeline.
    Preference: lower-degree model unless a higher-degree model improves score by at least min_improvement.
    Stops early if a model reaches 'thresh' accuracy.

    Parameters:
    - X, y: data
    - degree: starting degree (int >=1)
    - plot: if True, plot decision boundary for the current best model at each evaluated degree
    - max_iter: passed to LogisticRegression
    - thresh: early-stop threshold for test accuracy
    - max_degree: maximum polynomial degree to try
    - min_improvement: minimum absolute increase in test score required to prefer a higher-degree model
    """
    # Use stratified split if all classes have enough samples; with test_size=0.1
    # sklearn needs at least 1 sample per class in each split.
    min_class_count = np.min(np.bincount(y.astype(int)))
    stratify = y if min_class_count >= 10 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, stratify=stratify)

    best_pipe = None
    best_score = -np.inf
    current_degree = max(1, int(degree))

    for d in range(current_degree, max_degree + 1):
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=d, include_bias=False)),
            ('estimator', LogisticRegression(max_iter=max_iter, random_state=0))
        ])

        pipe.fit(X_train, y_train)
        score = pipe.score(X_test, y_test)

        improved = (score - best_score) >= min_improvement

        # Prefer lower-degree when improvement is smaller than min_improvement.
        if best_pipe is None or improved:
            best_pipe = pipe
            best_score = score
            chosen_degree = d

        # Early stop if we reach threshold
        if best_score >= thresh:
            break

    if best_score < thresh:
        print(f"Best score {best_score:.3f} reached at degree {chosen_degree}, below thresh {thresh}")

    return best_pipe, best_score


def split_on_action(tree, states, acts, mask, thresh=0.95, ratio_thresh=0.9):
    """
    Learn an initial mapping from regions to actions by splitting leaves that
    have a mixed action distribution. Iterates until every leaf is action-pure
    (or no further splits can be made), so it correctly handles any number of
    actions.

    Each iteration does a binary split: the majority action in a leaf is split
    off from all other actions. This means a leaf with k distinct actions needs
    at most k-1 splits to become pure.

    Parameters:
    - tree: the TreeObserver object to update with the learned action mapping
    - states: numpy array of shape (N, K) containing the observed states
    - acts: numpy array of shape (N,) containing the actions taken in each state
    - mask: numpy array of shape (N,) containing a boolean mask indicating which
            states are valid (True) and which are not (False)
    - thresh: float in [0, 1] indicating the purity threshold for considering a
      leaf deterministic

    Returns:
    - None (the function updates the tree in place)
    """
    changed = True
    while changed:
        changed = False
        tree.root.put(states[mask], acts=acts[mask])
        for leaf in tree.leaves():
            if leaf.terminal:
                continue

            unique = np.unique(leaf.acts)
            if len(unique) == 1:
                leaf.action = unique.item()
                continue

            elif len(unique) > 1:

                # Binary split: majority action vs everything else.
                majority = int(np.bincount(leaf.acts.astype(int)).argmax())
                y = (leaf.acts == majority).astype(int)

                if sum(y) / len(y) > ratio_thresh:
                    continue

                leaf.action = None  # required by split_leaf
                branch = tree.split_leaf(leaf.states, y, leaf, thresh=thresh)

                preds = branch.pipe.predict(leaf.states)
                # preds==1 side is majority action; preds==0 side is everything else
                try:
                    branch.left.action  = int(np.bincount(leaf.acts[preds == 0].astype(int)).argmax())
                except ValueError:
                    print("issue with argmaxing actions after split")
                    import ipdb; ipdb.set_trace()

                branch.right.action = majority

                changed = True
                break  # tree.leaves() is now stale — restart the loop


def split_on_transition_tv(region2states, tree, tv_thresh=0.3, acc_thresh=0.7, thresh_ratio=0.05):
    """
    Alternative to split_on_transition that determines split candidates purely
    in transition probability space, then gates on spatial learnability.

    The original split_on_transition derives the cluster count from the number of
    distinct deterministic target regions, which means regions where all transitions
    are stochastic are never split even if they contain states with clearly different
    transition distributions. This function always clusters with k=2 directly in
    probability space, so both deterministic and stochastic transitions contribute
    equally to the split criterion.

    Three gates control whether a candidate split is committed:

      Gate 1 — Balance: the smaller cluster must be at least `thresh_ratio` times
        the size of the larger cluster. Prevents splitting on tiny outlier groups
        that are likely noise or boundary misclassification artefacts.

      Gate 2 — TV distance: the total variation distance between the two cluster
        mean transition vectors must exceed `tv_thresh`. TV(p,q) = 0.5 * sum|p-q|
        ranges from 0 (identical distributions) to 1 (completely disjoint support).
        Prevents splits where the two groups behave nearly identically.

      Gate 3 — Spatial learnability: a polynomial boundary must achieve at least
        `acc_thresh` accuracy on a held-out test split. This is the key guard
        against the "clusters that cannot be sensibly split in point space" failure
        mode. If the two probability-space clusters are geometrically interleaved,
        no boundary will separate them and the split is skipped.

    Parameters
    ----------
    region2states : list of lists, length n_regions
        Output of sample_next_states.
    tree : TreeObserver
    tv_thresh : float
        Minimum TV distance between cluster means required to attempt a split.
        Raise to demand more distinct transition behavior before splitting.
    acc_thresh : float
        Minimum spatial boundary accuracy required to commit to a split.
        Raise to require cleaner geometric separation between the clusters.
    thresh_ratio : float
        Minimum ratio of smaller to larger cluster size.
    """
    print('split leaves (TV)\n')
    leaves = tree.leaf_dict

    for region in range(len(region2states)):
        states = region2states[region]
        if len(states) < 10:
            continue

        leaf = leaves[region]
        if leaf.terminal:
            continue

        probs  = np.array([s.trans_prob for s in states])
        points = np.array([s.point for s in states])

        # Cluster directly in probability space with k=2.  Unlike
        # split_on_transition, this treats stochastic and deterministic
        # transitions uniformly — any two states with different transition
        # distributions will be pulled into different clusters.
        y = AgglomerativeClustering(n_clusters=2).fit_predict(probs)

        # Gate 1: balance.
        n0, n1 = np.sum(y == 0), np.sum(y == 1)
        ratio = min(n0, n1) / max(n0, n1)
        if ratio <= thresh_ratio:
            continue

        # Gate 2: TV distance between the two cluster mean vectors.
        mean0 = np.mean(probs[y == 0], axis=0)
        mean1 = np.mean(probs[y == 1], axis=0)
        tv = 0.5 * np.sum(np.abs(mean0 - mean1))
        if tv < tv_thresh:
            continue

        # Gate 3: spatial learnability.  Fit a polynomial boundary and check
        # the held-out accuracy.  If the clusters are not geometrically separable
        # the split is skipped, preventing the low-quality boundaries that
        # motivated this function.
        pipe, score = poly_log_reg(points, y, thresh=acc_thresh)
        if score < acc_thresh:
            print(f'  Region {region}: skipping (TV={tv:.2f}, spatial acc={score:.2f} < {acc_thresh})')
            continue

        # Pass the already-fitted pipe to split_leaf to avoid a second fit.
        tree.split_leaf(points, y, leaf, pipe=pipe)

    tree.reorder_leaf_labels()


def split_on_transition_unified(region2states, tree, acc_thresh=0.9, thresh_ratio=0.05, tv_thresh=0.1):
    """
    Unified transition-based splitting combining the structural grounding of
    split_on_transition with an explicit geometric feasibility gate.

    For each region:
    - If deterministic target groups exist (n_det_clusters > 1), use them to
      determine k and cluster in probability space, same as split_on_transition.
      Unlike split_on_transition, the split is only committed if a polynomial
      boundary achieves at least acc_thresh accuracy in point space.
    - If the region is fully stochastic (n_det_clusters <= 1), fall back to k=2
      clustering in probability space, gated by TV distance between cluster means.
      The geometric feasibility gate still applies.

    In both cases the balance ratio gate applies.

    Parameters
    ----------
    region2states : list of lists, length n_regions
    tree : TreeObserver
    acc_thresh : float
        Minimum spatial boundary accuracy to commit a split. Applied in both
        the deterministic and stochastic branches.
    thresh_ratio : float
        Minimum ratio of smaller to larger cluster size.
    tv_thresh : float
        Minimum TV distance between cluster means required in the stochastic
        fallback branch. Not applied when deterministic structure guides k.
    """
    print('split leaves (unified)\n')
    leaves = tree.leaf_dict

    for region in range(len(region2states)):
        states = region2states[region]
        if len(states) < 10:
            continue

        leaf = leaves[region]
        if leaf.terminal:
            continue

        probs  = np.array([s.trans_prob for s in states])
        points = np.array([s.point for s in states])

        det_groups = get_deterministic_args(probs)
        n_clusters = len(det_groups)

        if n_clusters > 1:
            # Deterministic structure available: use it to determine k.
            clustering = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(probs)

            if n_clusters > 2:
                # Collapse to 2 groups by clustering the per-cluster point-space
                # centroids, preserving geometric coherence.
                clust_data = characterize_clusters(points, clustering)
                clust2 = AgglomerativeClustering(n_clusters=2).fit_predict(clust_data)
                c2 = np.argwhere(clust2 == 1)
                y = np.zeros_like(clustering)
                y[np.isin(clustering, c2)] = 1
            else:
                y = clustering
        else:
            # Fully stochastic: fall back to k=2 in probability space.
            y = AgglomerativeClustering(n_clusters=2).fit_predict(probs)

            mean0 = np.mean(probs[y == 0], axis=0)
            mean1 = np.mean(probs[y == 1], axis=0)
            tv = 0.5 * np.sum(np.abs(mean0 - mean1))
            if tv < tv_thresh:
                continue

        # Balance gate.
        n0, n1 = np.sum(y == 0), np.sum(y == 1)
        ratio = min(n0, n1) / max(n0, n1)
        if ratio <= thresh_ratio:
            continue

        # Geometric feasibility gate — applied in both branches.
        pipe, score = poly_log_reg(points, y, thresh=acc_thresh)
        if score < acc_thresh:
            print(f'  Region {region}: skipping (spatial acc={score:.2f} < {acc_thresh})')
            continue

        tree.split_leaf(points, y, leaf, pipe=pipe)

    tree.reorder_leaf_labels()


def compute_heterogeneity(states):
    """
    Within-region heterogeneity: mean TV distance between each state's
    transition vector and the region's mean transition vector.

    Returns 0 if fewer than 2 states.
    """
    if len(states) < 2:
        return 0.0
    probs = np.array([s.trans_prob for s in states])
    mean = np.mean(probs, axis=0)
    tv_distances = 0.5 * np.sum(np.abs(probs - mean), axis=1)
    return float(np.mean(tv_distances))


def _attempt_split(region, region2states, tree, thresh_ratio, entropy_thresh, gate_counts):
    """
    Try to split `region`. Returns True if a split was committed, False otherwise.
    Assumes the caller has already verified het > het_thresh.
    Does NOT call reorder_leaf_labels.

    gate_counts is a dict with keys 'min_samples', 'terminal', 'no_det_groups',
    'balance', 'entropy', 'split' — incremented in place to track why splits are blocked.
    """
    leaves = tree.leaf_dict
    if region not in leaves or leaves[region].terminal:
        gate_counts['terminal'] += 1
        return False

    states = region2states[region]
    if len(states) < 10:
        gate_counts['min_samples'] += 1
        return False

    leaf = leaves[region]
    probs = np.array([s.trans_prob for s in states])
    points = np.array([s.point for s in states])

    det_groups = get_deterministic_args(probs)
    n_clusters = len(det_groups)
    if n_clusters <= 1:
        gate_counts['no_det_groups'] += 1
        return False

    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(probs)

    if n_clusters > 2:
        clust_data = characterize_clusters(points, clustering)
        clust2 = AgglomerativeClustering(n_clusters=2).fit_predict(clust_data)
        c2 = np.argwhere(clust2 == 1)
        y = np.zeros_like(clustering)
        y[np.isin(clustering, c2)] = 1
    else:
        y = clustering

    n0, n1 = np.sum(y == 0), np.sum(y == 1)
    if min(n0, n1) / max(n0, n1) <= thresh_ratio:
        gate_counts['balance'] += 1
        return False

    mean0 = np.mean(probs[y == 0], axis=0)
    mean1 = np.mean(probs[y == 1], axis=0)
    mean_all = (n0 * mean0 + n1 * mean1) / (n0 + n1)
    reduction = _entropy(mean_all) - (n0 * _entropy(mean0) + n1 * _entropy(mean1)) / (n0 + n1)
    if reduction < entropy_thresh:
        gate_counts['entropy'] += 1
        return False

    tree.split_leaf(points, y, leaf, thresh=0.9)
    gate_counts['split'] += 1
    return True


def _propagate_split(region, region2states, tree, old_n, het_thresh, het_scores, heap):
    """
    After splitting `region` (label `r`), propagate induced heterogeneity to
    neighbouring regions.

    When split_leaf is called on label `r`:
      - left child keeps label `r`
      - right child gets label `old_n` (= tree.n_leaves - 1 after the split)

    Assumes the caller has already partitioned region2states[region] into left-
    child states and appended the right-child states at region2states[old_n].

    For any state s in region Q whose next_states previously mapped to `r`,
    some of those next_states may now map to `old_n`. We re-label them,
    recompute trans_prob, then recheck Q's heterogeneity.

    States in regions that never transitioned to `r` just have their trans_prob
    vector extended by one zero to stay consistent with the new tree size.
    """
    new_n = tree.n_leaves   # = old_n + 1

    affected = set()
    for q, q_states in enumerate(region2states):
        if q == region or q == old_n:
            continue
        q_affected = False
        for s in q_states:
            if region in s.next_regions:
                s.next_regions = tree.get_labels(s.next_states)
                s.make_transition_probability_vector(new_n)
                q_affected = True
            else:
                # Extend the vector; no transitions went to the split region.
                s.trans_prob = np.append(s.trans_prob, 0.0)
        if q_affected:
            affected.add(q)

    # Update trans_prob for the two new child regions as well (self-loops).
    for q in (region, old_n):
        for s in region2states[q]:
            s.next_regions = tree.get_labels(s.next_states)
            s.make_transition_probability_vector(new_n)

    # Recompute het for affected regions and push to heap if above threshold.
    for q in affected:
        new_het = compute_heterogeneity(region2states[q])
        het_scores[q] = new_het
        if new_het >= het_thresh:
            heapq.heappush(heap, (-new_het, q))
            print(f'  Propagated: region {q} induced het={new_het:.3f}')


def split_on_transition_guided(region2states, tree, het_thresh=0.1,
                                thresh_ratio=0.05, entropy_thresh=0.05,
                                propagate=False):
    """
    Heterogeneity-guided transition splitting.

    Computes within-region heterogeneity (mean TV distance from region mean)
    for every region, then attempts splits only for regions above het_thresh,
    processing them in order of decreasing heterogeneity.

    Uses the same clustering logic as split_on_transition (deterministic
    target groups determine k, entropy reduction gate filters weak splits).

    Parameters
    ----------
    het_thresh : float
        Minimum heterogeneity required to attempt a split.
    thresh_ratio : float
        Minimum ratio of smaller to larger cluster size.
    entropy_thresh : float
        Minimum entropy reduction (nats) required to commit a split.
    propagate : bool
        If True, after each split re-evaluate heterogeneity for all regions
        that transitioned into the just-split region, and add newly
        heterogeneous regions to the work queue within the same round.
        This catches induced heterogeneity without waiting for the next round.

    Returns
    -------
    n_splits : int
        Number of splits made.
    het_scores : dict
        Heterogeneity score per region index (values updated during propagation
        if propagate=True).
    gate_counts : dict
        Number of times each gate blocked a split attempt, plus 'split' for
        successful splits. Keys: 'terminal', 'min_samples', 'no_det_groups',
        'balance', 'entropy', 'split'.
    """
    print('split leaves (guided)\n')
    leaves = tree.leaf_dict
    gate_counts = {'terminal': 0, 'min_samples': 0, 'no_det_groups': 0,
                   'balance': 0, 'entropy': 0, 'split': 0}

    # Compute initial heterogeneity for all regions.
    het_scores = {}
    for region, states in enumerate(region2states):
        if len(states) < 10 or leaves[region].terminal:
            het_scores[region] = 0.0
        else:
            het_scores[region] = compute_heterogeneity(states)

    # Max-heap via negation. Each entry is (-het, region).
    # het_scores is the source of truth: an entry is stale if its het value
    # no longer matches het_scores[region].
    heap = [(-het, r) for r, het in het_scores.items() if het >= het_thresh]
    heapq.heapify(heap)

    n_splits = 0
    while heap:
        neg_het, region = heapq.heappop(heap)
        het = -neg_het

        # Skip stale entries (het_scores may have been updated by propagation).
        if abs(het_scores.get(region, 0.0) - het) > 1e-9:
            continue
        if het < het_thresh:
            continue

        old_n = tree.n_leaves
        split_happened = _attempt_split(region, region2states, tree, thresh_ratio, entropy_thresh, gate_counts)

        if split_happened:
            n_splits += 1
            # Partition region2states[region] into left/right child states.
            # Left child keeps label `region`; right child has label `old_n`.
            # This must happen before propagation so future het computations
            # for region (now left child only) don't see the mixed state set.
            src_states = region2states[region]
            if src_states:
                point_labels = tree.get_labels(np.array([s.point for s in src_states]))
                region2states[region] = [s for s, l in zip(src_states, point_labels) if l == region]
                region2states.append([s for s, l in zip(src_states, point_labels) if l == old_n])
            else:
                region2states.append([])

            # Update het_scores for both children so stale high-het entries
            # for `region` in the heap are superseded.
            het_scores[region] = compute_heterogeneity(region2states[region])
            het_scores[old_n]  = compute_heterogeneity(region2states[old_n])

            if propagate:
                _propagate_split(region, region2states, tree, old_n, het_thresh, het_scores, heap)

    tree.reorder_leaf_labels()
    return n_splits, het_scores, gate_counts


def split_on_transition(region2states, tree, thresh_ratio=0.05, entropy_thresh=0.05):
    """
    For each region, check if the transition probabilities of the states in
    that region are different enough to warrant a split. If so, split the leaf
    using `tree.split_leaf` and assign the new transition probabilities to the
    new leaves.

    Parameters:
    - region2states: list of size `n_regions` where each element is a list
        of `State` objects generated from observed points in `obs` and annotated
        with their next states as sampled `n_samples` times from `env`
    - tree: the TreeObserver object to update with the learned action mapping
    - thresh_ratio: float in [0, 1] indicating the minimum ratio of the
        sizes of the new clusters to consider a split valid (e.g., if thresh_ratio=0.05,
        then the smaller cluster must have at least 5% of the points of the larger cluster)
    - entropy_thresh: minimum reduction in Shannon entropy (nats) required to
        commit a split. Compares the entropy of the region's aggregate transition
        distribution to the weighted average entropy of the two post-split
        cluster distributions. Splits that don't meaningfully reduce uncertainty
        are skipped.
    """
    print('split leaves\n')
    leaves = tree.leaf_dict

    for region in range(len(region2states)):
        states = region2states[region]
        if len(states) < 10:
            continue

        leaf = leaves[region]
        if leaf.terminal:
            continue

        probs = np.array([s.trans_prob for s in states])
        points = np.array([s.point for s in states])

        # find deterministic and stochastic transitions
        det_groups = get_deterministic_args(probs)

        # check if vectors are different
        n_clusters = len(det_groups)
        if n_clusters <= 1:
            continue

        clustering = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(probs)

        # if there are more than 2 clusters, characterize them and cluster
        # again to get 2 clusters for splitting
        if n_clusters > 2:
            clust_data = characterize_clusters(points, clustering)
            clust2 = AgglomerativeClustering(n_clusters=2).fit_predict(clust_data)

            c1, c2 = np.argwhere(clust2 == 0), np.argwhere(clust2 == 1)

            y = np.zeros_like(clustering)
            y[np.isin(clustering, c2)] = 1
        else:
            y = clustering
            c1, c2 = [0], [1]

        ratio = min(sum(y == 0) / sum(y == 1), sum(y == 1) / sum(y == 0))
        if ratio <= thresh_ratio:
            continue

        # Entropy reduction gate: only split if the two clusters have
        # meaningfully more predictable transition distributions than the region as a whole.
        n0, n1 = np.sum(y == 0), np.sum(y == 1)
        mean0 = np.mean(probs[y == 0], axis=0)
        mean1 = np.mean(probs[y == 1], axis=0)
        mean_all = (n0 * mean0 + n1 * mean1) / (n0 + n1)

        entropy_before = _entropy(mean_all)
        entropy_after  = (n0 * _entropy(mean0) + n1 * _entropy(mean1)) / (n0 + n1)
        reduction = entropy_before - entropy_after

        print(f' Entropy reduction: {reduction:.4f}')
        if reduction < entropy_thresh:
            print(f'  Region {region}: skipping (entropy reduction={reduction:.4f} < {entropy_thresh})')
            continue

        tree.split_leaf(points, y, leaf, thresh=0.9)

    tree.reorder_leaf_labels()
