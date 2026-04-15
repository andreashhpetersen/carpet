import heapq
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree
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

    Returns (None, 0.0) if X has fewer than 2 classes.
    """
    if len(np.unique(y)) < 2:
        return None, 0.0

    # Use stratified split if all classes have enough samples; with test_size=0.1
    # sklearn needs at least 1 sample per class in each split.
    min_class_count = np.min(np.bincount(y.astype(int)))
    stratify = y if min_class_count >= 10 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, stratify=stratify)

    # Without stratification, the minority class can fall entirely into the test
    # set.  If training has only one class, no boundary can be fitted.
    if len(np.unique(y_train)) < 2:
        return None, 0.0

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


def split_on_reachability(tree, obs, mask, bounds,
                          min_empty_ratio=0.3,
                          n_background=10000,
                          density_radius_factor=5.0,
                          acc_thresh=0.85,
                          min_samples=10):
    """
    Split leaves that contain large unreachable regions of the state space.

    For each leaf, background points are sampled uniformly from the full
    state-space bounds and filtered to those that land inside the leaf.
    Background points that are further than a per-leaf density radius from
    any observed training point are labelled 'empty' (unreachable).  If the
    fraction of empty background points exceeds min_empty_ratio, a polynomial
    boundary is fitted between the observed (reachable, y=1) and empty
    (unreachable, y=0) sides.  The split is committed only when the boundary
    achieves at least acc_thresh held-out accuracy.

    The density radius is set per-leaf as density_radius_factor × the median
    k-nearest-neighbour distance among the leaf's own training points, so it
    adapts automatically to local point density.

    This should be called after split_on_action as an initial preprocessing
    step, before CARPET refines the transition structure.  Making region
    geometry tight around the reachable set improves both formal verification
    (no spurious states) and mesh-based trajectory sampling (no mesh points
    in empty space).

    Parameters
    ----------
    tree : TreeObserver
    obs : ndarray, shape (N, K) or (n_runs, n_steps, K)
    mask : boolean ndarray
    bounds : ndarray, shape (n_dims, 2)  —  bounds[i] = [lo, hi]
    min_empty_ratio : float
        Minimum fraction of in-leaf background points classified as empty
        before a split is attempted.
    n_background : int
        Number of background points sampled per iteration.
    density_radius_factor : float
        Multiplier on median NN distance that defines the reachable neighbourhood.
    acc_thresh : float
        Minimum poly_log_reg accuracy to commit a split.
    min_samples : int
        Minimum training / background points required in a leaf to attempt a split.

    Returns
    -------
    n_splits : int
    """
    if obs.ndim == 3:
        flat_obs = obs.reshape(-1, obs.shape[-1])
        flat_mask = mask.reshape(-1).astype(bool)
    else:
        flat_obs = obs
        flat_mask = mask.astype(bool)
    all_states = flat_obs[flat_mask]

    n_dims = bounds.shape[0]
    n_splits = 0
    # Leaves identified as the unreachable side of a previous split.
    # They have few training points by definition, so we must never try to
    # split them again or we get an infinite loop.
    unreachable_labels = set()

    changed = True
    while changed:
        changed = False

        # Resample and relabel on every iteration — tree structure may have changed.
        bg_points    = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                         size=(n_background, n_dims))
        state_labels = tree.get_labels(all_states)
        bg_labels    = tree.get_labels(bg_points)

        for leaf in tree.leaves():
            if leaf.terminal or leaf.label in unreachable_labels:
                continue
            r = leaf.label

            X_pos = all_states[state_labels == r]
            if len(X_pos) < min_samples:
                print(f'  Region {r}: skip — too few training points ({len(X_pos)})')
                continue

            X_bg = bg_points[bg_labels == r]
            if len(X_bg) < min_samples:
                print(f'  Region {r}: skip — too few background points in leaf ({len(X_bg)})')
                continue

            # Per-leaf density radius: factor × median k-NN distance.
            k = min(5, len(X_pos))
            bt = BallTree(X_pos)
            nn_dist, _ = bt.query(X_pos, k=k)
            median_nn = np.median(nn_dist[:, -1])
            if median_nn == 0:
                continue
            radius = density_radius_factor * median_nn

            # Background points further than radius from any training point → empty.
            bg_nn_dist, _ = bt.query(X_bg, k=1)
            X_neg = X_bg[bg_nn_dist[:, 0] > radius]

            empty_ratio = len(X_neg) / (len(X_bg))
            print(f'  Region {r}: n_pos={len(X_pos)}, n_bg={len(X_bg)}, '
                  f'n_empty={len(X_neg)}, empty_ratio={empty_ratio:.2f}, radius={radius:.4f}')

            if len(X_neg) < min_samples:
                continue

            if empty_ratio < min_empty_ratio:
                continue

            # Subsample to keep fitting fast.
            max_fit = 2000
            idx_pos = np.random.choice(len(X_pos), min(max_fit, len(X_pos)), replace=False)
            idx_neg = np.random.choice(len(X_neg), min(max_fit, len(X_neg)), replace=False)
            X_fit = np.vstack([X_neg[idx_neg], X_pos[idx_pos]])
            y_fit = np.array([0] * len(idx_neg) + [1] * len(idx_pos))

            pipe, score = poly_log_reg(X_fit, y_fit, thresh=acc_thresh)
            if pipe is None or score < acc_thresh:
                print(f'  Region {r}: reachability split skipped '
                      f'(empty_ratio={empty_ratio:.2f}, acc={score:.3f})')
                continue

            print(f'  Region {r}: reachability split committed '
                  f'(empty_ratio={empty_ratio:.2f}, acc={score:.3f}, '
                  f'radius={radius:.4f})')
            tree.split_leaf(X_fit, y_fit, leaf, pipe=pipe)
            # Left child (pred=0, unreachable) keeps label r — exclude it from
            # future attempts or the loop recurses into the empty side forever.
            unreachable_labels.add(r)
            n_splits += 1
            changed = True
            break  # tree.leaves() is stale — restart

    tree.reorder_leaf_labels()
    print(f'split_on_reachability: {n_splits} splits committed.')
    return n_splits


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
                pipe, score = poly_log_reg(leaf.states, y, thresh=thresh, max_iter=1000)
                if pipe is None or score < 0.80:
                    leaf.action = majority  # restore so the leaf stays usable
                    continue

                preds = pipe.predict(leaf.states)
                left_acts = leaf.acts[preds == 0]
                right_acts = leaf.acts[preds == 1]
                if len(left_acts) == 0 or len(right_acts) == 0:
                    leaf.action = majority
                    continue

                branch = tree.split_leaf(leaf.states, y, leaf, pipe=pipe)
                # preds==1 side is majority action; preds==0 side is everything else
                branch.left.action  = int(np.bincount(left_acts.astype(int)).argmax())
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

    pipe, score = poly_log_reg(points, y, thresh=0.9)
    if pipe is None or score < 0.9:
        gate_counts['spatial'] += 1
        return False

    tree.split_leaf(points, y, leaf, pipe=pipe)
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
                                propagate=False,
                                max_regions=200):
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
    max_regions : int
        Hard upper bound on the number of leaves. Splitting stops as soon as
        tree.n_leaves reaches this limit, regardless of remaining het scores.

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
                   'balance': 0, 'entropy': 0, 'spatial': 0, 'split': 0}

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
        if tree.n_leaves >= max_regions:
            print(f'Region budget reached ({max_regions}), stopping splits.')
            break

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
