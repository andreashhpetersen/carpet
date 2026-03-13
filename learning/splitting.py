import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, Normalizer


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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

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
            print(f"Reached threshold with degree {chosen_degree}, score {best_score:.3f}")
            break

    if best_score < thresh:
        print(f"Best score {best_score:.3f} reached at degree {chosen_degree}, below thresh {thresh}")

    return best_pipe


def split_on_action(tree, states, acts, mask, thresh=0.95):
    """
    Learn an initial mapping from regions to actions by splitting leaves that
    have a mixed action distribution. For each leaf, if the actions taken in
    that leaf are not all the same, split the leaf using `tree.split_leaf` and
    assign the most common action in each new leaf.

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
    tree.root.put(states[mask], acts=acts[mask])

    for leaf in tree.leaves():
        if leaf.terminal:
            continue

        unique = np.unique(leaf.acts)
        if len(unique) > 1:
            leaf.action = None  # set attribute so split_leaf works
            branch = tree.split_leaf(
                leaf.states, leaf.acts, leaf, thresh=thresh
            )
            preds = branch.pipe.predict(leaf.states)
            branch.left.action = int(np.bincount(leaf.acts[preds == 0].astype(int)).argmax())
            branch.right.action = int(np.bincount(leaf.acts[preds == 1].astype(int)).argmax())

        elif len(unique) == 1:
            leaf.action = unique.item()


def split_on_transition(region2states, tree, thresh_ratio=0.05):
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
        if ratio > thresh_ratio:
            tree.split_leaf(points, y, leaf, thresh=0.9)

    tree.reorder_leaf_labels()
