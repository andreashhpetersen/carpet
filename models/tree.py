import warnings
import numpy as np
import joblib
import os
from collections import defaultdict
from learning.splitting import poly_log_reg
from utils import normalize_to_prob

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer


class State:
    def __init__(self, point, region=None):
        self.point = point
        self.region = region
        self.next_states = []
        self.next_regions = []

    def make_transition_probability_vector(self, n_regions):
        """
        Make a probability vector of this state to every other region
        """
        vector = np.zeros((n_regions,))
        for region in self.next_regions:
            vector[region] += 1

        self.trans_prob = normalize_to_prob(vector)


class TreeObserver:
    def __init__(self, n_dims, n_acts, bounds, initial_preds=None, resolution=400):
        """
        Args:
            n_dims (int): number of dimensions in state space
            n_acts (int): number of actions in action space
            bounds (np.array of shape (n_dims, 2)): bounds for each dimension of
                state space, where bounds[i] = (lower_bound, upper_bound) for
                dimension i
            initial_preds (list of tuples): list of initial predicates to split on,
                where each predicate is a tuple (var_idx, c) representing an axis
        """
        self.root = None
        self.n_leaves = 0
        self.n_dims = n_dims
        self.n_acts = n_acts
        self.bounds = bounds
        self.resolution = resolution
        self.T = None

        if initial_preds is not None:
            self.initialize_axis_branches(initial_preds)

    @property
    def leaf_dict(self):
        """Return a dictionary mapping labels to leaves"""
        return { l.label: l for l in self.leaves() }

    def leaves(self):
        leaves = []
        queue = [self.root]
        while len(queue) > 0:
            node = queue.pop()

            if node.is_leaf:
                leaves.append(node)
            else:
                queue.append(node.right)
                queue.append(node.left)

        return leaves

    def split_leaf(self, X, y, leaf, thresh=0.95, pipe=None):
        """
        Finds a polynomial decision boundary for X and y and creates a new
        PolyBranch to replace `leaf`.

        If `pipe` is provided it is used directly, skipping the poly_log_reg fit.
        This allows callers that have already fitted and screened a pipeline (e.g.
        split_on_transition_tv) to avoid fitting it a second time.
        """

        if leaf.terminal:
            warnings.warn(f"Trying to split terminal leaf {leaf.label}, skipping.")
            return None

        if pipe is None:
            pipe, _ = poly_log_reg(X, y, plot=False, max_iter=1000, thresh=thresh)

        branch = PolyBranch(pipe)
        branch.parent = leaf.parent
        branch.left.label = leaf.label
        branch.left.action = leaf.action
        branch.right.label = self.n_leaves
        branch.right.action = leaf.action
        self.n_leaves += 1

        # set new branch as either left or right child
        if leaf.parent.left == leaf:
            leaf.parent.left = branch
        else:
            leaf.parent.right = branch

        return branch

    def split_for_action(self, states, mask, thresh=0.95):
        """
        Split regions if their states disagree on what action to perform next
        (ie. if some states in R_1 goes to R_2 and some to R_3 and R_2.action
        is different from R_3.action, then we split R_1)
        """
        P = []
        leaves = self.leaf_dict
        # labels = self.get_labels(states)
        labels = self.get_labels(states)
        groups = defaultdict(list)

        # map state idx to action in next state
        next_acts = []

        label_offset = 0
        for i in range(len(mask)-1):
            if not (mask[i] and mask[i+1]):
                next_acts.append(-1)
                continue

            groups[labels[i]].append(i)
            next_acts.append(leaves[labels[i+1]].action)

        next_acts = np.array(next_acts)

        for label, state_idxs in groups.items():
            leaf = leaves[label]
            if leaf.terminal:
                continue

            if len(np.unique(next_acts[state_idxs])) > 1:
                X = states[state_idxs]
                y = np.zeros((len(X),))
                y[next_acts[state_idxs] == 1] = 1
                branch = self.split_leaf(X, y, leaf, thresh=thresh)

        self.reorder_leaf_labels()
        self.set_transition_scores(states, mask)

    def mark_terminal_states(self, obs, mask):
        terminals = obs[np.where(mask[:,:-1] & ~mask[:,1:])]
        labels = self.get_labels(terminals)
        leaves = self.leaf_dict
        for label in labels:
            leaves[label].terminal = True

    def get(self, x):
        """
        Returns:
            leaf (PolyLeaf)
        """
        if x.ndim == 1:
            x = x.reshape(1,-1)
        return self.root.get(x)

    def get_labels(self, x):
        idxs = np.arange(len(x))
        labels = np.zeros((len(x),), dtype=np.int32) - 1
        self.root.get_labels(x, idxs, labels)
        return labels

    def reorder_leaf_labels(self):
        stack = [self.root]
        current_label = 0
        while len(stack) > 0:
            node = stack.pop()
            if node.is_leaf:
                node.label = current_label
                current_label += 1
            else:
                stack.append(node.right)
                stack.append(node.left)

    def set_transition_scores(self, states, mask, n_step=1):
        """
        Compute the n_step-step transition matrix and store it as tree.T.

        tree.T has shape (n_leaves,) * (n_step + 1), where
        tree.T[r_0, r_1, ..., r_{n_step}] is the probability of visiting
        region r_{n_step} given that the trajectory passed through
        r_0, r_1, ..., r_{n_step-1}. Normalized along the last axis.
        """
        if len(states.shape) == 3:
            states = states.reshape(-1, states.shape[-1])
            mask = mask.reshape(-1)

        leaves = self.leaf_dict
        labels = self.get_labels(states)
        T = np.zeros((self.n_leaves,) * (n_step + 1))

        for i in range(len(mask) - n_step):
            if not all(mask[i:i + n_step + 1]):
                continue
            idx = tuple(labels[i:i + n_step + 1])
            if leaves[idx[0]].terminal:
                continue
            T[idx] += 1

        self.T = normalize_to_prob(T, axis=-1)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Tree saved to {path}")

    @staticmethod
    def load(path):
        tree = joblib.load(path)
        print(f"Tree loaded from {path}")
        return tree

    def initialize_axis_branches(self, predicates):
        for var_idx, c in predicates:
            self._add_axis(var_idx, c)

        # add labels 0,1,2... from 'left' to 'right'
        label = 0
        queue = [self.root]
        while len(queue) > 0:
            node = queue.pop()

            if node.is_leaf:
                node.label = label
                label += 1
            else:
                queue.append(node.right)
                queue.append(node.left)

    def _add_axis(self, var_idx, c):
        if self.root is None:
            self.root = AxisBranch(var_idx, c)
            self.n_leaves = 2
        else:
            queue = [self.root]
            while len(queue) > 0:
                node = queue.pop()

                if node.left.is_leaf:
                    node.left = AxisBranch(var_idx, c)
                    node.left.parent = node
                    self.n_leaves += 1
                else:
                    queue.append(node.left)

                if node.right.is_leaf:
                    node.right = AxisBranch(var_idx, c)
                    node.right.parent = node
                    self.n_leaves += 1
                else:
                    queue.append(node.right)


class BranchBase:
    def __init__(self, left=None, right=None):
        self.parent = None

        if left is None:
            left = PolyLeaf(parent=self)

        if right is None:
            right = PolyLeaf(parent=self)

        self.left = left
        self.right = right
        self.is_leaf = False
        self.is_axis_branch = False
        self.is_poly_branch = False

    def get(self, x):
        prediction = self.predict(x)
        if not prediction:
            return self.left if self.left.is_leaf else self.left.get(x)
        else:
            return self.right if self.right.is_leaf else self.right.get(x)

    def get_labels(self, x, idxs, labels):
        preds = self.predict(x)
        mask = preds[idxs].astype(np.bool)
        if self.left.is_leaf:
            labels[idxs[~mask]] = self.left.label

        else:
            self.left.get_labels(x, idxs[~mask], labels)

        if self.right.is_leaf:
            labels[idxs[mask]] = self.right.label
        else:
            self.right.get_labels(x, idxs[mask], labels)


class AxisBranch(BranchBase):
    def __init__(self, var_idx, c, left=None, right=None):
        super().__init__(left=left, right=right)
        self.var_idx = var_idx
        self.c = c
        self.is_axis_branch = True

    def __repr__(self):
        return f'AxisBranch<x[{self.var_idx}] < {self.c}>'

    def predict(self, x):
        return x[:,self.var_idx] < self.c

    def put(self, x, acts=None):
        prediction = self.predict(x)
        if acts is not None:
            self.left.put(x[~prediction], acts=acts[~prediction])
            self.right.put(x[prediction], acts=acts[prediction])
        else:
            self.left.put(x[~prediction])
            self.right.put(x[prediction])


class PolyBranch(BranchBase):
    def __init__(self, pipe, left=None, right=None):
        super().__init__(left=left, right=right)
        self.pipe = pipe
        self.is_poly_branch = True

    def predict(self, x, astype=bool):
        return self.pipe.predict(x).astype(astype)

    def put(self, x, acts=None):
        prediction = self.predict(x)
        self.left.put(x[~prediction], acts=None if acts is None else acts[~prediction])
        self.right.put(x[prediction], acts=None if acts is None else acts[prediction])


class PolyLeaf:
    def __init__(self, parent=None):
        self.states = []
        self.acts = None
        self.label = None
        self.parent = parent
        self.terminal = False
        self.action = 0  # this is a bit of a hack, but we need something for the terminal leaves

    def __repr__(self):
        return f'Leaf<{self.label}>'

    @property
    def is_leaf(self):
        return True

    def put(self, states, acts=None):
        self.states = states
        self.acts = acts
