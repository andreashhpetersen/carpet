import os
import re
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle


def scatter_plot(data, labels=None):
    if labels is None:
        plt.scatter(data[:,0], data[:,1])
    else:
        for label in np.unique(labels):
            plt.scatter(data[:,0][label], data[:,1][label])

    plt.show()


def plot_leaf(leaf, states=None, xlim=None, ylim=None, resolution=400):
    cmap = plt.get_cmap('tab20')
    n = 1 if states is None else 1 + len(states)
    colors = [cmap(i / n) for i in range(n)]
    mesh = get_leaf_mesh(leaf, resolution=resolution)
    plt.scatter(*mesh.T, color=colors[0])

    if states is not None:
        for i in range(n):
            plt.scatter(*states[i-1].T, color=colors[i], marker='x')

    # plt.legend([leaf.label for leaf in leaves])
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)

    plt.show()
    plt.close()


def plot_runs(env, model, n_runs=1):
    obs, acts, _, _ = generate_agent_data(model, env, n_runs=n_runs)

    for i in range(n_runs):
        plt.scatter(obs[i,:,0], obs[i,:,1], c=acts[i], cmap='viridis', s=5)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Agent Trajectories Colored by Action')
    plt.colorbar(label='Action')
    plt.show()


def plot_log_reg(X, y, clf=None):

    # Plot
    plt.figure(figsize=(8, 6))

    if clf is not None:
        # Create grid covering feature space
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                             np.linspace(y_min, y_max, 500))
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Predict probabilities or classes on the grid
        probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)   # probability for class 1
        preds = clf.predict(grid).reshape(xx.shape)              # class labels

        # decision boundary by probability contour at 0.5
        contour = plt.contour(xx, yy, probs, levels=[0.5], linewidths=2, colors='k')

        # filled regions for predicted classes (optional)
        plt.contourf(xx, yy, preds, alpha=0.2, cmap=plt.cm.coolwarm)

    # scatter original points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.xlabel('feature 1 (scaled)')
    plt.ylabel('feature 2 (scaled)')
    plt.title('Logistic Regression decision boundary')
    plt.show()


def get_leaf_mesh(leaf, resolution=400):
    node = leaf.parent
    child = leaf

    bounds = []
    pipes = []
    while node is not None:  # condition true when we reach root
        if node.is_axis_branch:
            bounds.append((node.var_idx, node.c, child == node.right))

        if node.is_poly_branch:
            pipes.append((node.pipe, child == node.right))

        child = node
        node = node.parent

    x_lim, y_lim = (0,15), (-15,15)

    # build grid points
    xmin, xmax = x_lim
    ymin, ymax = y_lim
    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    xv, yv = np.meshgrid(xs, ys)
    pts = np.vstack([xv.ravel(), yv.ravel()]).T  # shape (N,2)
    idxs = np.arange(len(pts))

    for var_idx, c, right in bounds[::-1]:
        mask = (pts[idxs, var_idx] < c) == right
        idxs = idxs[mask]

    for pipe, right in pipes:
        mask = pipe.predict(pts[idxs]) == right
        idxs = idxs[mask]

    return pts[idxs]


def plot_tree_partition(
    tree, draw_boundaries=True,
    cmap='tab20', alpha=0.6, contour_resolution=200,
    points=None, acts=None, mask=None, title=None,
    save_dir='notes/imgs', point_size=3, save_point_size=1):
    """
    Plot 2D partition for TreeObserver (HERE BE DRAGONS). Only supports 2D
    state spaces. Can optionally overlay split boundaries and scatter points on
    top.

    Args:
    - tree: TreeObserver instance with 2D state space
    - draw_boundaries: whether to overlay split boundaries
    - cmap: colormap for leaf regions
    - alpha: transparency for leaf region colors
    - contour_resolution: resolution for plotting poly branch boundaries
    - points: optional (N,2) array of points to scatter on top
    - acts: optional (N,) array of actions corresponding to points for coloring
    - mask: optional boolean array of shape (N,) to select subset of points to plot
    - title: optional title for the plot
    - save_dir: directory to save the figure in. The filename is derived from
      `title` (lowercased, whitespace collapsed to underscores). Pass None to
      disable saving.
    - point_size: marker size for scatter points (default 3, smaller than
      matplotlib's default of 36 so the region colors remain visible)
    """
    print("lets plot!")

    if tree.n_dims != 2:
        raise ValueError("Can only plot partitions for 2D state spaces")

    resolution = tree.resolution

    # build grid points
    xmin, xmax = tree.bounds[0]
    ymin, ymax = tree.bounds[1]
    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    xv, yv = np.meshgrid(xs, ys)
    pts = np.vstack([xv.ravel(), yv.ravel()]).T  # shape (N,2)

    # array to hold leaf labels for each point
    labels = np.empty(len(pts), dtype=int)
    label2leaf = defaultdict(int)

    def classify_subset(node, idxs):
        """Recursively assign labels for pts[idxs]."""
        if idxs.size == 0:
            return

        if node.is_leaf:
            labels[idxs] = node.label
            label2leaf[node.label] = node
            return

        # Axis split
        if node.is_axis_branch:
            mask_left = pts[idxs, node.var_idx] >= node.c

        # PolyBranch split
        if node.is_poly_branch:
            # predict classification for points given by idxs
            preds = node.predict(pts[idxs])

            # assume binary: min->left, other->right
            uniq = np.unique(preds)
            left_label = uniq.min()

            # find idxs for left and right path
            mask_left = preds == left_label

        left_idxs = idxs[mask_left]
        right_idxs = idxs[~mask_left]

        # we assume left and right are always defined
        classify_subset(node.left, left_idxs)
        classify_subset(node.right, right_idxs)

    classify_subset(tree.root, np.arange(len(pts)))

    # reshape label grid
    lab_grid = labels.reshape((resolution, resolution))

    # color mapping
    unique_labels = np.unique(labels)
    cmap_obj = plt.get_cmap(cmap, len(unique_labels))
    label_to_color = {lab: cmap_obj(i) for i, lab in enumerate(unique_labels)}
    color_grid = np.zeros((resolution, resolution, 4))
    for lab in unique_labels:
        color_grid[lab_grid == lab] = label_to_color[lab]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(color_grid, origin='lower', extent=(xmin, xmax, ymin, ymax), aspect='auto', alpha=alpha)

    # overlay split visualizations
    def draw_splits(node, region):
        if node is None or node.is_leaf:
            return

        xmin_r, xmax_r, ymin_r, ymax_r = region

        if node.is_axis_branch:
            if node.var_idx == 0:
                ax.plot([node.c, node.c], [ymin_r, ymax_r], color='k', linestyle='--', linewidth=1)
                left_region = (max(node.c, xmin_r), xmax_r, ymin_r, ymax_r)
                right_region = (xmin_r, min(node.c, xmax_r), ymin_r, ymax_r)
            else:
                ax.plot([xmin_r, xmax_r], [node.c, node.c], color='k', linestyle='--', linewidth=1)
                left_region = (xmin_r, xmax_r, max(node.c, ymin_r), ymax_r)
                right_region = (xmin_r, xmax_r, ymin_r, min(node.c, ymax_r))

            draw_splits(node.left, left_region)
            draw_splits(node.right, right_region)
            return

        if node.is_poly_branch:
            # sample region and plot contour of classifier prediction
            rxs = np.linspace(xmin_r, xmax_r, contour_resolution)
            rys = np.linspace(ymin_r, ymax_r, contour_resolution)
            rvx, rvy = np.meshgrid(rxs, rys)
            rpts = np.vstack([rvx.ravel(), rvy.ravel()]).T
            pred = node.predict(rpts).reshape(rvx.shape)
            # try to draw contours between unique labels
            levels = np.unique(pred).astype(float)
            if levels.size >= 2:
                # draw contour lines between classes (this will show boundaries)
                ax.contour(rvx, rvy, pred, levels=(levels[:-1] + levels[1:]) / 2.0,
                           colors='k', linestyles='--', linewidths=1)
            draw_splits(node.left, region)
            draw_splits(node.right, region)
            return

    if draw_boundaries:
        draw_splits(tree.root, (xmin, xmax, ymin, ymax))

    scatter_artists = []
    if points is not None:
        masked_points = points[mask]
        if acts is not None:
            masked_acts = acts[mask]
            unique_acts = np.unique(masked_acts)
            for act in unique_acts:
                sc = ax.scatter(
                    masked_points[masked_acts == act][:,0],
                    masked_points[masked_acts == act][:,1],
                    s=point_size,
                )
                scatter_artists.append(sc)
        else:
            sc = ax.scatter(masked_points[:,0], masked_points[:,1], s=point_size)
            scatter_artists.append(sc)

    # legend
    print_labels = []
    for l in unique_labels:
        leaf = label2leaf[l]
        if tree.T is not None and tree.T.ndim == 2:
            next_reg = np.argmax(tree.T[leaf.label])
            print_labels.append(f'{leaf.label} (-> {next_reg})')
        else:
            print_labels.append(f'{leaf.label}')

    patches = [Rectangle((0,0),1,1, color=label_to_color[lab]) for lab in unique_labels]
    ax.legend(patches, print_labels, title="leaf labels",
              bbox_to_anchor=(1.05, 1), loc='upper left')

    if title:
        ax.set_title(title, fontsize=14)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x0", fontsize=12)
    ax.set_ylabel("x1", fontsize=12)
    ax.tick_params(labelsize=11)
    plt.tight_layout()

    if save_dir is not None:
        for sc in scatter_artists:
            sc.set_sizes(np.full(len(sc.get_offsets()), save_point_size))
        slug = re.sub(r'\s+', '_', title.lower()) if title else 'partition'
        slug = re.sub(r'[^\w_]', '', slug)   # strip any non-word characters
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{slug}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f'Saved plot to {save_path}')
        for sc in scatter_artists:
            sc.set_sizes(np.full(len(sc.get_offsets()), point_size))

    if save_dir is None:
        plt.show()
    plt.close()


def draw_leaves(tree, *leaves, states=None, xlim=None, ylim=None, resolution=None):
    if tree.n_dims != 2:
        raise ValueError("Can only draw leaves for 2D state spaces")

    if xlim is None:
        xlim = (tree.bounds[0,0], tree.bounds[0,1])

    if ylim is None:
        ylim = (tree.bounds[1,0], tree.bounds[1,1])

    if resolution is None:
        resolution = tree.resolution

    cmap = plt.get_cmap('tab20')

    n = len(leaves)

    meshes = [get_leaf_mesh(leaf, xlim=xlim, ylim=ylim, resolution=resolution) for leaf in leaves]
    colors = [cmap(i / (len(leaves) - 1)) for i in range(n)]

    for i in range(n):
        plt.scatter(*meshes[i].T, color=colors[i])

    # states shouldn't be plotted for first leaf
    if states is not None:
        for i in range(n - 1):
            plt.scatter(*states[i].T, color=colors[i+1], marker='x')

    plt.legend([leaf.label for leaf in leaves])

    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)

    plt.show()
    plt.close()
