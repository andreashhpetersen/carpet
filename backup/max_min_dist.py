import numpy as np
import matplotlib.pyplot as plt


def dist(p1, p2):
    return np.linalg.norm(p1 - p2)


def get_dist_matrix(data):
    n = len(data)
    out = []
    for i in range(n):
        dists = []
        for j in range(n):
            dists.append(dist(data[i], data[j]))
        out.append(dists)

    return np.array(out)


def get_first_point(dists):
    return np.argmax(dists.sum(axis=1))



def mdp_score(sol_idxs, d):
    return d[sol_idxs,:].T[sol_idxs,:].T.sum() / 2


def main(data, k):
    dist_mtx = get_dist_matrix(data)

    p1 = get_first_point(dist_mtx)
    p2 = dist_mtx[p1].argmax()

    L = { i for i in range(dist_mtx.shape[0]) }
    S = {p1, p2}
    while len(S) < k:
        rest_L = tuple(L.symmetric_difference(S))
        rest_dists = dist_mtx[rest_L,:].T[tuple(S),:].T
        S.add(rest_L[rest_dists.min(axis=1).argmax()])

    S = list(S)
    return S


if __name__=='__main__':
    N, K, d = 2000, 40, 2
    try:
        data = np.load('./data/normal_2000_loc1_scale10.npy')
    except FileNotFoundError:
        data = np.random.normal(1, 10, (N, d))

    res = main(data, K)

    fig, ax = plt.subplots()

    rest = np.delete(data, res, axis=0)
    ax.scatter(rest[:,0], rest[:,1], c='b')

    ax.scatter(data[tuple(res),0], data[tuple(res),1], c='r')

    plt.show()
