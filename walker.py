import numpy as np
# from numba import njit
rng = np.random.default_rng(0)


def main(w_mat, version, no_samples):

    w_mat += np.ones(w_mat.shape) / (w_mat.shape[0] ** 2)

    if version == 'nonadj':
        mod = 2* int(w_mat.shape[0] * np.log2(w_mat.shape[0]))
        trans = np.array([(i, j) for i in range(w_mat.shape[0]) for j in range(i + 1, w_mat.shape[0])])
        trans_supp = np.array([
            {(idx, (i, j)) for idx, (i, j) in enumerate(trans) if i == h or j == h} for h in range(w_mat.shape[0])
        ])
        supp = np.empty(w_mat.shape, dtype=object)
        for i in range(w_mat.shape[0]):
            for j in range(i + 1, w_mat.shape[0]):
                supp[i, j] = sorted(set.union(trans_supp[i],trans_supp[j]))
    else:
        raise Exception('Version {0} not supported!'.format(version))

    pi = rng.permutation(np.array(range(w_mat.shape[0])))
    p = np.array([w_mat[pi[j], pi[i]] for i, j in trans])
    p_sum = p.sum()

    pis = np.empty((no_samples, *pi.shape), dtype=pi.dtype)
    steps_per_sample = rng.binomial(mod, 0.5, size=no_samples)

    for i, n_steps in enumerate(steps_per_sample):
        print(f'Sample {i + 1}')
        for _ in range(n_steps):
            pi, p, p_sum = walk(pi, p, p_sum, w_mat, trans, supp)
        pis[i] = pi

    return pis

def walk(pi, p, p_sum, w_mat, trans, supp):
    i_star, j_star = trans[rng.choice(range(trans.shape[0]), p=p/p_sum)]
    pi[i_star], pi[j_star] = pi[j_star], pi[i_star]
    for idx, (i, j) in supp[i_star, j_star]:
        val = w_mat[pi[j], pi[i]]
        p_sum += val - p[idx]
        p[idx] = val

    return pi, p, p_sum

# def walk(pi, p, p_sum, w_mat, trans, supp):
#     if rng.uniform() <= 0.5:
#         i_star, j_star = trans[rng.choice(range(trans.shape[0]), p=p/p_sum)]
#         idx_l, i_l, j_l = zip(*((idx, i, j) for idx, (i, j) in supp[i_star, j_star]))
#         pi, p, p_sum = update(i_star, j_star, idx_l, i_l, j_l, pi, p, p_sum, w_mat)
#
#     return pi, p, p_sum
#
#
# @njit
# def update(i_star, j_star, idx_l, i_l, j_l, pi, p, p_sum, w_mat):
#     pi[i_star], pi[j_star] = pi[j_star], pi[i_star]
#     for idx, i, j in zip(idx_l, i_l, j_l):
#         val = w_mat[pi[j], pi[i]]
#         p_sum += val - p[idx]
#         p[idx] = val
#
#     return pi, p, p_sum

