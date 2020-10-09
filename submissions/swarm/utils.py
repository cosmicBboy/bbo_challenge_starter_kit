import torch
import numpy as np
import scipy.stats as ss


def order_stats(X):
    _, idx, cnt = np.unique(X, return_inverse=True, return_counts=True)
    obs = np.cumsum(cnt)  # Need to do it this way due to ties
    o_stats = obs[idx]
    return o_stats


def copula_standardize(X):
    X = np.nan_to_num(np.asarray(X))  # Replace inf by something large
    assert X.ndim == 1 and np.all(np.isfinite(X))
    o_stats = order_stats(X)
    quantile = np.true_divide(o_stats, len(X) + 1)
    X_ss = ss.norm.ppf(quantile)
    return X_ss


def exponentiated_log(x: np.ndarray, gamma=0.1):
    """Bounds functions with a range of >= 0 to range of [0, 1].
    This function is used for scorers where larger scores mean worse
    performance. This function returns 1.0 if the score is 0, and returns
    values that asymptotically approach 0 as the score approaches positive
    infinity.
    :param numpy ndarray x: value to transform
    :param float gamma: decay coefficient. Larger gamma means function decays
        more quickly as x gets larger.
    :returns: transformed value.
    :rtype: float
    """
    if (x < 0).all():
        raise ValueError("value %s not a valid input. Must be >= 0")
    if (x == 0).all():
        # since the below function is undefined at x=0, return 1 if x=0.
        return 1.0
    return 1 / (1 + np.power(np.e, np.log(gamma * x)))


def derive_reward(y, best_y=None):
    n_orig = len(y)
    y = np.array(y + [best_y] if best_y is not None else y)
    # yy = exponentiated_log(y) if all(i > 0 for i in y) else y
    yy = copula_standardize([-i for i in y])
    rewards = torch.tensor(yy)
    rewards = (rewards - rewards.median()) / (rewards.std() + 1e-6)
    return rewards[:n_orig]


def compute_grad_norm(agent):
    grad_norm = 0.0
    for name, params in agent.named_parameters():
        if params.grad is not None:
            param_norm = params.grad.data.norm(2)
            grad_norm += param_norm.item() ** 2
    grad_norm = grad_norm ** 0.5
    return grad_norm


def latin_hypercube(n_pts, dim):
    """Basic Latin hypercube implementation with center perturbation."""
    X = np.zeros((n_pts, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    for i in range(dim):  # Shuffle the center locataions for each dimension.
        X[:, i] = centers[np.random.permutation(n_pts)]

    # Add some perturbations within each box
    pert = np.random.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)
    X += pert
    return X


def to_unit_cube(x, lb, ub):
    """Project to [0, 1]^d from hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = (x - lb) / (ub - lb)
    return xx


def from_unit_cube(x, lb, ub):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = x * (ub - lb) + lb
    return xx
