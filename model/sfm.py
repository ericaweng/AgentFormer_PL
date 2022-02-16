import torch
from torch.nn import functional as F

EPS = 1.e-10


def pdiff(x):
    return x[:, None, :] - x[None, :, :]


def _compute_collision_w(p_ij, v_ij, params):
    beta = params['beta']
    v_ij_n = v_ij / (v_ij.norm(dim=-1, keepdim=True) + EPS)
    p_ij_n = p_ij / (p_ij.norm(dim=-1, keepdim=True) + EPS)
    wf = (0.5 * (1 - (v_ij_n * p_ij_n).sum(dim=-1))) ** beta
    res = wf
    return res


def collision_term(p_i, v_i, params):
    sigma_d = params['sigma_d']
    ind = torch.triu_indices(p_i.shape[0], p_i.shape[0], offset=1)
    p_ij = pdiff(p_i)[ind[0], ind[1]]
    v_ij = pdiff(v_i)[ind[0], ind[1]]
    # diff = p_ij.norm(dim=-1)[ind[0], ind[1]] - F.pdist(p_i)
    w = _compute_collision_w(p_ij, v_ij, params)
    energy = torch.exp(-0.5 * p_ij.norm(dim=-1)**2 / sigma_d**2)
    col = w * energy
    return col.sum()


def compute_grad_feature(state, params):
    with torch.enable_grad():
        state.requires_grad_(True)
        p_i, v_i = state[..., :2], state[..., 2:]
        col = collision_term(p_i, v_i, params)
        grad = torch.autograd.grad(col, state)[0]
    return grad