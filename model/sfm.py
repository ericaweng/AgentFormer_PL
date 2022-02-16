import torch

EPS = 1.e-10


def cart2pol(x, y, mask=None):
    r = torch.sqrt((x + EPS)**2 + (y + EPS)**2)
    r = torch.unsqueeze(r, dim=-1)
    theta = torch.atan2(y + EPS, x + EPS).reshape(x.shape)
    theta = torch.unsqueeze(theta, dim=-1)

    if mask is not None:
        r = torch.where(mask.bool(), 0., r.double())
        theta = torch.where(mask.bool(), 0., theta.double())

    return r, theta


def _compute_collision_w(self, v_i, p_ij, p_ij_r, p_ij_theta):
        wd = torch.exp(-0.5 * p_ij_r**2 / self.sigma_w**2)

        v_i_n = v_i / (v_i.norm(dim=-1, keepdim=True) + EPS)
        p_ij_n = p_ij / (p_ij.norm(dim=-1, keepdim=True) + EPS)
        wf = (0.5 * (1 +
                     (v_i_n * p_ij_n).sum(dim=-1, keepdim=True)))**self.beta
        res = wd * wf
        return res

def _compute_collision_dstar(self, v_hat, v_j, p_ij, neighbor_mask=None):
    v_hatj = v_hat - v_j
    v_hatj_r, _ = cart2pol(v_hatj[:, :, 0],
                            v_hatj[:, :, 1],
                            mask=neighbor_mask)
    kq = (p_ij * v_hatj).sum(dim=-1, keepdims=True)
    dstar = p_ij - (kq / (v_hatj_r**2 + 1.e-12)) * v_hatj
    dstar = torch.where(dstar.isnan(), torch.inf, dstar.double())
    return dstar

def _compute_collision_energy(self, dstar):
    return torch.exp(-(dstar**2).sum(dim=-1, keepdims=True) /
                        (2 * (self.sigma_d**2)))

def collision_term(self,
                   v_hat,
                   v_i,
                   v_j,
                   p_ij,
                   p_ij_r,
                   p_ij_theta,
                   neighbor_mask=None):
    w = self._compute_collision_w(v_i, p_ij, p_ij_r, p_ij_theta)
    dstar = self._compute_collision_dstar(v_hat,
                                          v_j,
                                          p_ij,
                                          neighbor_mask=neighbor_mask)
    energy = self._compute_collision_energy(dstar)
    # print(w[:, -1], dstar[:, -1], energy[:, -1])
    col = energy
    if neighbor_mask is not None:
        col = torch.where(neighbor_mask.bool(), 0., col.double())

    return col.sum(dim=1)