import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class Sinkhorn(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.

    Arguments:
        - eps (float): regularization coefficient
        - max_iter (int): maximum number of Sinkhorn iterations
        - reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'

    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps, eps_parameter, max_iter, thresh, reduction="none", device="cpu"):
        super(Sinkhorn, self).__init__()
        self.device = device
        self.eps_parameter = eps_parameter

        self.eps = eps
        if self.eps_parameter:
            self.eps = nn.Parameter(torch.tensor(self.eps))

        self.max_iter = max_iter
        self.thresh = thresh
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        cost_normalization = C.max()
        C = (
            C / cost_normalization
        )  # Needs to normalize the matrix to be consistent with reg

        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float, requires_grad=False).fill_(
            1.0 / x_points).squeeze().to(self.device)

        nu = torch.empty(batch_size, y_points, dtype=torch.float, requires_grad=False).fill_(
            1.0 / y_points).squeeze().to(self.device)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = (
                self.eps
                * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1))
                + u
            )
            v = (
                self.eps
                * (
                    torch.log(nu + 1e-8)
                    - torch.logsumexp(self.M(C, u,
                                      v).transpose(-2, -1), dim=-1)
                )
                + v
            )
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < self.thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == "mean":
            cost = cost.mean()
        elif self.reduction == "sum":
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        """Modified cost for logarithmic updates
        $M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$
        """
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1
    
class OpTA(nn.Module):
    def __init__(
            self,
            regularization: float,
            max_iter: int,
            stopping_criterion: float,
            device: str = "cpu"):
        """
        Initializes an OpTA inference module.

        Arguments:
            - regularization (float): regularization coefficient
            - max_iter (int): maximum number of Sinkhorn iterations
            - stopping_criterion (float): threshold for Sinkhorn algorithm
            - device (str): device used for optimal transport calculations

        Returns:
            - tuple(transported support prototypes, unchanged query features)
        """
        super(OpTA, self).__init__()
        self.sinkhorn = Sinkhorn(
            eps=regularization,
            max_iter=max_iter,
            thresh=stopping_criterion,
            eps_parameter=False,
            device=device)

    def forward(self, z_support: torch.Tensor, z_query: torch.Tensor):
        """
        Applies Optimal Transport between support and query features.

        Arguments:
            - z_support (torch.Tensor): support prototypes (or features)
            - z_query (torch.Tensor): query features

        Returns:
            - tuple(transported support prototypes, unchanged query features)
        """
        cost, transport_plan, _ = self.sinkhorn(z_support, z_query)

        z_support_transported = torch.matmul(
            transport_plan / transport_plan.sum(axis=1, keepdims=True), z_query
        )

        return z_support_transported, z_query


def groupedAvg(myArray, N):
    result = np.cumsum(myArray, 0)[N-1::N]/float(N)
    result[1:] = result[1:] - result[:-1]
    return result

def trans_test_gen(cur_sup_f,cur_qry_f, n_shot):
    one_shot_ot_passes = 5
    five_shot_ot_passes = 2
    ratio_OT = 1

    prototypes_before = groupedAvg(cur_sup_f, n_shot) #5,512 ; 1

    # initilize OpTA module
    transportation_module = OpTA(regularization=0.05,
                                    max_iter=1000,
                                    stopping_criterion=1e-4)

    ot_passes = one_shot_ot_passes if n_shot == 1 else five_shot_ot_passes
    prototypes = prototypes_before
    for i in range(ot_passes):
        prototypes, cur_qry_f = transportation_module(
            torch.from_numpy(prototypes), torch.from_numpy(cur_qry_f))

        if i == 0:
            first_pass_prototypes = prototypes

        prototypes = prototypes.detach().cpu().numpy()
        cur_qry_f = cur_qry_f.detach().cpu().numpy()

    if n_shot != 1:
        prototypes = prototypes * ratio_OT + \
            prototypes_before * (1 - ratio_OT)
        
    return prototypes, cur_qry_f