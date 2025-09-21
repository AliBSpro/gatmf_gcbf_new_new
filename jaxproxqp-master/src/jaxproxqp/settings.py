from __future__ import annotations
import jax.numpy as jnp
from attrs import define
from typing import Optional

# 兼容性导入 - 如果loguru不可用則使用标准日誌
try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

from jaxproxqp.utils.jax_types import FloatScalar

_HAS_LOGGED = False


@define
class Settings:
    alpha_bcl: float = 0.1
    beta_bcl: float = 0.9

    mu_min_eq: float = 1e-9
    mu_min_in: float = 1e-8
    mu_max_eq_inv: float = 1e9
    mu_max_in_inv: float = 1e8

    mu_update_factor: float = 0.1
    mu_update_inv_factor: float = 10.0

    cold_reset_mu_eq: float = 1.0 / 1.1
    cold_reset_mu_in: float = 1.0 / 1.1
    cold_reset_mu_eq_inv: float = 1.1
    cold_reset_mu_in_inv: float = 1.1

    eps_abs: float = 1.0e-5
    # Smallest possible value for bcl_eta_in
    eps_in_min: float = 1e-9

    pri_res_thresh_abs: float = 1e-5
    dua_res_thresh_abs: float = 1e-5
    dua_gap_thresh_abs: Optional[float] = 1e-5

    # Threshold for early exit from inner loop for a step size that is too small.
    step_size_thresh: float = 1e-8

    max_iter: int = 100
    max_iter_in: int = 15

    # Threshold for early exit from inner loop for an error that is too small.
    err_in_thresh: float = 1e-3

    # We use a different threshold for the inner loop when solving the linear system iteratively.
    err_in_thresh_iterative: float = 1e-3

    # Iterative refinement
    nb_iterative_refinement: int = 0

    # Threshold for early exit from iterative refinement for an error that is too small.
    eps_refact: float = 1e-8

    # Threshold for early exit from iterative refinement for an error that is too small.
    eps_refact_in: float = 1e-8

    # Threshold for early exit from iterative refinement for an error that is too small.
    eps_refact_dua: float = 1e-8

    # Threshold for early exit from iterative refinement for an error that is too small.
    eps_refact_pri: float = 1e-8

    # Preconditioning
    preconditioner_max_iter: int = 10
    preconditioner_eps: float = 1e-3

    # Verbose
    verbose: bool = False
    print_interval: int = 1

    # Compute timings
    compute_timings: bool = False

    # Check duality gap
    check_duality_gap: bool = True

    # Statistics
    statistics: bool = False

    # Warm start
    initial_guess: bool = True

    # Scaling
    eps_primal_inf: float = 1e-4
    eps_dual_inf: float = 1e-4

    # Factorization frequency
    refactorize_frequency: int = 0

    # Regularization
    regularization_eps: float = 1e-8

    # Compute reduced costs
    compute_reduced_costs: bool = False

    # Check optimality
    check_optimality: bool = True

    # Proximal penalty parameter
    rho: float = 1e-6

    # Scaling
    scaling: bool = True

    # Scaling max iter
    scaling_max_iter: int = 10

    # Scaling eps
    scaling_eps: float = 1e-3

    # Scaling regularization
    scaling_regularization: float = 1e-8

    # Scaling alpha
    scaling_alpha: float = 0.5

    # Scaling beta
    scaling_beta: float = 0.9

    # Scaling gamma
    scaling_gamma: float = 1.0

    # Scaling delta
    scaling_delta: float = 1.0

    # Scaling epsilon
    scaling_epsilon: float = 1e-8

    def __post_init__(self):
        global _HAS_LOGGED
        if self.verbose and not _HAS_LOGGED:
            logger.info(f"JaxProxQP settings: {self}")
            _HAS_LOGGED = True

    def as_dict(self):
        """Convert settings to dictionary."""
        return {
            'alpha_bcl': self.alpha_bcl,
            'beta_bcl': self.beta_bcl,
            'mu_min_eq': self.mu_min_eq,
            'mu_min_in': self.mu_min_in,
            'mu_max_eq_inv': self.mu_max_eq_inv,
            'mu_max_in_inv': self.mu_max_in_inv,
            'mu_update_factor': self.mu_update_factor,
            'mu_update_inv_factor': self.mu_update_inv_factor,
            'cold_reset_mu_eq': self.cold_reset_mu_eq,
            'cold_reset_mu_in': self.cold_reset_mu_in,
            'cold_reset_mu_eq_inv': self.cold_reset_mu_eq_inv,
            'cold_reset_mu_in_inv': self.cold_reset_mu_in_inv,
            'eps_abs': self.eps_abs,
            'eps_in_min': self.eps_in_min,
            'pri_res_thresh_abs': self.pri_res_thresh_abs,
            'dua_res_thresh_abs': self.dua_res_thresh_abs,
            'dua_gap_thresh_abs': self.dua_gap_thresh_abs,
            'step_size_thresh': self.step_size_thresh,
            'max_iter': self.max_iter,
            'max_iter_in': self.max_iter_in,
            'err_in_thresh': self.err_in_thresh,
            'err_in_thresh_iterative': self.err_in_thresh_iterative,
            'nb_iterative_refinement': self.nb_iterative_refinement,
            'eps_refact': self.eps_refact,
            'eps_refact_in': self.eps_refact_in,
            'eps_refact_dua': self.eps_refact_dua,
            'eps_refact_pri': self.eps_refact_pri,
            'preconditioner_max_iter': self.preconditioner_max_iter,
            'preconditioner_eps': self.preconditioner_eps,
            'verbose': self.verbose,
            'print_interval': self.print_interval,
            'compute_timings': self.compute_timings,
            'check_duality_gap': self.check_duality_gap,
            'statistics': self.statistics,
            'initial_guess': self.initial_guess,
            'eps_primal_inf': self.eps_primal_inf,
            'eps_dual_inf': self.eps_dual_inf,
            'refactorize_frequency': self.refactorize_frequency,
            'regularization_eps': self.regularization_eps,
            'compute_reduced_costs': self.compute_reduced_costs,
            'check_optimality': self.check_optimality,
            'rho': self.rho,
            'scaling': self.scaling,
            'scaling_max_iter': self.scaling_max_iter,
            'scaling_eps': self.scaling_eps,
            'scaling_regularization': self.scaling_regularization,
            'scaling_alpha': self.scaling_alpha,
            'scaling_beta': self.scaling_beta,
            'scaling_gamma': self.scaling_gamma,
            'scaling_delta': self.scaling_delta,
            'scaling_epsilon': self.scaling_epsilon
        }