from __future__ import annotations
from typing import NamedTuple, Tuple, Any

import equinox as eqx
import equinox.internal as eqi
import jax.debug as jd
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
import numpy as np
from flax import struct

import jaxproxqp.utils.print_debug as pd
from jaxproxqp.precond.ruiz import Ruiz
from jaxproxqp.qp_problems import QPBox, QPModel
from jaxproxqp.qp_types import EVec, IVec, IXBool, IXVec, WVec, XVec
from jaxproxqp.settings import Settings
from jaxproxqp.utils.jax_math_utils import (
    add_diag_seg,
    default_float,
    get_machine_eps,
    infty_norm,
    neg_part,
    pos_part,
    sqnorm,
)
from jaxproxqp.utils.jax_types import BoolScalar, FloatScalar, IntScalar, assert_x64_enabled, float32_is_default


class Multipliers(NamedTuple):
    mu_eq: FloatScalar = 1e-3
    mu_eq_inv: FloatScalar = 1e3

    mu_in: FloatScalar = 1e-1
    mu_in_inv: FloatScalar = 1e1

    rho: FloatScalar = 1e-6


class Resids(NamedTuple):
    dua_res: XVec
    eq_res: EVec
    in_res_hi: IXVec
    in_res_lo: IXVec


class PriRes(NamedTuple):
    # Primal infeasibility. max( |Ax-b|, |Cx-u|, |Cx-l|, |x-u_box|, |x-l_box| )
    lhs: float
    # |Ax|
    eq_rhs_0: float
    # |Cx|
    in_rhs_0: float
    # |Ax-b|
    eq_lhs: float
    # |Cx - u| + |Cx - l| + |x-u_box|, |x-l_box|
    in_lhs: float


class DuaRes(NamedTuple):
    # norm(dual_residual_scaled)
    lhs: FloatScalar
    # norm( unscaled( Hx ) )
    rhs_0: FloatScalar
    # norm( unscaled( ATy ) )
    rhs_1: FloatScalar
    # norm( unscaled( CTz ) )
    rhs_3: FloatScalar
    # max of a bunch of terms.
    rhs_duality_gap: FloatScalar
    # <x, Hx> + <g, x> + <b, y> + <z, u> + <z, l> + <z, u_box> + <z, l_box>
    duality_gap: FloatScalar

    def __str__(self) -> str:
        return "lhs: {:.4e} rhs 0: {:.4e}, 1: {:.4e}, 3: {:.4e}, rhs_gap: {:.4e}, gap: {:.4e}".format(
            self.lhs, self.rhs_0, self.rhs_1, self.rhs_3, self.rhs_duality_gap, self.duality_gap
        )


@struct.dataclass
class ActiveSetInfo:
    active_set_hi: IXBool
    active_set_lo: IXBool

    n_in: int = struct.field(pytree_node=False)

    @staticmethod
    def create(dim: int, n_in: int):
        active_set_hi = jnp.ones(n_in + dim, dtype=bool)
        active_set_lo = jnp.ones(n_in + dim, dtype=bool)
        return ActiveSetInfo(active_set_hi, active_set_lo, n_in)

    @property
    def active_set_hi_head(self) -> IVec:
        return self.active_set_hi[: self.n_in]

    @property
    def active_set_hi_tail(self) -> IVec:
        return self.active_set_hi[self.n_in :]

    @property
    def active_set_lo_head(self) -> IVec:
        return self.active_set_lo[: self.n_in]

    @property
    def active_set_lo_tail(self) -> IVec:
        return self.active_set_lo[self.n_in :]


class IterInfo(NamedTuple):
    # Total inner loop iterations
    iter_inner: int = 0
    # External loop iterations
    iter_ext: int = 0

    mu_updates: int = 0
    # Rho is never updated!
    rho_updates: int = 0


class PriResWork(NamedTuple):
    eq_res: EVec
    in_res: IXVec
    Cx_unscaled: IVec
    x_unscaled: XVec


class DuaResWork(NamedTuple):
    # scaled( g + Hx + A^T y + C^T z + z )
    dua_res: XVec


class BCLCoeffs(NamedTuple):
    bcl_eta_ext: FloatScalar
    bcl_eta_in: FloatScalar
    bcl_mu_in: FloatScalar
    bcl_mu_eq: FloatScalar
    bcl_mu_in_inv: FloatScalar
    bcl_mu_eq_inv: FloatScalar


class PrimalDualDerivResult(NamedTuple):
    # Second order poly coeff of merit fn.
    a: FloatScalar
    # First order poly coeff of merit fn.
    b: FloatScalar
    # Derivative of merit function.
    grad: FloatScalar


@struct.dataclass
class Solution:
    x: XVec
    y: EVec
    z: IXVec

    n_in: int = struct.field(pytree_node=False)

    @staticmethod
    def zero_init(dim: int, n_eq: int, n_in: int):
        x = jnp.zeros(dim)
        y = jnp.zeros(n_eq)
        z = jnp.zeros(n_in + dim)
        return Solution(x, y, z, n_in)

    @property
    def z_head(self) -> IVec:
        assert self.z.shape == (self.n_in + self.dim,)
        return self.z[: self.n_in]

    @property
    def z_tail(self) -> XVec:
        assert self.z.shape == (self.n_in + self.dim,)
        s = self.n_in
        return self.z[s : s + self.dim]

    @property
    def dim(self):
        return self.x.shape[0]

    @property
    def n_eq(self):
        return self.y.shape[0]

    def replace(self, **updates) -> "Solution":
        ...


class QPSolution(NamedTuple):
    x: XVec
    y: EVec
    z: IXVec

    obj_value: FloatScalar

    pri_res: FloatScalar
    dua_res: FloatScalar
    duality_gap: FloatScalar

    info: IterInfo


class ResWorkStep(NamedTuple):
    Hdx: XVec
    Adx: EVec
    ATdy: XVec


class MatVecs(NamedTuple):
    Hdx: XVec
    Adx: EVec
    ATdy: XVec
    Cdx: IXVec


class PriDuaResidsAndOthers(NamedTuple):
    pri_res: PriRes
    dua_res: DuaRes

    # Work.
    pri_res_work: PriResWork
    dua_res_work: DuaResWork


class LineSearchAlphaCarry(NamedTuple):
    alpha_last_neg: FloatScalar
    last_neg_grad: FloatScalar
    alpha_first_pos: FloatScalar
    first_pos_grad: FloatScalar


class OuterLoopCarry(NamedTuple):
    sol: Solution
    sol_prev: Solution
    mults: Multipliers
    bcl_coeffs: BCLCoeffs
    active_info: ActiveSetInfo
    iter_info: IterInfo
    resids: PriDuaResidsAndOthers


class JaxProxQP:
    Settings = Settings
    QPModel = QPModel

    def __init__(self, qp: QPModel, settings: Settings = None):
        # Place immutables in self.
        self._qp = qp

        if settings is None:
            settings = Settings.default()
        self._settings = settings

        # Make sure the qp dtype matches the default. i.e., if default is float32, make sure qp has no float64.
        if float32_is_default():
            leaves = jtu.tree_leaves(qp)
            for leaf in leaves:
                assert leaf.dtype != jnp.float64, "QPModel has float64, but default is float32!"

        # Ruiz equilibration.
        ruiz = Ruiz.create(qp.dim, qp.n_eq, qp.n_in)
        i_scaled = np.zeros(0)
        qp_box = QPBox(qp.H, qp.g, qp.A, qp.C, qp.b, qp.u, qp.l, i_scaled, qp.u_box, qp.l_box)
        self._ruiz, self._scaled = ruiz.scale_qp(qp_box, max_iter=self._settings.preconditioner_max_iter)

        # Make sure that we have 64 bit accessible if we need it.
        if self._settings.use_f64_refine:
            assert_x64_enabled()

    @property
    def dim(self):
        return self._qp.dim

    @property
    def n_eq(self):
        return self._qp.n_eq

    @property
    def n_in(self):
        return self._qp.n_in

    @property
    def H_scaled(self):
        return self._scaled.H

    @property
    def g_scaled(self):
        return self._scaled.g

    @property
    def A_scaled(self):
        return self._scaled.A

    @property
    def C_scaled(self):
        return self._scaled.C

    @property
    def b_scaled(self):
        return self._scaled.b

    @property
    def u_scaled(self):
        return self._scaled.u

    @property
    def l_scaled(self):
        return self._scaled.l

    @property
    def ubox_scaled(self):
        return self._scaled.u_box

    @property
    def lbox_scaled(self):
        return self._scaled.l_box

    @property
    def i_scaled(self):
        return self._scaled.I

    @property
    def _alpha_gpdal(self):
        return self._settings.alpha_gpdal

    def split_wvec(self, w_vec: WVec):
        dx = w_vec[: self.dim]

        s = self.dim
        dy = w_vec[s : s + self.n_eq]

        s += self.n_eq
        dz_head = w_vec[s : s + self.n_in]

        s += self.n_in
        dz_tail = w_vec[s : s + self.dim]

        dz = w_vec[-(self.n_in + self.dim) :]

        return dx, dy, dz, dz_head, dz_tail

    def solve(self) -> QPSolution:
        def cond(carry: OuterLoopCarry) -> BoolScalar:
            pri_res, dua_res = carry.resids.pri_res, carry.resids.dua_res

            rhs_pri = self._settings.pri_res_thresh_abs
            is_pri_feas = pri_res.lhs <= rhs_pri

            rhs_dua = self._settings.dua_res_thresh_abs
            is_dua_feas = dua_res.lhs <= rhs_dua
            is_solved = is_pri_feas & is_dua_feas

            if self._settings.dua_gap_thresh_abs is not None:
                small_dua_gap = dua_res.duality_gap <= self._settings.dua_gap_thresh_abs
                is_solved = is_solved & small_dua_gap

            # Continue if not solved.
            return ~is_solved

        def body(carry: OuterLoopCarry):
            sol: Solution
            mults: Multipliers
            bcl_coeffs: BCLCoeffs
            active_info: ActiveSetInfo
            iter_info: IterInfo
            resids: PriDuaResidsAndOthers
            # TODO: Is sol_prev actually used before we udpate it?
            sol, sol_prev, mults, bcl_coeffs, active_info, iter_info, resids = carry

            ######################
            # Logging.
            ######################
            if self._settings.verbose:
                self._print_outer_loop(carry)

            #########################################################################################
            iter_info = iter_info._replace(iter_ext=iter_info.iter_ext + 1)
            sol_prev = sol

            # Scale the computations from global_primal_residual.
            pri_res_work, dua_res_work = resids.pri_res_work, resids.dua_res_work
            scaled_Cx = self._ruiz.scale_pri_res_in(pri_res_work.Cx_unscaled)
            scaled_x = self._ruiz.scale_box_pri_res_in(pri_res_work.x_unscaled)
            # scaled_x = sol.x
            pri_res_in_sc_up = jnp.concatenate([scaled_Cx, scaled_x], axis=0)

            # Now contains [scaled(Cx + mu_in * z_prev); scaled(x + mu_in * z_prev)]
            pri_res_in_sc_up = pri_res_in_sc_up + sol_prev.z * mults.mu_in
            # TODO: Why?
            pri_res_in_sc_up = pri_res_in_sc_up + (self._settings.alpha_gpdal - 1.0) * mults.mu_in * sol.z
            si = pri_res_in_sc_up

            # Now contains scaled( Cx - u + mu_in * z_prev )
            pri_res_in_sc_up = pri_res_in_sc_up.at[: self.n_in].add(-self.u_scaled)
            # Now contains scaled( Cx - l + mu_in * z_prev )
            si = si.at[: self.n_in].add(-self.l_scaled)

            # if box_constraints
            # Now contains scaled( x - u_box + mu_in * z_prev )
            pri_res_in_sc_up = pri_res_in_sc_up.at[-self.dim :].add(-self.ubox_scaled)
            # Now contains scaled( x - l_box + mu_in * z_prev )
            si = si.at[-self.dim :].add(-self.lbox_scaled)

            # Set pri_res_in_sc_up and si.
            res = Resids(dua_res_work.dua_res, pri_res_work.eq_res, pri_res_in_sc_up, si)
            # jd.print("    dua   : {}", res.dua_res, ordered=True)
            # jd.print("    eq    : {}", res.eq_res, ordered=True)
            # jd.print("    in_hi : {}", res.in_res_hi, ordered=True)
            # jd.print("    in_lo : {}", res.in_res_lo, ordered=True)
            # jd.breakpoint(ordered=True)
            inner_iters, res, sol, active_info_new = self.primal_dual_newton_semi_smooth(
                res, sol, sol_prev, mults, bcl_coeffs.bcl_eta_in
            )

            iter_info = iter_info._replace(iter_inner=iter_info.iter_inner + inner_iters)
            # jd.print("    x: {}", sol.x, ordered=True)
            # jd.print("    y: {}", sol.y, ordered=True)
            # jd.print("    z: {}", sol.z, ordered=True)
            # jd.breakpoint(ordered=True)

            # Check primal resids again.
            pri_res_new, pri_res_work_new = self.global_primal_residual(sol)

            # BCL update. We may rollback the y and z of sol if it was a bad update.
            sol, bcl_coeffs_new = self.bcl_update(pri_res_new.lhs, sol, sol_prev, bcl_coeffs, mults)


            # Compute dual residuals. We don't need to check primal residuals, since x didn't change.
            dua_res_new, dua_res_work_new = self.global_dual_residual(sol, active_info_new)

            # TODO: Cold Restart. Use multipliers BEFORE the update.
            pri_res_higher = pri_res_new.lhs >= resids.pri_res.lhs
            dua_res_higher = dua_res_new.lhs >= resids.dua_res.lhs
            mu_is_small = mults.mu_in <= 1e-5
            should_cold_restart = pri_res_higher & dua_res_higher & mu_is_small
            # mults = eqx.error_if(mults, should_cold_restart, "Need to cold restart!")
            settings = self._settings
            mu_eq_new = jnp.where(should_cold_restart, settings.cold_reset_mu_eq, bcl_coeffs_new.bcl_mu_eq)
            mu_in_new = jnp.where(should_cold_restart, settings.cold_reset_mu_in, bcl_coeffs_new.bcl_mu_in)
            mu_eq_inv_new = jnp.where(should_cold_restart, settings.cold_reset_mu_eq_inv, bcl_coeffs_new.bcl_mu_eq_inv)
            mu_in_inv_new = jnp.where(should_cold_restart, settings.cold_reset_mu_in_inv, bcl_coeffs_new.bcl_mu_in_inv)
            if self._settings.verbose:
                lax.cond(should_cold_restart, lambda: jd.print("\033[1;31m<< Cold Restart! >>\033[0m"), lambda: None)

            # If mu has been updated, then update it?
            has_mu_update = (mults.mu_in != bcl_coeffs_new.bcl_mu_in) | (mults.mu_eq != bcl_coeffs_new.bcl_mu_eq)
            iter_info_new = iter_info._replace(mu_updates=iter_info.mu_updates + has_mu_update)
            mults_new = mults._replace(
                mu_eq=mu_eq_new,
                mu_in=mu_in_new,
                mu_eq_inv=mu_eq_inv_new,
                mu_in_inv=mu_in_inv_new,
            )

            # End of outer loop body, repeat.
            resids = PriDuaResidsAndOthers(pri_res_new, dua_res_new, pri_res_work_new, dua_res_work_new)
            return OuterLoopCarry(sol, sol_prev, mults_new, bcl_coeffs_new, active_info_new, iter_info_new, resids)

        # Zero init.
        # TODO: Do the equality constrained only solve thing.
        sol_init = Solution.zero_init(self.dim, self.n_eq, self.n_in)
        active_info_init = ActiveSetInfo.create(self.dim, self.n_in)
        mults_init = Multipliers()
        bcl_coeffs_init = BCLCoeffs(
            self._settings.bcl_eta_ext_init,
            1.0,
            mults_init.mu_in,
            mults_init.mu_eq,
            mults_init.mu_in_inv,
            mults_init.mu_eq_inv,
        )
        iter_info_init = IterInfo()

        resids_init = self.compute_pri_dua_resids(sol_init, active_info_init)
        carry_init = OuterLoopCarry(
            sol_init, sol_init, mults_init, bcl_coeffs_init, active_info_init, iter_info_init, resids_init
        )
        carry_out = eqi.while_loop(cond, body, carry_init, max_steps=self._settings.max_iter, kind="lax")

        # To mimic the behavior of the original code, print one more time after exit.
        if self._settings.verbose:
            self._print_outer_loop(carry_out)

        # Unscale everything at the end.
        x = self._ruiz.unscale_pri(carry_out.sol.x)
        y = self._ruiz.unscale_dua_eq(carry_out.sol.y)
        z = self._ruiz.unscale_dua_ix_in(carry_out.sol.z)

        # Compute objective.
        obj = 0.5 * jnp.dot(x, self._qp.H @ x) + jnp.dot(self._qp.g, x)
        pri_res_out, dua_res_out = carry_out.resids.pri_res, carry_out.resids.dua_res

        qp_solution = QPSolution(
            x, y, z, obj, pri_res_out.lhs, dua_res_out.lhs, dua_res_out.duality_gap, carry_out.iter_info
        )
        if self._settings.verbose:
            self.print_summary(qp_solution)
        return qp_solution

    def _print_outer_loop(self, carry: OuterLoopCarry):
        pd.print("", ordered=True)
        pd.print("\033[1;32m[outer iteration {}]\033[0m", carry.iter_info.iter_ext + 1, ordered=True)
        pd.print(
            "| pri res={:.4e} | dua res={:.4e} | duality gap={:.4e} | mu_in={:.4e} | rho={:.4e} |",
            carry.resids.pri_res.lhs,
            carry.resids.dua_res.lhs,
            carry.resids.dua_res.duality_gap,
            carry.mults.mu_in,
            carry.mults.rho,
            ordered=True,
        )

    def _print_inner_loop(
        self, ii: IntScalar, err_in: FloatScalar, alpha: FloatScalar, step_size: FloatScalar, eps_int: FloatScalar
    ):
        jd.print("\033[1;34m[inner iteration {}]\033[0m", ii + 1, ordered=True)
        pd.print(
            "| inner resid={:8.2e} | alpha={:8.2e} | step={:8.2e} | eps={:8.2e} |",
            err_in,
            alpha,
            step_size,
            eps_int,
            ordered=True,
        )

    def print_summary(self, sol: QPSolution):
        pd.print("-------------------SOLVER STATISTICS-------------------")
        pd.print("outer iter:   {}", sol.info.iter_ext)
        pd.print("total iter:   {}", sol.info.iter_inner)
        pd.print("mu updates:   {}", sol.info.mu_updates)
        pd.print("rho updates:  {}", sol.info.rho_updates)
        pd.print("objective:    {}", sol.obj_value)

    def compute_pri_dua_resids(self, sol: Solution, active_info: ActiveSetInfo):
        pri_res, pri_res_work = self.global_primal_residual(sol)
        dua_res, dua_res_work = self.global_dual_residual(sol, active_info)
        return PriDuaResidsAndOthers(pri_res, dua_res, pri_res_work, dua_res_work)

    def bcl_update(
        self,
        pri_feas_lhs_new: FloatScalar,
        sol: Solution,
        sol_prev: Solution,
        bcl_coeffs: BCLCoeffs,
        mults: Multipliers,
    ):
        settings = self._settings
        # These values were passed in by they aren't updated.
        # eps_in_min = np.minimum(settings.eps_abs, 1.0e-9)
        # We don't want to pick the smaller of the two, since with float32 the precision is much lower.
        eps_in_min = settings.eps_in_min
        bcl_eta_ext_init = settings.bcl_eta_ext_init

        is_good_step = pri_feas_lhs_new <= bcl_coeffs.bcl_eta_ext  # or iter > self._settings.safe_guard
        # If good, then reduce eta_ext and eta_in. eta is the tolerance for the inner and outer loops.
        bcl_eta_ext_good = bcl_coeffs.bcl_eta_ext * (mults.mu_in**settings.beta_bcl)
        bcl_eta_in_good = jnp.maximum(bcl_coeffs.bcl_eta_in * mults.mu_in, eps_in_min)

        # Otherwise, rollback the y and z.
        y_new = jnp.where(is_good_step, sol.y, sol_prev.y)
        z_new = jnp.where(is_good_step, sol.z, sol_prev.z)
        sol_new = sol.replace(y=y_new, z=z_new)

        bcl_mu_in_bad = jnp.maximum(mults.mu_in * settings.mu_update_factor, settings.mu_min_in)
        bcl_mu_eq_bad = jnp.maximum(mults.mu_eq * settings.mu_update_factor, settings.mu_min_eq)
        bcl_mu_in_inv_bad = jnp.minimum(mults.mu_in_inv * settings.mu_update_inv_factor, settings.mu_max_in_inv)
        bcl_mu_eq_inv_bad = jnp.minimum(mults.mu_eq_inv * settings.mu_update_inv_factor, settings.mu_max_eq_inv)
        bcl_eta_ext_bad = bcl_eta_ext_init * (bcl_mu_in_bad**settings.alpha_bcl)
        bcl_eta_in_bad = jnp.maximum(bcl_mu_in_bad, eps_in_min)

        bcl_eta_ext_new = jnp.where(is_good_step, bcl_eta_ext_good, bcl_eta_ext_bad)
        bcl_eta_in_new = jnp.where(is_good_step, bcl_eta_in_good, bcl_eta_in_bad)
        bcl_mu_in_new = jnp.where(is_good_step, bcl_coeffs.bcl_mu_in, bcl_mu_in_bad)
        bcl_mu_in_inv_new = jnp.where(is_good_step, bcl_coeffs.bcl_mu_in_inv, bcl_mu_in_inv_bad)
        bcl_mu_eq_new = jnp.where(is_good_step, bcl_coeffs.bcl_mu_eq, bcl_mu_eq_bad)
        bcl_mu_eq_inv_new = jnp.where(is_good_step, bcl_coeffs.bcl_mu_eq_inv, bcl_mu_eq_inv_bad)

        bcl_coeffs_new = BCLCoeffs(
            bcl_eta_ext_new, bcl_eta_in_new, bcl_mu_in_new, bcl_mu_eq_new, bcl_mu_in_inv_new, bcl_mu_eq_inv_new
        )

        if self._settings.verbose:
            jd.print("good_step: {}", is_good_step, ordered=True)
        return sol_new, bcl_coeffs_new

    def global_primal_residual(self, sol: Solution) -> Tuple[PriRes, PriResWork]:
        ruiz = self._ruiz

        # qpresults.se
        se = ruiz.unscale_pri_res_eq(self.A_scaled @ sol.x)
        pri_feas_eq_rhs_0 = infty_norm(se)

        # if box_connstraints
        pri_res_in_sc_up_tail = ruiz.unscale_pri(sol.x)

        # qpwork.primal_residual_in_scaled_up.head(qpmodel.n_in)
        pri_res_in_sc_up_head = ruiz.unscale_pri_res_in(self.C_scaled @ sol.x)
        pri_feas_in_rhs_0 = infty_norm(pri_res_in_sc_up_head)

        si_head = pos_part(pri_res_in_sc_up_head - self._qp.u) + neg_part(pri_res_in_sc_up_head - self._qp.l)
        # This is probably a bug? Should probably be taking the infty_norm of unscaled x...
        # si_tail = qpresults.si[-self.dim :]

        # if box constraints. si_tail should contain scaled(x + z_prev * mu_in).
        pri_feas_in_rhs_0 = jnp.maximum(pri_feas_in_rhs_0, infty_norm(pri_res_in_sc_up_tail))
        si_tail = pos_part(pri_res_in_sc_up_tail - self._qp.u_box) + neg_part(pri_res_in_sc_up_tail - self._qp.l_box)

        se = se - self._qp.b
        pri_feas_eq_lhs = infty_norm(se)

        # TODO: Leave head and tail separated?
        si = jnp.concatenate([si_head, si_tail])
        pri_feas_in_lhs = infty_norm(si)

        pri_feas_lhs = jnp.maximum(pri_feas_eq_lhs, pri_feas_in_lhs)

        se = ruiz.scale_pri_res_eq(se)

        pri_res = PriRes(pri_feas_lhs, pri_feas_eq_rhs_0, pri_feas_in_rhs_0, pri_feas_eq_lhs, pri_feas_in_lhs)
        # qpresults_new = qpresults._replace(se=se, si=si)
        # pri_res_in_sc_up = jnp.concatenate([pri_res_in_sc_up_head, pri_res_in_sc_up_tail], axis=0)

        # Alias to what they really represent.
        eq_res = se
        in_res = si_head
        Cx_unscaled = pri_res_in_sc_up_head
        x_unscaled = pri_res_in_sc_up_tail

        # qpwork_new = qpwork._replace(primal_residual_in_scaled_up=pri_res_in_sc_up)
        pri_resids = PriResWork(eq_res, in_res, Cx_unscaled, x_unscaled)

        return pri_res, pri_resids

    def global_dual_residual(self, sol: Solution, active_info: ActiveSetInfo):
        """
        The original function clobbers CTz. Here, we keep it alone.
        This also computes
            qpwork.dual_residual_scaled = g + Hx + A^T y + C^T z + z
        """
        ruiz = self._ruiz
        qpwork_dua_res_scaled = self.g_scaled

        # I think they are clobbering CTz here.
        CTz = self.H_scaled @ sol.x
        qpwork_dua_res_scaled = qpwork_dua_res_scaled + CTz

        # Contains unscaled Hx.
        CTz = ruiz.unscale_dua_res(CTz)
        dua_feas_rhs_0 = infty_norm(CTz)

        x = ruiz.unscale_pri(sol.x)
        dua_gap = jnp.dot(self._qp.g, x)
        rhs_dua_gap = jnp.abs(dua_gap)

        xHx = jnp.dot(CTz, x)
        # Now contains <x, Hx> + <g, x>
        dua_gap = dua_gap + xHx
        rhs_dua_gap = jnp.maximum(rhs_dua_gap, jnp.abs(xHx))
        # x = ruiz.scale_pri(x)

        # Compute A^T y
        CTz = self.A_scaled.T @ sol.y
        qpwork_dua_res_scaled = qpwork_dua_res_scaled + CTz
        CTz = ruiz.unscale_dua_res(CTz)
        dua_feas_rhs_1 = infty_norm(CTz)

        # Compute C^T z
        CTz = self.C_scaled.T @ sol.z_head
        qpwork_dua_res_scaled = qpwork_dua_res_scaled + CTz
        CTz = ruiz.unscale_dua_res(CTz)
        dua_feas_rhs_3 = infty_norm(CTz)

        # if box constraints
        CTz = sol.z_tail
        CTz = CTz * self.i_scaled
        qpwork_dua_res_scaled = qpwork_dua_res_scaled + CTz
        CTz = ruiz.unscale_dua_res(CTz)
        dua_feas_rhs_3 = jnp.maximum(dua_feas_rhs_3, infty_norm(CTz))

        qpwork_dua_res_unscaled = ruiz.unscale_dua_res(qpwork_dua_res_scaled)
        dua_feas_lhs = infty_norm(qpwork_dua_res_unscaled)
        # Do we need this?
        qpwork_dua_res_scaled = ruiz.scale_dua_res(qpwork_dua_res_unscaled)

        # b^T y
        y = ruiz.unscale_dua_eq(sol.y)
        by = jnp.dot(self._qp.b, y)
        rhs_dua_gap = jnp.maximum(rhs_dua_gap, jnp.abs(by))
        dua_gap = dua_gap + by
        # # Unused, since we are not modifying in place.
        # y = ruiz.scale_dua_eq(y)

        # zu and zl.
        z_head = ruiz.unscale_dua_in(sol.z_head)
        zu = jnp.where(active_info.active_set_hi_head, z_head, 0).dot(self._qp.u)
        rhs_dua_gap = jnp.maximum(rhs_dua_gap, jnp.abs(zu))
        dua_gap = dua_gap + zu

        zl = jnp.where(active_info.active_set_lo_head, z_head, 0).dot(self._qp.l)
        rhs_dua_gap = jnp.maximum(rhs_dua_gap, jnp.abs(zl))
        dua_gap = dua_gap + zl
        # # Unused, since we are not modifying in place.
        # z_head = ruiz.scale_dua_in(z_head)

        # zu and zl for box constraints
        z_tail = ruiz.unscale_box_dua_in(sol.z_tail)
        zu_box = jnp.where(active_info.active_set_hi_tail, z_tail, 0).dot(self._qp.u_box)
        rhs_dua_gap = jnp.maximum(rhs_dua_gap, jnp.abs(zu_box))
        dua_gap = dua_gap + zu_box

        zl_box = jnp.where(active_info.active_set_lo_tail, z_tail, 0).dot(self._qp.l_box)
        rhs_dua_gap = jnp.maximum(rhs_dua_gap, jnp.abs(zl_box))
        dua_gap = dua_gap + zl_box
        # # Unused, since we are not modifying in place.
        # z_tail = ruiz.scale_box_dua_in(z_tail)

        dua_res = DuaRes(dua_feas_lhs, dua_feas_rhs_0, dua_feas_rhs_1, dua_feas_rhs_3, rhs_dua_gap, dua_gap)
        # qpwork_new = qpwork._replace(dual_residual_scaled=qpwork_dua_res_scaled)
        return dua_res, DuaResWork(qpwork_dua_res_scaled)

    def primal_dual_newton_semi_smooth(
        self, res: Resids, sol: Solution, sol_prev: Solution, mults: Multipliers, eps_int: FloatScalar
    ):
        CarryType = Tuple[int, Resids, Solution, FloatScalar, FloatScalar]

        def cond(carry: CarryType) -> BoolScalar:
            ii, res_, sol_, err_in, step_size = carry

            # Exit if the step size is too small or the error is small enough.
            step_size_too_small = step_size < self._settings.step_size_thresh

            err_large = err_in > eps_int
            should_continue = err_large & (~step_size_too_small)
            return should_continue

        def body(carry: CarryType):
            ii, res_, sol_, err_in_prev, step_size_prev = carry
            dw_aug, active_set_info, res_work = self.primal_dual_semi_smooth_newton_step(res_, sol_, mults, eps_int)
            dx, dy, dz, dz_head, dz_tail = self.split_wvec(dw_aug)

            Cdx_head = self.C_scaled @ dx
            CTdz = self.C_scaled.T @ dz_head

            CTdz = CTdz + dz_tail * self.i_scaled
            Cdx_tail = dx * self.i_scaled

            # switch case GPDAL.
            Cdx = jnp.concatenate([Cdx_head, Cdx_tail], axis=0)
            Cdx = Cdx + (self._settings.alpha_gpdal - 1.0) * mults.mu_in * dz

            # if n_in > 0 || box_constraints
            # do EXACT non-smooth linesearch.
            matvecs = MatVecs(res_work.Hdx, res_work.Adx, res_work.ATdy, Cdx)
            alpha = self.primal_dual_ls(dw_aug, res_, matvecs, sol_, sol_prev, mults)

            step_size = infty_norm(alpha * dw_aug)

            x = sol_.x + alpha * dx
            y = sol_.y + alpha * dy
            z = sol_.z + alpha * dz

            # Contains now:  C( x + alpha * dx ) - u + mu_in * z_prev
            in_res_hi = res_.in_res_hi + alpha * Cdx
            # Contains now:  C( x + alpha * dx ) - l + mu_in * z_prev
            in_res_lo = res_.in_res_lo + alpha * Cdx
            eq_res = res_.eq_res + alpha * (res_work.Adx - mults.mu_eq * dy)
            dua_res_scaled = res_.dua_res + alpha * (mults.rho * dx + res_work.Hdx + res_work.ATdy + CTdz)

            res_new = Resids(dua_res_scaled, eq_res, in_res_hi, in_res_lo)
            sol_new = sol_.replace(x=x, y=y, z=z)
            err_in = self.compute_inner_loop_saddle_point(res_new, sol_new, mults)

            if self._settings.verbose:
                self._print_inner_loop(ii, err_in, alpha, step_size, eps_int)

            return ii + 1, res_new, sol_new, err_in, step_size

        carry_init = (0, res, sol, 1.0, 1.0)

        # Run it once before the check, since there's not point in checking before running the loop.
        carry_init2 = body(carry_init)
        carry_out = eqi.while_loop(cond, body, carry_init2, max_steps=self._settings.max_iter_in, kind="lax")

        # Recompute active_set_info from res.
        # ii_out is always at least 1.
        ii_out, res_out, sol_out, err_in_out, step_size = carry_out
        active_set_hi = res_out.in_res_hi >= 0
        active_set_lo = res_out.in_res_lo <= 0
        active_set_info_out = ActiveSetInfo(active_set_hi, active_set_lo, self.n_in)

        return ii_out, res_out, sol_out, active_set_info_out

    def compute_inner_loop_saddle_point(self, res: Resids, sol: Solution, mults: Multipliers) -> FloatScalar:
        active_part_z = pos_part(res.in_res_hi) + neg_part(res.in_res_lo)
        # Ok...?
        active_part_z = active_part_z - self._settings.alpha_gpdal * sol.z * mults.mu_in

        err = infty_norm(active_part_z)
        err_eq = res.eq_res

        # || Ax - b - (y - y_prev) / mu ||
        prim_eq_e = infty_norm(err_eq)
        err = jnp.maximum(err, prim_eq_e)

        # || Hx + rho (x - xprev) + g + <A, y> + <C, z> ||
        dual_e = infty_norm(res.dua_res)

        err = jnp.maximum(err, dual_e)
        return err

    def primal_dual_ls(
        self, dw: WVec, res: Resids, matvecs: MatVecs, sol: Solution, sol_prev: Solution, mults: Multipliers
    ):
        machine_eps = get_machine_eps()

        def gpdal_derivative_results(alpha_):
            return self.gpdal_derivative_results(dw, res, matvecs, sol, sol_prev, mults, alpha_)

        # Get all alphas, with a mask for if they are valid or not.
        alpha_h = -res.in_res_hi / (matvecs.Cdx + machine_eps)
        alpha_l = -res.in_res_lo / (matvecs.Cdx + machine_eps)
        Cdx_nonzero = matvecs.Cdx != 0

        # Append only if Cdx is not zero, and alpha is larger than machine_eps.
        # Since we are sorting, make all the invalid alphas a super large number.
        alpha_h = jnp.where(Cdx_nonzero & (alpha_h > machine_eps), alpha_h, jnp.inf)
        alpha_l = jnp.where(Cdx_nonzero & (alpha_l > machine_eps), alpha_l, jnp.inf)
        a_alphas = jnp.concatenate([alpha_h, alpha_l], axis=0)
        n_alphas = len(a_alphas)

        all_invalid = jnp.all(jnp.isinf(a_alphas))
        # a_alphas = eqx.error_if(a_alphas, all_invalid, "All alphas are invalid!")

        # def true_fn():
        #     jd.breakpoint()
        #
        # def false_fn():
        #     return None
        # lax.cond(all_invalid, true_fn, false_fn)

        # Sort alphas.
        a_alphas_sorted = jnp.sort(a_alphas)

        # Find first alpha that satisfies all conditions.
        # State: (alpha_last_neg, last_neg_grad, alpha_first_pos, alpha_pos_grad)
        def cond_fun(state: Tuple[int, LineSearchAlphaCarry]) -> BoolScalar:
            # Keep going if we have not found a positive alpha.
            i, (alpha_last_neg_, last_neg_grad_, alpha_first_pos_, alpha_pos_grad_) = state
            should_continue = jnp.isinf(alpha_first_pos_)
            return should_continue

        def body_fun(state: Tuple[int, LineSearchAlphaCarry]):
            i, (alpha_last_neg_, last_neg_grad_, alpha_first_pos_, first_pos_grad_) = state
            alpha = a_alphas_sorted[i]
            is_alpha_valid = jnp.isfinite(alpha)

            gpdal_res = gpdal_derivative_results(alpha)
            gr = gpdal_res.grad
            # jd.print("    i={}    gr={}, a={}, b={}, alp={}", i, gr, gpdal_res.a, gpdal_res.b, alpha, ordered=True)
            is_neg = gr < 0
            is_pos = ~is_neg

            # Update if negative and alpha is valid.
            alpha_last_neg_ = jnp.where(is_neg & is_alpha_valid, alpha, alpha_last_neg_)
            last_neg_grad_ = jnp.where(is_neg & is_alpha_valid, gr, last_neg_grad_)

            # Update if positive and alpha is valid => Keep old value if negative.
            # alpha_first_pos_ = jnp.where(is_neg, alpha_first_pos_, alpha)
            # first_pos_grad_ = jnp.where(is_neg, first_pos_grad_, gr)
            alpha_first_pos_ = jnp.where(is_pos & is_alpha_valid, alpha, alpha_first_pos_)
            first_pos_grad_ = jnp.where(is_pos & is_alpha_valid, gr, first_pos_grad_)

            return i + 1, LineSearchAlphaCarry(alpha_last_neg_, last_neg_grad_, alpha_first_pos_, first_pos_grad_)

        zero_int = jnp.zeros((), dtype=jnp.int32)
        zero_float = jnp.zeros((), dtype=default_float())
        inf_float = jnp.full((), jnp.inf, dtype=default_float())
        init_val = (zero_int, LineSearchAlphaCarry(zero_float, inf_float, inf_float, inf_float))
        _, output = eqi.while_loop(cond_fun, body_fun, init_val, max_steps=n_alphas, kind="lax")
        alpha_last_neg, last_neg_grad, alpha_first_pos, first_pos_grad = output

        # If the first alpha is positive, then set last_neg_grad to alpha=0.
        first_alpha_is_positive = alpha_last_neg == 0.0
        zero_res = gpdal_derivative_results(0.0)
        zero_grad = zero_res.grad
        last_neg_grad = jnp.where(first_alpha_is_positive, zero_grad, last_neg_grad)

        # If all alphas are invalid, then return -b/a for alpha=0.
        all_invalid_alpha = -zero_res.b / zero_res.a

        # If we never see a positive_alpha, then the optimal alpha is within the interval [last_alpha_neg, +∞).
        is_all_neg = jnp.isinf(alpha_first_pos)
        all_neg_res = gpdal_derivative_results(2 * alpha_last_neg + 1)
        all_neg_alpha = -all_neg_res.b / all_neg_res.a

        # Otherwise, the optimal alpha is in the interval [last_alpha_neg, first_alpha_pos] and can be computed exactly
        # as phi' is affine in alpha.
        alpha = alpha_last_neg - last_neg_grad * (alpha_first_pos - alpha_last_neg) / (first_pos_grad - last_neg_grad)
        alpha = jnp.abs(alpha)

        alpha = jnp.where(all_invalid, all_invalid_alpha, jnp.where(is_all_neg, all_neg_alpha, alpha))

        # If alpha is nan or inf, then to prevent it from spreading just set it to some tiny number.
        alpha = jnp.where(~jnp.isfinite(alpha), 10 * machine_eps, alpha)

        return alpha

    def gpdal_derivative_results(
        self,
        dw: WVec,
        res: Resids,
        matvecs: MatVecs,
        sol: Solution,
        sol_prev: Solution,
        mults: Multipliers,
        alpha: FloatScalar,
    ) -> PrimalDualDerivResult:
        """Assumes that
        qpwork.pri_res_in_sc_up             = scaled( Cx - u + mu_in * z_prev )
        qpresults.si                        = scaled( Cx - l + mu_in * z_prev )
        """
        Cx_min_u_p_mu_z = res.in_res_hi
        Cx_min_l_p_mu_z = res.in_res_lo

        pri_res_in_sc_up_plus_alphaCdx = Cx_min_u_p_mu_z + matvecs.Cdx * alpha
        pri_res_in_sc_low_plus_alphaCdx = Cx_min_l_p_mu_z + matvecs.Cdx * alpha
        up_plus_alphaCdx, low_plus_alphaCdx = pri_res_in_sc_up_plus_alphaCdx, pri_res_in_sc_low_plus_alphaCdx

        # Contains a = <dx, H dx> + rho * norm(dx)^2 + mu_eq_inv * norm(A dx)^2
        info = mults
        mu_eq, mu_eq_inv, mu_in, mu_in_inv, rho = info.mu_eq, info.mu_eq_inv, info.mu_in, info.mu_in_inv, info.rho
        dx, dy, dz, dz_head, dz_tail = self.split_wvec(dw)
        Hdx, Adx = matvecs.Hdx, matvecs.Adx

        a = jnp.dot(dx, Hdx) + mu_eq_inv * sqnorm(Adx) + rho * sqnorm(dx)
        # jd.print("1:  a={}", a, ordered=True)

        # Contains a = <dx, H dx> + rho * norm(dx)^2 + mu_eq_inv * norm(A dx)^2 + mu_eqinv * norm(Adx-dy*mu_eq)^2
        err_y = Adx - dy * mu_eq
        a = a + sqnorm(err_y) * mu_eq_inv
        # jd.print("2:  a={}", a, ordered=True)

        # Contains b = <dx, Hx + rho * (x - xe) + g> + mu_eq_inv * <Adx, res_eq>
        err_x = info.rho * (sol.x - sol_prev.x) + self.g_scaled
        b = jnp.dot(sol.x, Hdx) + jnp.dot(err_x, dx) + mu_eq_inv * jnp.dot(Adx, res.eq_res + sol.y * mu_eq)

        # Contains b = <dx, Hx + rho * (x - xe) + g> + mu_eq_inv * <Adx, res_eq>
        #              + nu * mu_eq_inv * <Adx -dy * mu_eq, res_eq - y * mu_eq>
        rhs_y = res.eq_res
        b = b + mu_eq_inv * jnp.dot(err_y, rhs_y)

        # Helpers.
        up_active = up_plus_alphaCdx > 0.0
        lo_active = low_plus_alphaCdx < 0.0

        # Derive Cdx_act
        err_ix = jnp.where(up_active | lo_active, matvecs.Cdx, 0)
        # Contains a = <dx, H dx> + rho * norm(dx)^2 + mu_eq_inv * norm(A dx)^2 + mu_eqinv * norm(Adx-dy*mu_eq)^2
        #              + norm(dw_act)^2 / (mu_in * alpha_gpdal
        a = a + mu_in_inv * sqnorm(err_ix) / self._alpha_gpdal
        # jd.print("3:  a={}", a, ordered=True)

        a = a + mu_in * (1.0 - self._alpha_gpdal) * sqnorm(dz)
        # jd.print("4:  a={}", a, ordered=True)

        # Derive vector [w-u]_+ + [w-l]_-
        active_part_z = jnp.where(up_active, Cx_min_u_p_mu_z, 0) + jnp.where(lo_active, Cx_min_l_p_mu_z, 0)

        # Contains b = <dx, Hx + rho * (x - xe) + g> + mu_eq_inv * <Adx, res_eq>
        #              + nu * mu_eq_inv * <Adx -dy * mu_eq, res_eq - y * mu_eq>
        #              + mu_in_inv * <dw_act, [w-u]_+ + [w-l]_- > / alpha_gpdal
        b = b + mu_in_inv * jnp.dot(active_part_z, err_ix) / self._alpha_gpdal

        # lol the comment is wrong?
        b = b + mu_in * (1 - self._alpha_gpdal) * jnp.dot(dz, sol.z)

        return PrimalDualDerivResult(a, b, a * alpha + b)

    def primal_dual_semi_smooth_newton_step(self, res: Resids, sol: Solution, mults: Multipliers, eps: FloatScalar):
        """
        dua_res     =   qpwork.dual_res_scaled  = Hx + g + <A, y> + <C, z>
        eq_res      =   qpresults.se            = Ax - b
        in_res_hi   =   qpwork.pri_res_in_sc_up = scaled( Cx - u + mu_in * z_prev )
        in_res_lo   =   qpresults.si            = scaled( Cx - l + mu_in * z_prev )
        """
        dim, n_eq, n_in = self.dim, self.n_eq, self.n_in

        active_set_hi = res.in_res_hi >= 0
        active_set_lo = res.in_res_lo <= 0
        active_ineqs = active_set_hi | active_set_lo

        rhs_dx = -res.dua_res
        rhs_dy = -res.eq_res
        rhs_dz = jnp.where(active_set_hi, -res.in_res_hi, -res.in_res_lo) + sol.z * mults.mu_in * self._alpha_gpdal
        rhs_dz = jnp.where(active_ineqs, rhs_dz, 0)

        active_part_z_tail = sol.z_tail * self.i_scaled

        # We also need to remove the contribution of the <C, z> part for inactive constraints.
        # (n_in, dim) * (n_in, 1)
        inactive_z = jnp.where(active_ineqs, 0, jnp.concatenate([sol.z_head, active_part_z_tail], axis=0))
        assert inactive_z.shape == (n_in + dim,)

        Cz_part = self.C_scaled.T @ inactive_z[:n_in]
        assert Cz_part.shape == (dim,)
        rhs_dx = rhs_dx + Cz_part + inactive_z[n_in:]

        rhs = jnp.concatenate([rhs_dx, rhs_dy, rhs_dz])
        dw_aug, res_work_newton = self.iterative_solve_with_permut_fact(rhs, active_ineqs, mults, eps)

        # After solving, replace the inactive dz solutions with dz=-z.
        # We don't need to permute since the dz is all in order.
        dz = dw_aug[-(n_in + dim) :]
        dz = jnp.where(active_ineqs, dz, -sol.z)
        dw_aug = dw_aug.at[-(n_in + dim) :].set(dz)

        active_set_info = ActiveSetInfo(active_set_hi, active_set_lo, self.n_in)

        return dw_aug, active_set_info, res_work_newton

    def iterative_solve_with_permut_fact(
        self, rhs: WVec, active_ineqs: IXBool, mults: Multipliers, eps: FloatScalar
    ) -> Tuple[WVec, ResWorkStep]:
        dw_aug, lu_factors = self.solve_linear_system(rhs, active_ineqs, mults)

        # TODO: Do we need to compute residuals in flaot64?
        err, res_work = self.iterative_residual(dw_aug, rhs, active_ineqs, mults)

        # Iterative refinement.
        preverr = infty_norm(err)
        do_iterative_refine = preverr >= eps

        # If we aren't doing iterative refine, then we can early exit.
        if self._settings.max_iterative_refine == 0:
            res_work = eqx.error_if(res_work, do_iterative_refine, "Need to do iterative refine...")
            return dw_aug, res_work

        # If we need to do iterative refine, then while loop.
        def ir_cond(carry) -> BoolScalar:
            ii, dw_aug_, err_, errnorm, _, it_stability = carry

            # Break if our errornorm is smaller than eps.
            low_err = errnorm < eps
            # Also break if our numerical stability is becoming worse.
            stability_fail = it_stability >= 2
            should_break = low_err | stability_fail
            should_continue = ~should_break
            return should_continue

        def ir_body(carry):
            # Solve linear system using err we computed from iterative_residual.
            ii, dw_aug_, err_, errnorm_prev, _, it_stability = carry
            sol = self.solve_linear_system_from_lu(err_, lu_factors)
            dw_aug_ = dw_aug_ + sol

            # Recompute error again using the new dw_aug.
            if self._settings.use_f64_refine and float32_is_default():
                # If we are using float32 and use_f64_refine is true, then use float64 for the residual.
                err_new, res_work_new = self.iterative_residual_f64(dw_aug_, rhs, active_ineqs, mults)
            else:
                err_new, res_work_new = self.iterative_residual(dw_aug_, rhs, active_ineqs, mults)

            errnorm_new = infty_norm(err_new)

            # If our error norm is getting worse, then increment it_stability.
            worse_stability = errnorm_new > errnorm_prev
            it_stability_new = jnp.where(worse_stability, it_stability + 1, 0)

            return ii + 1, dw_aug_, err_new, errnorm_new, res_work_new, it_stability_new

        zero = jnp.zeros((), dtype=jnp.int32)
        carry_init = (zero, dw_aug, err, preverr, res_work, zero)
        carry_out = eqi.while_loop(
            ir_cond, ir_body, carry_init, max_steps=self._settings.max_iterative_refine, kind="lax"
        )

        ii_out, dw_aug_out, err_out, errnorm_out, res_work_out, it_stability_out = carry_out

        if self._settings.verbose:

            def do_print():
                pd.print(
                    "\033[1;31m[Need iterative refine!]\033[0m " "err={:10.4e} -> {:10.4e} in {} iters, eps={:8.2e}",
                    preverr,
                    errnorm_out,
                    ii_out,
                    eps,
                )

            lax.cond(do_iterative_refine, do_print, lambda: None)

        return dw_aug_out, res_work_out

    def iterative_residual_f64(self, dw_aug: WVec, rhs: WVec, active_ineqs: IXBool, mults: Multipliers):
        # Compute residuals in double precision, then round to single precision.
        f32, f64 = jnp.float32, jnp.float64
        dw_aug_64 = dw_aug.astype(f64)
        dx, dy, dz, dz_head, dz_tail = self.split_wvec(dw_aug_64)

        is_active_head, is_active_tail = active_ineqs[: self.n_in], active_ineqs[self.n_in :]

        H_scaled = self.H_scaled.astype(f64)
        A_scaled = self.A_scaled.astype(f64)
        C_scaled = self.C_scaled.astype(f64)
        i_scaled = self.i_scaled.astype(f64)

        #################################
        qpwork_err = rhs.astype(f64)
        Hdx = H_scaled @ dx
        add_vec_x = -Hdx - mults.rho * dx
        #################################
        ATdy = A_scaled.T @ dy
        add_vec_x = add_vec_x - ATdy
        #################################
        active_part_z_tail = dx * i_scaled
        #################################
        # Box ineqs.
        add_vec_x = add_vec_x - jnp.where(is_active_tail, dz_tail * i_scaled, 0)
        add_vec_box = -(active_part_z_tail - dz_tail * mults.mu_in)
        # TODO: If we are masking in the end, this is not needed.
        add_vec_box = jnp.where(is_active_tail, add_vec_box, 0)
        #################################
        # Regular ineq
        add_vec_x = add_vec_x - C_scaled.T @ jnp.where(is_active_head, dz_head, 0)
        add_vec_in = -(C_scaled @ dx - dz_head * mults.mu_in)
        add_vec_in = jnp.where(is_active_head, add_vec_in, 0)
        #################################
        # Eq
        Adx = A_scaled @ dx
        add_vec_y = -Adx + dy * mults.mu_eq
        #################################

        qpwork_err_head = qpwork_err[: self.dim + self.n_eq] + jnp.concatenate([add_vec_x, add_vec_y], axis=0)
        qpwork_err_tail = qpwork_err[self.dim + self.n_eq :] + jnp.concatenate([add_vec_in, add_vec_box], axis=0)
        qpwork_err_tail = jnp.where(active_ineqs, qpwork_err_tail, 0)
        qpwork_err = jnp.concatenate([qpwork_err_head, qpwork_err_tail], axis=0)

        assert qpwork_err.dtype == jnp.float64
        return qpwork_err.astype(f32), ResWorkStep(Hdx.astype(f32), Adx.astype(f32), ATdy.astype(f32))

    def iterative_residual(self, dw_aug: WVec, rhs: WVec, active_ineqs: IXBool, mults: Multipliers):
        dx, dy, dz, dz_head, dz_tail = self.split_wvec(dw_aug)

        is_active_head, is_active_tail = active_ineqs[: self.n_in], active_ineqs[self.n_in :]

        #################################
        qpwork_err = rhs
        Hdx = self.H_scaled @ dx
        add_vec_x = -Hdx - mults.rho * dx
        #################################
        ATdy = self.A_scaled.T @ dy
        add_vec_x = add_vec_x - ATdy
        #################################
        active_part_z_tail = dx * self.i_scaled
        #################################
        # Box ineqs.
        add_vec_x = add_vec_x - jnp.where(is_active_tail, dz_tail * self.i_scaled, 0)
        add_vec_box = -(active_part_z_tail - dz_tail * mults.mu_in)
        # TODO: If we are masking in the end, this is not needed.
        add_vec_box = jnp.where(is_active_tail, add_vec_box, 0)
        #################################
        # Regular ineq
        add_vec_x = add_vec_x - self.C_scaled.T @ jnp.where(is_active_head, dz_head, 0)
        add_vec_in = -(self.C_scaled @ dx - dz_head * mults.mu_in)
        add_vec_in = jnp.where(is_active_head, add_vec_in, 0)
        #################################
        # Eq
        Adx = self.A_scaled @ dx
        add_vec_y = -Adx + dy * mults.mu_eq
        #################################

        qpwork_err_head = qpwork_err[: self.dim + self.n_eq] + jnp.concatenate([add_vec_x, add_vec_y], axis=0)
        qpwork_err_tail = qpwork_err[self.dim + self.n_eq :] + jnp.concatenate([add_vec_in, add_vec_box], axis=0)
        qpwork_err_tail = jnp.where(active_ineqs, qpwork_err_tail, 0)
        qpwork_err = jnp.concatenate([qpwork_err_head, qpwork_err_tail], axis=0)

        return qpwork_err, ResWorkStep(Hdx, Adx, ATdy)

    def solve_linear_system_from_lu(self, rhs: WVec, lu_factors) -> WVec:
        result = jsp.linalg.lu_solve(lu_factors, rhs)
        return result

    def solve_linear_system(self, rhs: WVec, active_ineqs: IXBool, mults: Multipliers):
        n, n_eq, n_in = self.dim, self.n_eq, self.n_in

        # Mask out inactive constraints. active_ineqs: (n_ineq, ),  C: (n_ineq, dim))
        assert active_ineqs.shape == (n_in + n,)
        C_active = jnp.where(active_ineqs[:n_in, None], self.C_scaled, 0)
        i_scaled_active = jnp.where(active_ineqs[n_in:], self.i_scaled, 0)

        # (n_eq, n), (n_in, n), (n, n) -> (n_eq + n_in + n, n)
        constr_col = jnp.concatenate([self.A_scaled, C_active, jnp.diag(i_scaled_active)], axis=0)
        assert constr_col.shape == (n_eq + n_in + n, n)

        # noinspection PyTypeChecker
        kkt = jnp.block(
            [
                [self.H_scaled, constr_col.T],
                [constr_col, jnp.zeros((n_eq + n_in + n, n_eq + n_in + n))],
            ]
        )
        # Add the regularization terms.
        kkt = add_diag_seg(kkt, 0, n, mults.rho)
        kkt = add_diag_seg(kkt, n, n_eq, -mults.mu_eq)
        kkt = add_diag_seg(kkt, n + n_eq, n_in + n, -mults.mu_in)

        # Solve     KKT @ x = dw
        lu_factors = jsp.linalg.lu_factor(kkt)
        result = jsp.linalg.lu_solve(lu_factors, rhs)
        return result, lu_factors
