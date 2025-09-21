from typing import NamedTuple

import jax.numpy as jnp

from jaxproxqp.qp_problems import QPBox
from jaxproxqp.qp_types import EVec, IVec, IXVec, XVec
from jaxproxqp.utils.jax_math_utils import infty_norm, get_machine_eps
from jaxproxqp.utils.jax_types import FloatScalar


class Ruiz(NamedTuple):
    delta: jnp.ndarray
    c: FloatScalar

    dim: int
    n_eq: int
    n_in: int

    @property
    def delta_x(self) -> XVec:
        s = 0
        return self.delta[s : s + self.dim]

    @property
    def delta_eq(self) -> EVec:
        s = self.dim
        return self.delta[s : s + self.n_eq]

    @property
    def delta_in(self) -> IVec:
        s = self.dim + self.n_eq
        return self.delta[s : s + self.n_in]

    @property
    def delta_box(self) -> XVec:
        s = self.dim + self.n_eq + self.n_in
        return self.delta[s : s + self.dim]

    @property
    def delta_ix_in(self) -> IXVec:
        s = self.dim + self.n_eq
        return self.delta[s : s + self.n_in + self.dim]

    def scale_pri(self, pri: XVec) -> XVec:
        return pri * self.delta_x

    def scale_pri_res_eq(self, pri_eq: EVec) -> EVec:
        return pri_eq * self.delta_eq

    def scale_pri_res_in(self, pri_in: IVec) -> IVec:
        return pri_in * self.delta_in

    def scale_box_pri_res_in(self, pri_box: XVec) -> XVec:
        return pri_box * self.delta_box

    def scale_dua_res(self, dua: XVec) -> XVec:
        return dua * self.delta_x * self.c

    def scale_dua_eq(self, y: EVec) -> EVec:
        return y / self.delta_eq * self.c

    def scale_dua_in(self, z: IVec) -> IVec:
        return z / self.delta_in * self.c

    def scale_box_dua_in(self, dua: XVec) -> XVec:
        return dua / self.delta_box * self.c

    ###############################################

    def unscale_pri(self, x: XVec) -> XVec:
        return x * self.delta_x

    def unscale_pri_res_eq(self, pri_eq: EVec) -> EVec:
        return pri_eq / self.delta_eq

    def unscale_pri_res_in(self, pri_in: IVec) -> IVec:
        return pri_in / self.delta_in

    def unscale_box_pri_res_in(self, pri_box_in: XVec) -> XVec:
        return pri_box_in / self.delta_box

    def unscale_dua_res(self, dua: XVec) -> XVec:
        return dua / (self.delta_x * self.c)

    def unscale_dua_eq(self, y: EVec) -> EVec:
        return y * self.delta_eq / self.c

    def unscale_dua_in(self, z: IVec) -> IVec:
        return z * self.delta_in / self.c

    def unscale_box_dua_in(self, dual: XVec) -> XVec:
        return dual * self.delta_box / self.c

    def unscale_dua_ix_in(self, dual_ix_in: IXVec) -> IXVec:
        return dual_ix_in * self.delta_ix_in / self.c

    @staticmethod
    def create(dim: int, n_eq: int, n_in: int):
        delta = jnp.ones(dim + n_eq + n_in + dim)
        c = 0.0
        return Ruiz(delta, c, dim, n_eq, n_in)

    def scale_qp(self, qp: QPBox, max_iter: int):
        c = 1.0

        i_scaled = jnp.ones(qp.dim)

        machine_eps = get_machine_eps()
        # delta = jnp.ones(qp.dim + qp.n_eq + qp.n_constraints)

        S = jnp.ones(qp.dim + qp.n_eq + qp.n_in + qp.dim)
        H, g, A, C, b, u, l, _, u_box, l_box = qp

        for ii in range(max_iter):
            ###########
            # Scale x
            ###########
            # (nx, )
            H_inf = infty_norm(H, axis=0)
            # (nx, )
            A_inf = infty_norm(A, axis=0)
            # (nx, )
            C_inf = infty_norm(C, axis=0)
            # (nx, )
            I_inf = i_scaled
            # (nx, 4)
            infs = jnp.stack([H_inf, A_inf, C_inf, I_inf], axis=1).max(-1)
            # print("infs: ", infs)
            # (nx, )
            aux = jnp.sqrt(infs)
            delta_x = jnp.where(aux == 0.0, 1.0, 1.0 / (aux + machine_eps))

            ###########
            # Scale y
            ###########
            aux = jnp.sqrt(infty_norm(A, axis=1))
            delta_eq = jnp.where(aux == 0.0, 1.0, 1.0 / (aux + machine_eps))

            ###########
            # Scale z
            ###########
            aux = jnp.sqrt(infty_norm(C, axis=1))
            delta_in = jnp.where(aux == 0.0, 1.0, 1.0 / (aux + machine_eps))

            delta_box = 1 / jnp.sqrt(i_scaled + machine_eps)

            # Normalize matrices.
            H = delta_x[:, None] * H * delta_x[None, :]
            A = delta_eq[:, None] * A * delta_x[None, :]
            C = delta_in[:, None] * C * delta_x[None, :]
            i_scaled = i_scaled * delta_x * delta_box
            u_box = u_box * delta_box
            l_box = l_box * delta_box

            # print(H)

            # Normalize vectors.
            g = g * delta_x
            b = b * delta_eq
            u = u * delta_in
            l = l * delta_in

            # Additional normalization for the cost function.
            colwise_max = H.max(axis=0).mean()
            gamma = 1 / jnp.maximum(1, colwise_max)

            delta = jnp.concatenate([delta_x, delta_eq, delta_in, delta_box])
            g = g * gamma
            S = S * delta
            c = c * gamma

            # print(delta)

        qp = QPBox(H, g, A, C, b, u, l, i_scaled, u_box, l_box)

        return self._replace(delta=S, c=c), qp
