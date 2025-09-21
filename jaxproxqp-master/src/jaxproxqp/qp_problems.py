from typing import NamedTuple

import numpy as np

from jaxproxqp.qp_types import EMat, EVec, IMat, IVec, XMat, XVec


class QPModel(NamedTuple):
    H: XMat
    g: XVec

    A: EMat
    C: IMat

    b: EVec
    u: IVec
    l: IVec

    u_box: XVec
    l_box: XVec

    @staticmethod
    def create(H, g, C, u, l_box, u_box, l=None, A=None, b=None):
        n_in, dim = C.shape

        assert H.shape == (dim, dim)
        assert g.shape == (dim,)
        assert C.shape == (n_in, dim)
        assert u.shape == (n_in,)
        assert l_box.shape == u_box.shape == (dim,)

        if l is None:
            # l = np.full(u.shape, -np.inf, dtype=u.dtype)
            l = np.full(u.shape, -1e9, dtype=u.dtype)
        assert l.shape == u.shape

        if A is None:
            A = np.zeros((0, dim), dtype=u.dtype)

            assert b is None
            b = np.zeros((0,), dtype=u.dtype)

        return QPModel(H, g, A, C, b, u, l, u_box, l_box)

    @property
    def dim(self):
        return self.H.shape[0]

    @property
    def n_eq(self):
        return self.A.shape[0]

    @property
    def n_in(self):
        return self.C.shape[0]

    def __str__(self) -> str:
        H, g = str(self.H), str(self.g)
        A, C = str(self.A), str(self.C)
        b, u, l = str(self.b), str(self.u), str(self.l)
        u_box, l_box = str(self.u_box), str(self.l_box)
        return (
            f"nx: {self.dim}, ne: {self.n_eq}, ni: {self.n_in}\n"
            f"        | H |:\n{H}\n"
            f"> g: {g}  \n\n"
            f"        | A |:\n{A}\n"
            f"> b: {b}  \n\n"
            f"        | C |:\n{C}\n"
            f"> l: {l}  \n> u: {u}\n"
            f"> lbox: {l_box}  \n> ubox: {u_box}\n"
        )


class QPBox(NamedTuple):
    H: XMat
    g: XVec

    A: EMat
    C: IMat

    b: EVec
    u: IVec
    l: IVec

    # What is this?
    I: IVec

    u_box: XVec
    l_box: XVec

    @property
    def dim(self):
        return self.H.shape[0]

    @property
    def n_eq(self):
        return self.A.shape[0]

    @property
    def n_in(self):
        return self.C.shape[0]

    def __str__(self) -> str:
        H, g = str(self.H), str(self.g)
        A, C = str(self.A), str(self.C)
        b, u, l = str(self.b), str(self.u), str(self.l)
        u_box, l_box = str(self.u_box), str(self.l_box)
        return (
            f"nx: {self.dim}, ne: {self.n_eq}, ni: {self.n_in}\n"
            f"        | H |:\n{H}\n"
            f"> g: {g}  \n\n"
            f"        | A |:\n{A}\n"
            f"> b: {b}  \n\n"
            f"        | C |:\n{C}\n"
            f"> l: {l}  \n> u: {u}\n"
            f"> lbox: {l_box}  \n> ubox: {u_box}\n"
        )
