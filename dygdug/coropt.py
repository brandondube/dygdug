"""Coronagraph optimizer."""

from prysm.mathops import np

INTENSITY = 1
CMPLX_E   = 2

class CoronagraphOptimizer:
    def __init__(self, dark_hole, coro, wvl, opt=CMPLX_E):
        self.dh = dark_hole
        self.dh_mask = np.isfinite(self.dh)
        self.dh_pix_target = self.dh[self.dh_mask]
        self.coro = coro
        self.wvl = wvl
        self.opt = opt

    def fg(self, x):
        self.coro.update_all_dms(x)
        E = self.coro.fwd(self.wvl)
        if self.opt == INTENSITY:
            aE = abs(E)
            im = aE*aE
            dh_pix = im[self.dh_mask]
            cost = np.sum((dh_pix - self.dh_pix_target)**2)
            Ibar = np.zeros_like(im)
            Ibar[self.dh_mask] = 2*(dh_pix - self.dh_pix_target)
            # Will jac-free Eq. 51
            # Ibar = 2 * eta * delta * (Idz - IT)
            # Idz = fwd I
            # IT = target
            Ebar = 2 * Ibar * E
        elif self.opt == CMPLX_E:
            # because error metic is mean square error
            # we can't quite work with complex numbers
            # all the way through,
            # since c*c != re(c)*re(c) + 1j * im(c)*im(c)
            dh_pix = E[self.dh_mask]
            diff = dh_pix - self.dh_pix_target
            reD = diff.real
            imD = diff.imag
            cost = (reD * reD).sum() + (imD * imD).sum()
            Ebar = np.zeros_like(E)
            # a lot is happening in the expression for cost,
            # consider the calculation as a graph
            #
            #          E          |||
            #         / \         |||
            #        /   \        |||
            #       /     \       |||
            #     Re       Im     |||
            #     |         |     |||
            #  Re * Re   Im * Im  ||| d = 2*(M-D)
            #  \    /    \    /   |||          .
            #   \  /      \  /    |||          .
            #    \/        \/     |||          .
            #    sum      sum     |||          .
            #    \         /      |||          .
            #     \       /       |||          .
            #      \     /        |||          .
            #       \   /         |||          .
            #        \/           |||          .
            #       sum           |||  sums are transparent
            #
            # note Jurling thesis Eq. 4.65~
            # z = x + i y
            # xbar = Re(zbar)
            # ybar = -Re(i zbar)
            Ebar[self.dh_mask] = 2*(dh_pix - self.dh_pix_target)

    # def _compute_squared_differences(weight, data, model, norm):
    #     return np.sum((model-data)**2) / norm

    # def _compute_squared_difference_grad(weight, data, model, norm):
    #     return 2*(model-data) / norm
        # return cost, Ebar
        # print(Ebar.dtype)
        grad = self.coro._rev(Ebar, self.wvl)
        return cost, grad
