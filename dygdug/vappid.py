"""vAPP optimizer."""

from prysm.mathops import np, fft

def ft_fwd(x):
    return fft.ifftshift(fft.fft2(fft.fftshift(x), norm='ortho'))

def ft_rev(x):
    return fft.ifftshift(fft.ifft2(fft.fftshift(x), norm='ortho'))


class VAPPOptimizer:
    def __init__(self, amp, wvl, basis, dark_hole, dh_target=1e-6, initial_phase=None):
        if initial_phase is None:
            phs = np.zeros(amp.shape, dtype=float)

        self.amp = amp
        self.amp_select = self.amp > 1e-9
        self.wvl = wvl
        self.basis = basis
        self.dh = dark_hole
        self.D = dh_target
        self.phs = phs
        self.zonal = False

    def set_optimization_method(self, zonal=False):
        self.zonal = zonal

    def update(self, x):
        if not self.zonal:
            phs = np.tensordot(basis, x, axes=(0,0))
        else:
            phs = np.zeros(self.amp.shape, dtype=float)
            phs[self.amp_select] = x

        W = (2 * np.pi / self.wvl) * phs
        g = self.amp * np.exp(1j * W)
        G = ft_fwd(g)
        I = np.abs(G)**2
        E = np.sum((I[self.dh] - self.D)**2)
        self.phs = phs
        self.W = W
        self.g = g
        self.G = G
        self.I = I
        self.E = E
        return

    def fwd(self, x):
        self.update(x)
        return self.E

    def rev(self, x):
        self.update(x)
        Ibar = np.zeros(self.dh.shape, dtype=float)
        Ibar[self.dh] = 2*(self.I[self.dh] - self.D)
        Gbar = 2 * Ibar * self.G
        gbar = ft_rev(Gbar)
        Wbar = 2 * np.pi / self.wvl * np.imag(gbar * np.conj(self.g))
        if not self.zonal:
            abar = np.tensordot(self.basis, Wbar)

        self.Ibar = Ibar
        self.Gbar = Gbar
        self.gbar = gbar
        self.Wbar = Wbar

        if not self.zonal:
            self.abar = abar
            return self.abar
        else:
            return self.Wbar[self.amp_select]

    def fg(self, x):
        g = self.rev(x)
        f = self.E
        return f, g
