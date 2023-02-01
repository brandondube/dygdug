"""vAPP optimizer."""

from prysm.mathops import np, fft

from prysm.propagation import focus_fixed_sampling, focus_fixed_sampling_backprop

def ft_fwd(x):
    return fft.ifftshift(fft.fft2(fft.fftshift(x), norm='ortho'))

def ft_rev(x):
    return fft.ifftshift(fft.ifft2(fft.fftshift(x), norm='ortho'))


def normalize(vapp):
    phs = np.zeros_like(vapp.amp)
    old_phs = vapp.phs.copy()
    vapp.phs = phs
    vapp.fwd(vapp.phs[vapp.amp_select])
    Imax = vapp.I.max()
    norm = np.sqrt(Imax)
    vapp.amp = vapp.amp * (1/norm)  # cheaper to divide scalar, mul array
    vapp.fwd(old_phs[vapp.amp_select])
    return

class VAPPOptimizer:
    def __init__(self, amp, wvl, basis, dark_hole, dh_target=1e-10, initial_phase=None):
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
            phs = np.tensordot(self.basis, x, axes=(0,0))
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


class VAPPOptimizer2:
    def __init__(self, amp, amp_dx, efl, wvl, basis, dark_hole, dh_dx, dh_target=1e-10, initial_phase=None):
        if initial_phase is None:
            phs = np.zeros(amp.shape, dtype=float)

        self.amp = amp
        self.amp_select = self.amp > 1e-9
        self.amp_dx = amp_dx
        self.efl = efl
        self.wvl = wvl
        self.basis = basis
        self.dh = dark_hole
        self.dh_dx = dh_dx
        self.D = dh_target
        self.phs = phs
        self.zonal = False

    def set_optimization_method(self, zonal=False):
        self.zonal = zonal

    def update(self, x):
        if not self.zonal:
            phs = np.tensordot(self.basis, x, axes=(0,0))
        else:
            phs = np.zeros(self.amp.shape, dtype=float)
            phs[self.amp_select] = x

        W = (2 * np.pi / self.wvl) * phs
        g = self.amp * np.exp(1j * W)
        # G = ft_fwd(g)
        G = focus_fixed_sampling(
            wavefunction=g,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.dh.shape,
            shift=(0, 0),
            method='mdft')
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
        # gbar = ft_rev(Gbar)
        gbar = focus_fixed_sampling_backprop(
            wavefunction=Gbar,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.phs.shape,
            shift=(0, 0),
            method='mdft')

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
