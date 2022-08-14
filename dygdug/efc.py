from tqdm import tqdm

from prysm.mathops import np


def calculate_jacobian_fwddiff(lc, h=0.1, wvl=None):
    """Calculate the Jacobian via forward differences.

    As opposed to central differences

    Parameters
    ----------
    lc : dygdug.models.LyotCoronagraphSingleDM
        the coronagraph model
    h : float
        step size for forward differences
    wvl : float
        wavelength of light, microns

    Returns
    -------
    numpy.ndarray
        npix x nact array of complex forward differences

    """
    acts = lc.dm.actuators.copy()
    shp = acts.shape
    nact = shp[0]*shp[1]
    indices = np.arange(nact)
    iy, ix = np.unravel_index(indices, shp)
    mask = lc.dh_mask

    jac = np.empty((int(mask.sum()), nact), dtype=complex)

    # calculate resting E-field
    E0 = lc.fwd(wvl).data[mask]
    for i in tqdm(range(nact)):
        iiy, iix = iy[i], ix[i]
        # poke
        acts[iiy, iix] += h
        lc.update_dm(acts)
        Eup = lc.fwd(wvl).data[mask]
        D = (Eup-E0)/h
        jac[:, i] = D
        # restore
        acts[iiy, iix] -= h

    # final update to make sure last un-poke happens
    lc.update_dm(acts)
    return jac


class EFC:
    """Electric Field Conjugation."""
    def __init__(self, G, c, rcond, loop_gain=0.5):
        """Create a new Electric Field Conjugation instance.

        Parameters
        ----------
        G : numpy.ndarray
            the array returned by one of the jacobian calculating routines
            TODO: the array must be modified by the user to have extra "pixels"
            for the real and imaginary parts, i.e.
            M = G.shape[0]
            N = G.shape[1]
            G2 = np.empty((2*M, N), dtype=float)
            G2[:M] = G.real
            G2[M:] = G.imag
            and give G2 to EFC()
        lc : dygdug.models.LyotCoronagraphSingleDM
            the coronagraph model
        rcond :float
            the cutoff point used in Tikhonov regularization, the log10 of beta
        loop_gain : float
            the gain used.  A gain of one half is equivalent to steepest descent
            while a gain of two is equal to steepest descent with a double step
            algorithm;  see praise/SteepestDescent for similar language and
            Fienup1993 for further discussion

        """
        self.rcond = rcond
        self.G = G
        self.c = c
        self.loop_gain = loop_gain
        # calculate Gstar using Tikhonov regularization
        U, S, Vt = np.linalg.svd(G, full_matrices=False)
        sf = rcond*S.max()
        Sinv = S / (S*S + sf*sf)

        # Gstar is the regularized pseudo-inverse
        # of G
        self.Gstar = (Vt.T * Sinv).dot(U.T)
        M = self.G.shape[0]//2
        self.Ework = np.empty((2*M), dtype=float)
        self.M = M
        self.act_cmds = c.dm.actuators.copy().ravel()

    def step(self):
        """Advance one step."""
        field = self.c.fwd(None)
        E = field.data[self.c.dh_mask]
        M = self.M
        Ew = self.Ework
        Ew[:M] = E.real
        Ew[M:] = E.imag

        y = self.Gstar.dot(Ew)
        self.act_cmds -= self.loop_gain * y
        old_wfe = self.c.dm_wfe
        self.c.update_dm(self.act_cmds)
        return old_wfe, field.intensity
