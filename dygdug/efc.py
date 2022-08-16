from tqdm import tqdm

from prysm.mathops import np
from prysm.conf import config

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
    # calculate resting E-field

    # allocate the memory for the forward differences
    E0 = lc.fwd(wvl).data
    jac = np.empty((*E0.shape, nact), dtype=complex)
    for i in tqdm(range(nact)):
        iiy, iix = iy[i], ix[i]
        # poke
        acts[iiy, iix] += h
        lc.update_dm(acts)
        Eup = lc.fwd(wvl).data
        D = (Eup-E0)/h
        jac[..., i] = D
        # restore
        acts[iiy, iix] -= h

    # final update to make sure last un-poke happens
    lc.update_dm(acts)
    return jac


class EFC:
    """Electric Field Conjugation."""
    def __init__(self, G, c, beta, dh_mask, loop_gain=0.5):
        """Create a new Electric Field Conjugation instance.

        Parameters
        ----------
        G : numpy.ndarray
            (NxM)xK array, where (NxM) is the image area, without mask, and K
            is the total number of actuators
        c : dygdug.models.LyotCoronagraphSingleDM
            the coronagraph model
        rcond :float
            the cutoff point used in Tikhonov regularization, the log10 of beta
        dh_mask : numpy.ndarray
            mask which selects and (TODO) optionally weights the dark hole
        loop_gain : float
            the gain used.  A gain of one half is equivalent to steepest descent
            while a gain of two is equal to steepest descent with a double step
            algorithm;  see praise/SteepestDescent for similar language and
            Fienup1993 for further discussion

        """
        self.Graw = G
        self.beta = beta
        self.c = c
        self.dh_mask = dh_mask
        self.loop_gain = loop_gain

        self.modify_jacobian_matrix()
        self.compute_control_matrix()
        self.act_cmds = c.dm.actuators.ravel().copy()
        self.iter = 0

    def modify_jacobian_matrix(self):
        # ellipsis; if there is a spectral dimension on the front, skip over it
        # and if not, select all wavelengths
        G2 = self.Graw[..., self.dh_mask, :]
        # collapse all spatial and spectral dimensions, making an MxN matrix
        # where M ~= number of pixels and N = number of actuators
        G2 = G2.reshape((-1, G2.shape[-1]))
        M, N = G2.shape
        self.M = M
        self.N = N
        # now break out real and imag
        G3 = np.empty((2*M, N), dtype=config.precision)
        G3[:M] = G2.real
        G3[M:] = G2.imag
        self.Gmod = G3
        self.Ework = np.zeros(2*M, dtype=config.precision)
        return G3

    def compute_control_matrix(self):
        self.U, self.S, self.Vt = np.linalg.svd(self.Gmod, full_matrices=False)
        self.regularize_control_matrix()
        return self.Gstar

    def regularize_control_matrix(self):
        U, S, Vt = self.U, self.S, self.Vt
        # DOI 10.1117/12.2274687
        # Eq. 4
        # delta h = V * diag ( ... ) * U^* * E
        # ... =
        #          s_i^2               1
        # ------------------------ *  ---
        # s_i^2 + s_1^2 * 10^\beta    s_i
        #
        # =
        #           s_i
        # ------------------------
        # s_1^2 * 10^\beta + s_i^2
        S1 = S[0]
        denom = S1*S1 * 10 ** self.beta + S*S
        Sinv = S / denom

        # Gstar is the regularized pseudo-inverse
        # of G
        self.Gstar = (Vt.T * Sinv).dot(U.T)
        return self.Gstar

    def step(self):
        """Advance one step."""
        field = self.c.fwd(None)
        E = field.data[self.dh_mask]
        M = self.M
        Ew = self.Ework
        Ew[:M] = E.real
        Ew[M:] = E.imag

        y = self.Gstar.dot(Ew)
        self.act_cmds -= self.loop_gain * y
        old_wfe = self.c.dm_wfe
        self.c.update_dm(self.act_cmds)
        self.iter += 1
        return old_wfe, field.intensity
