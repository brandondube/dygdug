from tqdm import tqdm

from prysm.mathops import np
from prysm.conf import config


def calculate_jacobian_fwddiff(query_actuators, set_actuators, current_E_fields, h=0.1):
    """Calculate the Jacobian via forward differences.

    As opposed to central differences

    Parameters
    ----------
    query_actuators : callable
        a function that when called returns all actuators in a system,
        packed into a single vector
    set_actuators : callable
        a function that when called with a vector of all actuators in a system,
        updates the state of the system
    current_E_fields : callable
        a function which returns one or more (mxn) shaped E-fields, where
        'one or more' refers to discrete wavelengths or sub-bands
    h : float
        step size in forward differences, same units as those expected by
        set_actuators e.g. nanometers of surface figure, or voltage

    Returns
    -------
    numpy.ndarray
        nwvl x (mpix x npix) x nact forward sensivity matrix

    """
    acts = query_actuators().copy()
    nact = len(acts)
    # dearest reader, you will see np.array called on E-fields in this function
    # this is because the user may return a list or tuple of E-fields, which
    # we can't get the shape of, and do not nicely assign into a block of another
    # array for all numpy-like libraries.  If the user gives us an array, np.array()
    # does nothing.  If they give us a list or a tuple, it costs almost nothing
    # because numpy will make a non-contiguous array and it will Just Work TM
    E0 = np.array(current_E_fields())
    jac = np.empty((*E0.shape, nact), dtype=complex)
    for i in tqdm(range(nact)):
        # poke
        acts[i] += h
        set_actuators(acts)
        Eup = np.array(current_E_fields())
        D = (Eup-E0)/h
        jac[..., i] = D
        # restore
        acts[i] -= h

    # final update to make sure last un-poke happens
    set_actuators(acts)
    return jac


class EFC:
    """Electric Field Conjugation."""
    def __init__(self, G, beta, dh_mask,
                 query_actuators, set_actuators, current_E_fields,
                 spectral_weights=None, loop_gain=0.5):
        """Create a new Electric Field Conjugation instance.

        Parameters
        ----------
        G : numpy.ndarray
            (NxM)xK array, where (NxM) is the image area, without mask, and K
            is the total number of actuators
        beta : float
            log10 of the corner of the Tikhonov window; typically values of
            -1.5 to -2.5 result in "relaxed" control that is stable,
            while results in the regime of -5 or smaller result in hyper
            aggressive control that is likely divergent if sustained
            use regularize_control_matrix(new_beta) to perform beta kicking
            and similar operations
        dh_mask : numpy.ndarray
            mask which selects and optionally weights the dark hole;
            pixels which are of value zero are excluded from control
        query_actuators : callable
            a function that when called returns all actuators in a system,
            packed into a single vector
        set_actuators : callable
            a function that when called with a vector of all actuators in a system,
            updates the state of the system
        current_E_fields : callable
            a function which returns one or more (mxn) shaped E-fields, where
            'one or more' refers to discrete wavelengths or sub-bands
        spectral_weights : numpy.ndarray
            weights applied to each control sub-band, default is no weighting
        loop_gain : float
            the gain used.  A gain of one half is equivalent to steepest descent
            while a gain of two is equal to steepest descent with a double step
            algorithm;  see praise/SteepestDescent for similar language and
            Fienup1993 for further discussion

        """
        self.Graw = G
        self.beta = beta
        self.dh_mask = dh_mask
        if dh_mask.dtype == bool:
            self.binary_dh_mask = dh_mask
            self.need_spatial_weighting = False
        else:
            self.binary_dh_mask = dh_mask != 0
            self.need_spatial_weighting = True

        self.spectral_weights = spectral_weights  # TODO: implement this...

        # the three functions that let EFC interact with the world
        self.query_actuators = query_actuators
        self.set_actuators = set_actuators
        self.current_E_fields = current_E_fields

        self.loop_gain = loop_gain

        self.modify_jacobian_matrix()
        self.compute_control_matrix()
        self.act_cmds = query_actuators()
        self.iter = 0

    def modify_jacobian_matrix(self):
        # ellipsis; if there is a spectral dimension on the front, skip over it
        # and if not, select all wavelengths
        if self.need_spatial_weighting:
            # ellipsis, :, :, newaxis = elementwise product on spatial axes only
            G = self.Graw * self.dh_mask[..., :, :, np.newaxis]
        else:
            G = self.Graw
        G2 = G[..., self.binary_dh_mask, :]
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
        self.regularize_control_matrix(self.beta)
        return self.Gstar

    def regularize_control_matrix(self, beta):
        self.beta = beta
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
        denom = S1*S1 * 10 ** beta + S*S
        Sinv = S / denom

        # Gstar is the regularized pseudo-inverse
        # of G
        self.Gstar = (Vt.T * Sinv).dot(U.T)
        return self.Gstar

    def step(self):
        """Advance one step."""
        field = np.array(self.current_E_fields())
        if self.need_spatial_weighting:
            # ellipsis, :, :, newaxis = elementwise product on spatial axes only
            E = field * self.dh_mask[..., :, :]
        else:
            E = field
        E = field[..., self.binary_dh_mask].ravel()
        M = self.M
        Ew = self.Ework
        Ew[:M] = E.real
        Ew[M:] = E.imag

        y = self.Gstar.dot(Ew)
        self.act_cmds -= self.loop_gain * y
        self.set_actuators(self.act_cmds)
        self.iter += 1
        return field


class EFC2:
    """Electric Field Conjugation."""
    def __init__(self, G, c, beta, dh_mask, wvls, weights, loop_gain=0.5):
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
        wvls : float or numpy.ndarray
            wavelengths of light
        weights : float or numpy.ndarray
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
