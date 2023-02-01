import inspect
from collections import namedtuple

from prysm import (
    propagation,
    coordinates,
    polynomials,
    geometry,
)
from prysm.experimental.dm import DM
from prysm._richdata import RichData
from prysm.mathops import np

from .vortexprop import (
    make_fft_mask,
    make_zoomed_mask,
    compute_partitions_fixed_samplings,
    vortex_prop,
)

WF = propagation.Wavefront

DEFAULT_IFN_FN = 'influence_dm5v2.fits'

ImgSamplingSpec = namedtuple('ImageSamplingSpec', ['N', 'lamD', 'px_per_lamD'])


class ImgSamplingSpec:
    """Specification for image plane sampling."""
    def __init__(self, N, dx, lamD):
        self.N = N
        self.dx = dx
        self.lamD = lamD

    @classmethod
    def from_N_lamD_px_per_lamD(cls, N, lamD, px_per_lamD):
        dx = lamD/px_per_lamD
        return cls(N=N, dx=dx, lamD=lamD)


class WavelengthDependentFunctionCache:
    def __init__(self, f):
        self.f = f
        self.storage = {}

    def clear(self):
        self.storage = {}

    def __call__(self, wvl):
        data = self.storage.get(wvl, None)
        if data is None:
            data = self.storage[wvl] = self.f(wvl)

        return data

    def nbytes(self):
        total = 0
        for v in self.storage.values():
            total += v.nbytes

        return total


def _geometry_factory(func, N, dx, **kwargs):
    sig = inspect.signature(func)
    params = set(sig.parameters.keys())
    xyrt = {'x', 'y', 'r', 't'}
    # intersection = filter function parameters to just x,y,r,t so we know what to pass
    need_args = xyrt.intersection(params)

    x, y = coordinates.make_xy_grid(N, dx=dx)
    xyrt = dict(x=x, y=y)
    if 'r' in need_args or 't' in need_args:
        r, t = coordinates.cart_to_polar(x, y)
        xyrt.update(dict(r=r, t=t))
    kwarg = {k: xyrt[k] for k in need_args}
    kwarg.update(kwargs)
    return func(**kwarg)


class Pupil:
    """Representation of a pupil."""
    def __init__(self, data, dx):
        """Create a new Pupil.

        Parameters
        ----------
        data : numpy.ndarray
            2D array, real or complex, containing the pupil
        dx : float
            intersample spacing, mm

        """
        self.data = data
        self.dx = dx
        # TODO: if we store this as an int instead of shape, all the prysm
        # routines will just go back to shape, but if user touches N, maybe
        # they want an int?  But int forces square, and prysm don't care...
        self.N = data.shape

    @classmethod
    def circle(cls, N, Dpup, Npup):
        dx = Dpup/Npup
        data = _geometry_factory(geometry.truecircle, N=N, dx=dx, radius=Dpup/2)
        return cls(data=data, dx=dx)


class FPM:
    """Representation of a focal plane mask."""
    def __init__(self, func_lam, dx):
        self.f = func_lam
        self.dx = dx

    def __call__(self, wvl):
        return self.f(wvl)

    @classmethod
    def lyot(cls, N, lamD, px_per_lamD, radius):
        # lamD has some physical unit (um, say)
        # px per lamD describes our resolution
        dx = lamD / px_per_lamD
        x, y = coordinates.make_xy_grid(N, dx=dx)
        r = np.hypot(x, y)

        def fpmfunc(wvl):
            return 1 - geometry.truecircle(radius*lamD, r)

        # TODO: need to del x, y here?  They should be collected because
        # they went out of scope, but maybe keeping r around through a closure
        # keeps them too?

        # humor
        caller_number_1 = WavelengthDependentFunctionCache(fpmfunc)
        return cls(caller_number_1, dx)

    @classmethod
    def unity(cls, N, lamD, px_per_lamD):
        dx = lamD / px_per_lamD
        ones = np.ones((N, N))

        def fpmfunc(wvl):
            return ones

        return cls(fpmfunc, dx)

    @classmethod
    def vortex(cls, N_pup, dx_pup, efl, charge,
               partitions='auto', divisor=4, zoomincr=2, padsamples=1):
        def fpmfunc(wvl):
            fftres, fftseam, mft_list = \
                compute_partitions_fixed_samplings(N_pup, dx_pup, efl, wvl,
                                                   partitions=partitions,
                                                   divisor=divisor,
                                                   zoomincr=zoomincr,
                                                   padsamples=padsamples)

            fft_mask = make_fft_mask(charge, N_pup, fftres, fftseam)
            mft_masks = [make_zoomed_mask(charge, *elem) for elem in mft_list]
            return fft_mask, mft_list, mft_masks

        caller_number_2 = WavelengthDependentFunctionCache(fpmfunc)
        return cls(caller_number_2, dx=None)


class SingleDMCoronagraph:
    """A single DM, Lyot-style coronagraph."""
    def __init__(self, pupil, dm, fpm, ls, imgspec, efl):
        """Create a new three plane single DM coronagraph model.

        Parameters
        ----------
        pupil : Pupil
            the input pupil
        dm : prysm.experimental.dm.DM
            the deformable mirror
        fpm : FPM
            the focal plane mask
        ls : Pupil
            the Lyot stop
        imgspec : ImgSamplingSpec
            the specification for image plane sampling
        efl : float
            focal length between the pupil and focus, mm

        """
        self.pupil = pupil
        self.dm = dm
        self.fpm = fpm
        self.ls = ls
        self.imgspec = imgspec
        self.efl = efl
        self.dm_wfe = self.dm.render(wfe=True)
        self.norm = 1
        self.method = 'mdft'

    def update_all_dms(self, actuators):
        """Update the actuators of all DMs.

        This coronagraph has but a single DM, however the name is compatible
        with multi-DM layouts.

        Parameters
        ----------
        actuators : numpy.ndarray
            array of actuators, can be 2D or 1D

        Returns
        -------
        sequence of ndarray
            wavefront error created by each DM

        """
        self.dm.update(actuators)
        self.dm_wfe = self.dm.render(wfe=True)
        return [self.dm_wfe]

    def current_actuators(self):
        """Commands for each actuator at this moment.

        Returns
        -------
        numpy.ndarray
            flat vector of actuators

        """
        acts = self.dm.actuators
        if self.dm.mask is not None:
            acts = acts[self.dm.mask]

        return acts.ravel()

    def current_dm_wfes(self):
        return [self.dm_wfe]

    def current_dm_actmaps(self):
        return [self.dm.actuators]

    def _fwd_no_coro(self, wvl):
        """For computing normalization only."""
        wf = WF.from_amp_and_phase(self.pupil.data, phase=None, wavelength=wvl, dx=self.pupil.dx)
        img = wf.focus_fixed_sampling(self.efl,
                                      dx=self.imgspec.dx,
                                      samples=self.imgspec.N,
                                      method=self.method)
        return img.intensity

    def fwd(self, wvl, norm=False, debug=False):
        """Forward model of the coronagraph.

        Parameters
        ----------
        wvl : float
            wavelength of light, microns
        norm : bool, optional
            if True, normalizes the field such that it repesents normalized intensity.
        debug : bool, optional
            if True, returns a dictionary with the input field, image field, as
            well as field before and after the FPM and Lyot stops

        Returns
        -------
        numpy.ndarray
            complex E-field at the image plane

        """
        fpm = self.fpm(wvl)
        wf = WF.from_amp_and_phase(self.pupil.data, phase=self.dm_wfe, wavelength=wvl, dx=self.pupil.dx)
        self._g = wf.data
        after_lyot, at_fpm, after_fpm, at_lyot = \
            wf.babinet(efl=self.efl, lyot=self.ls.data, fpm=fpm, fpm_dx=self.fpm.dx,
                       method=self.method, return_more=True)

        img = after_lyot.focus_fixed_sampling(self.efl,
                                              dx=self.imgspec.dx,
                                              samples=self.imgspec.N,
                                              method=self.method)

        if norm:
            img.data *= self.norm
        if debug:
            return {
                'input': wf,
                'fpm': (at_fpm, after_fpm),
                'lyot': (at_lyot, after_lyot),
                'img': img,
            }
        else:
            return img.data

    def _rev(self, protograd, wvl):
        fpm = self.fpm(wvl)
        step1 = propagation.focus_fixed_sampling_backprop(
            wavefunction=protograd,
            input_dx=self.pupil.dx,
            prop_dist=self.efl,
            wavelength=wvl,
            output_dx=self.imgspec.dx,
            output_samples=self.pupil.N,
            method=self.method)

        # return step1
        step1 = WF(step1, wvl, self.pupil.dx, space='pupil')
        step2 = step1.babinet_backprop(self.efl, self.ls.data, fpm, self.fpm.dx, method=self.method)
        # step2 contains the complex gradient at the DM1 plane,
        # compute the gradient w.r.t. phase at DM1
        # 1e3 = um to nm
        # return step2.data
        # return step2
        df_dphi = (2 * np.pi / wvl / 1e3) * np.imag(step2.data * np.conj(self._g))
        # return df_dphi
        df_dacts = self.dm.render_backprop(df_dphi, wfe=True)
        return df_dacts

    def fwd_bb(self, wvls, weights, debug=False):
        """Broad-band forward model of the coronagraph.

        Parameters
        ----------
        wvls : numpy.ndarray
            wavelengths of light, microns
        weights : numpy.ndarray
            vector of spectral weights; weights are interpreted as scaled for E-field
            not intensity, so units are sqrt(phot) and not phot, for example
        norm : bool, optional
            if True, normalizes the field such that it repesents normalized intensity.
        debug : bool, optional
            if True, returns a dictionary with the input field, image field, as
            well as field before and after the FPM and Lyot stops

        Returns
        -------
        sequence of numpy.ndarray
            complex E-field at the image plane, for each wavelength in order

        """
        if weights is not None:
            weights = np.array(weights) * self.norm
        else:
            weights = [self.norm for _ in range(len(wvls))]
        if debug:
            packets = [self.fwd(w, norm=False, debug=True) for w in wvls]
            for p, w in zip(packets, weights):
                p['img'].data *= w

            return packets
        else:
            fields = [self.fwd(w, norm=False, debug=False) for w in wvls]
            for f, w in zip(fields, weights):
                f.data *= w

            return np.array(fields)

    def set_norm(self, wvls, weights):
        """Set the radiometric normalization so that output E-fields are normalized intensity.

        Parameters
        ----------
        wvls : numpy.ndarray
            wavelengths of light, microns
        weights : numpy.ndarray
            vector of spectral weights; weights are interpreted as scaled for E-field
            not intensity, so units are sqrt(phot) and not phot, for example

        """
        if not hasattr(wvls, '__iter__'):
            # monochromatic
            self.norm = 1 / np.sqrt(self._fwd_no_coro(wvls).data.max())
            return

        stack = np.array([self._fwd_no_coro(w).data for w in wvls])
        psf = polynomials.sum_of_2d_modes(stack, weights)
        self.norm = 1 / np.sqrt(psf.max())
        return


class TwoDMCoronagraph:
    """A two DM, Lyot-style coronagraph."""
    def __init__(self, pupil, dm1, dm2, z_dm1_dm2, fpm, ls, imgspec, efl):
        """Create a new three plane single DM coronagraph model.

        Parameters
        ----------
        pupil : Pupil
            the input pupil
        dm1 : prysm.experimental.dm.DM
            the pupil plane DM
        dm2 : prysm.experimental.dm.DM
            the non pupil plane DM
        z_dm1_dm2 : float
            the distance between DM1 and DM2 in mm
        fpm : FPM
            the focal plane mask
        ls : Pupil
            the Lyot stop
        imgspec : ImgSamplingSpec
            the specification for image plane sampling
        efl : float
            focal length between the pupil and focus, mm

        """
        self.pupil = pupil
        self.dm1 = dm1
        self.dm2 = dm2
        self.z_dm1_dm2 = z_dm1_dm2
        self.fpm = fpm
        self.ls = ls
        self.imgspec = imgspec
        self.efl = efl
        self.dm1_wfe = self.dm1.render(wfe=True)
        self.dm2_wfe = self.dm2.render(wfe=True)

        def fwd_f(wvl):
            return propagation.angular_spectrum_transfer_function(dm1.Nout, wvl, dx=pupil.dx, z=self.z_dm1_dm2)

        def rev_f(wvl):
            return propagation.angular_spectrum_transfer_function(dm2.Nout, wvl, dx=pupil.dx, z=-self.z_dm1_dm2)

        self.dm1_to_dm2_tfs = WavelengthDependentFunctionCache(fwd_f)
        self.dm2_to_dm1_tfs = WavelengthDependentFunctionCache(rev_f)
        self.norm = 1
        self.method = 'mdft'

    def update_all_dms(self, actuators):
        """Update the actuators of all DMs.

        This coronagraph has but a single DM, however the name is compatible
        with multi-DM layouts.

        Parameters
        ----------
        actuators : numpy.ndarray
            array of actuators, can be 2D or 1D

        Returns
        -------
        sequence of ndarray
            wavefront error created by each DM

        """
        if self.dm1.mask is not None:
            N = self.dm1.mask.sum()

        self.dm1.update(actuators[:N])
        self.dm2.update(actuators[N:])
        self.dm1_wfe = self.dm1.render(wfe=True)
        self.dm2_wfe = self.dm2.render(wfe=True)
        return [self.dm1_wfe, self.dm2_wfe]

    def current_actuators(self):
        """Commands for each actuator at this moment.

        Returns
        -------
        numpy.ndarray
            flat vector of actuators

        """
        if self.dm1.mask is not None:
            N = self.dm1.mask.sum()
            adm1 = self.dm1.actuators[self.dm1.mask]
        else:
            N = self.dm1.Nact[0] * self.dm1.Nact[1]
            adm1 = self.dm1.actuators.ravel()

        if self.dm2.mask is not None:
            M = self.dm2.mask.sum()
            adm2 = self.dm2.actuators[self.dm2.mask]
        else:
            M = self.dm2.Nact[0] * self.dm2.Nact[1]
            adm2 = self.dm2.actuators.ravel()

        acts = np.empty(N+M)
        acts[:N] = adm1
        acts[N:] = adm2
        return acts

    def current_dm_wfes(self):
        return [self.dm1_wfe, self.dm2_wfe]

    def current_dm_actmaps(self):
        return [self.dm1.actuators, self.dm2.actuators]

    def _fwd_no_coro(self, wvl):
        """For computing normalization only."""
        wf = WF.from_amp_and_phase(self.pupil.data, phase=None, wavelength=wvl, dx=self.pupil.dx)
        img = wf.focus_fixed_sampling(self.efl,
                                      dx=self.imgspec.dx,
                                      samples=self.imgspec.N,
                                      method=self.method)
        return img.intensity

    def fwd(self, wvl, norm=True, debug=False):
        """Forward model of the coronagraph.

        Parameters
        ----------
        wvl : float
            wavelength of light, microns
        norm : bool, optional
            if True, normalizes the field such that it repesents normalized intensity.
        debug : bool, optional
            if True, returns a dictionary with the input field, image field, as
            well as field before and after the FPM and Lyot stops

        Returns
        -------
        numpy.ndarray
            complex E-field at the image plane

        """
        fpm = self.fpm(wvl)
        tf1_to_2 = self.dm1_to_dm2_tfs(wvl)
        tf2_to_1 = self.dm2_to_dm1_tfs(wvl)

        # propagate pupil+DM1 -> DM2 -> DM1
        wf = WF.from_amp_and_phase(self.pupil.data, phase=self.dm1_wfe, wavelength=wvl, dx=self.pupil.dx)
        wf_at_dm2 = wf.free_space(tf=tf1_to_2)
        dm2 = WF.from_amp_and_phase(1, self.dm2_wfe, wvl, dx=self.pupil.dx)
        wf_after_dm2 = wf_at_dm2 * dm2

        wf_at_intermediate_pupil = wf_after_dm2.free_space(tf=tf2_to_1)

        at_fpm = wf_at_intermediate_pupil.focus(efl=self.efl, Q=1)
        after_fpm = at_fpm * fpm
        at_lyot = after_fpm.unfocus(efl=self.efl, Q=1)
        after_lyot = at_lyot * self.ls.data
        img = after_lyot.focus_fixed_sampling(self.efl,
                                              dx=self.imgspec.dx,
                                              samples=self.imgspec.N,
                                              method=self.method)

        if norm:
            img.data *= self.norm
        if debug:
            return {
                'input': wf,
                'dm2':  (wf_at_dm2, wf_after_dm2),
                'pupil': wf_at_intermediate_pupil,
                'fpm': (at_fpm, after_fpm),
                'lyot': (at_lyot, after_lyot),
                'img': img,
            }
        else:
            return img.data

    def fwd_bb(self, wvls, weights, debug=False):
        """Broad-band forward model of the coronagraph.

        Parameters
        ----------
        wvls : numpy.ndarray
            wavelengths of light, microns
        weights : numpy.ndarray
            vector of spectral weights; weights are interpreted as scaled for E-field
            not intensity, so units are sqrt(phot) and not phot, for example
        norm : bool, optional
            if True, normalizes the field such that it repesents normalized intensity.
        debug : bool, optional
            if True, returns a dictionary with the input field, image field, as
            well as field before and after the FPM and Lyot stops

        Returns
        -------
        sequence of numpy.ndarray
            complex E-field at the image plane, for each wavelength in order

        """
        if weights is not None:
            weights = weights * self.norm
        else:
            weights = [self.norm for _ in range(len(wvls))]
        if debug:
            packets = [self.fwd(w, norm=False, debug=True) for w in wvls]
            for p, w in zip(packets, weights):
                p['img'].data *= w

            return packets
        else:
            fields = [self.fwd(w, norm=False, debug=False) for w in wvls]
            for f, w in zip(fields, weights):
                f.data *= w

            return fields

    def set_norm(self, wvls, weights):
        """Set the radiometric normalization so that output E-fields are normalized intensity.

        Parameters
        ----------
        wvls : numpy.ndarray
            wavelengths of light, microns
        weights : numpy.ndarray
            vector of spectral weights; weights are interpreted as scaled for E-field
            not intensity, so units are sqrt(phot) and not phot, for example

        """
        if not hasattr(wvls, '__iter__'):
            # monochromatic
            self.norm = 1 / np.sqrt(self._fwd_no_coro(wvls).data.max())
            return

        stack = np.array([self._fwd_no_coro(w).data for w in wvls])
        psf = polynomials.sum_of_2d_modes(stack, weights)
        self.norm = 1 / np.sqrt(psf.max())
        return


class SingleDMVortexCoronagraph(SingleDMCoronagraph):
    """A single DM coronagraph, with special vortex propagation through the FPM."""

    def fwd(self, wvl, norm=True, debug=False):
        """Forward model of the coronagraph.

        Parameters
        ----------
        wvl : float
            wavelength of light, microns
        norm : bool, optional
            if True, normalizes the field such that it repesents normalized intensity.
        debug : bool, optional
            if True, returns a dictionary with the input field, image field, as
            well as field before and after the FPM and Lyot stops

        Returns
        -------
        numpy.ndarray
            complex E-field at the image plane

        """
        fft_mask, mft_list, mft_masks = self.fpm(wvl)
        wf = WF.from_amp_and_phase(self.pupil.data, phase=self.dm_wfe, wavelength=wvl, dx=self.pupil.dx)

        at_lyot = vortex_prop(wf, self.efl, fft_mask, mft_list, mft_masks)
        after_lyot = at_lyot * self.ls.data

        img = after_lyot.focus_fixed_sampling(self.efl,
                                              dx=self.imgspec.dx,
                                              samples=self.imgspec.N,
                                              method=self.method)

        img.dx = img.dx / self.imgspec.lamD

        if norm:
            img.data *= self.norm
        if debug:
            return {
                'input': wf,
                'lyot': (at_lyot, after_lyot),
                'img': img,
            }
        else:
            return img.data


def curry_nb(model, wvl):
    """Create a function with no arguments that runs the forward model, monochromatic.

    Parameters
    ----------
    model : model
        the coronagraph model, e.g. ThreePlaneSingleDMCoronagraph
    wvl : float
        wavelength of light, microns

    Returns
    -------
    func()
        that returns model.fwd(wvl)

    """
    model.set_norm(wvl, 1)

    def paste():
        return model.fwd(wvl, norm=True)

    return paste


def curry_bb(model, wvls, weights):
    """Create a function with no arguments that runs the forward model, polychromatic.

    Parameters
    ----------
    model : model
        the coronagraph model, e.g. ThreePlaneSingleDMCoronagraph
    wvls : numpy.ndarray
        wavelengths of light, microns
    weights : numpy.ndarray
        vector of spectral weights; weights are interpreted as scaled for E-field
        not intensity, so units are sqrt(phot) and not phot, for example

    Returns
    -------
    func()
        that returns model.fwd_bb(wvls, weights)

    """
    model.set_norm(wvls, weights)

    def chutney():
        return model.fwd_bb(wvls, weights)

    return chutney


def one_sided_annulus(iss, iwa, owa, azmin, azmax):
    """Create a one sided annular mask.

    Parameters
    ----------
    iss : ImageSamplingSpec
        the image sampling spec
    iwa : float
        inner working angle, lam/D
    owa : float
        outer working angle, lam/D
    azmin : float
        minimum azimuth, degrees
    azmax : float
        maximum azimuth, degrees

    Returns
    -------
    numpy.ndarray
        binary mask

    """
    x, y = coordinates.make_xy_grid(iss.N, dx=iss.dx)
    r, t = coordinates.cart_to_polar(x, y)
    iwa = iwa * iss.lamD
    owa = owa * iss.lamD
    rs = np.radians(azmin)
    re = np.radians(azmax)
    mask1 = geometry.circle(iwa, r)
    mask2 = geometry.circle(owa, r)
    mask3 = t > rs
    mask4 = t < re
    dh_mask = (mask2 ^ mask1) & (mask3 & mask4)
    return dh_mask


def plottable(field, model):
    if isinstance(field, WF):
        return field.intensity
    if isinstance(field, RichData):
        return field

    if field.ndim == 3:
        # spectral cube
        I = incoh_sum(field)  # NOQA
    else:
        I = abs(field)**2  # NOQA
    # diameter = model.imgspec.N / model.imgspec.px_per_lamD
    # px_per_lamD = model.imgspec.lamD / model.imgspec.dx
    return RichData(I, dx=model.imgspec.dx, wavelength=None)


def incoh_sum(es):
    """Incoherent sum of E-fields."""
    Is = [abs(e)**2 for e in es]
    return sum(Is)
