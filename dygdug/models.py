import inspect
from pathlib import Path
from collections import namedtuple

from astropy.io import fits

from prysm import (
    propagation,
    coordinates,
    polynomials,
    geometry,
    fttools,
)
from prysm.experimental.dm import DM
from prysm._richdata import RichData
from prysm.mathops import np
WF = propagation.Wavefront

DEFAULT_IFN_FN = 'influence_dm5v2.fits'

ImgSamplingSpec = namedtuple('ImageSamplingSpec', ['N', 'lamD', 'px_per_lamD'])


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

    # reuse name now... |= is a  py 3.9+ dictionary union
    # (could use kwarg.update(kwargs) for <=3.9 compat)
    x, y = coordinates.make_xy_grid(N, dx=dx)
    xyrt = dict(x=x, y=y)
    if 'r' in need_args or 't' in need_args:
        r, t = coordinates.cart_to_polar(x, y)  # NOQA: linter blind to the fact that we may access r, t through locals
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


class ThreePlaneSingleDMCoronagraph:
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

    def _fwd_no_coro(self, wvl):
        """For computing normalization only."""
        wf = WF.from_amp_and_phase(self.pupil.data, phase=None, wavelength=wvl, dx=self.pupil.dx)
        img = wf.focus_fixed_sampling(self.efl,
                                      dx=self.imgspec.lamD/self.imgspec.px_per_lamD,
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
        wf = WF.from_amp_and_phase(self.pupil.data, phase=self.dm_wfe, wavelength=wvl, dx=self.pupil.dx)
        after_lyot, at_fpm, after_fpm, at_lyot = \
            wf.babinet(efl=self.efl, lyot=self.ls.data, fpm=1 - fpm, fpm_dx=self.fpm.dx,
                       method=self.method, return_more=True)

        img = after_lyot.focus_fixed_sampling(self.efl,
            dx=self.imgspec.lamD/self.imgspec.px_per_lamD,
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
        if isinstance(wvls, float):
            # monochromatic
            self.norm = 1 / np.sqrt(self._fwd_no_coro(wvls).data.max())
            return

        stack = np.array([self._fwd_no_coro(w).data for w in wvls])
        psf = polynomials.sum_of_2d_modes(stack, weights)
        self.norm = 1 / np.sqrt(psf.max())
        # self.norm = 1 / (len(wvls)*np.sqrt(self._fwd_no_coro(.550).data.max()))
        return


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
    dx = iss.lamD / iss.px_per_lamD
    x, y = coordinates.make_xy_grid(iss.N, dx=dx)
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
    return RichData(I, dx=1/model.imgspec.px_per_lamD, wavelength=None)


def incoh_sum(es):
    """Incoherent sum of E-fields."""
    Is = [abs(e)**2 for e in es]
    return sum(Is)
