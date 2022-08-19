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
        self.dm.update(actuators)
        self.dm_wfe = self.dm.render(wfe=True)
        return [self.dm_wfe]

    def current_actuators(self):
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
        # TODO logic about adjusting the norm?  As-is user has to set_norm() first,
        # seems generally OK even if it's a sharp edge?

        # why does radiometry always make me feel autistic......................
        # know the scale factor so that incoherent sum over fwd(w)*weight

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
    model.set_norm(wvl, 1)

    def paste():
        return model.fwd(wvl, norm=True)

    return paste

def curry_bb(model, wvls, weights):
    model.set_norm(wvls, weights)

    def chutney():
        return model.fwd_bb(wvls, weights)

    return chutney


def one_sided_annulus(iss, iwa, owa, azmin, azmax):
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
        I = incoh_sum(field)
    else:
        I = abs(field)**2
    # diameter = model.imgspec.N / model.imgspec.px_per_lamD
    return RichData(I, dx=1/model.imgspec.px_per_lamD, wavelength=None)


def incoh_sum(es):
    Is = [abs(e)**2 for e in es]
    return sum(Is)


class LyotCoronagraphSingleDM:
    def __init__(self, Nmodel, Npup, Nlyot, Nfpm, Nimg, Dpup, Nact, fpm_oversampling, image_oversampling, rFPM, wvl0, fno, data_root, ifn_fn, iwa, owa, start_az, end_az):
        self.Nmodel = Nmodel
        self.Npup = Npup
        self.Nlyot = Nlyot
        self.Nfpm = Nfpm
        self.Dpup = Dpup
        self.Nact = Nact
        self.fpm_oversampling = fpm_oversampling
        self.rFPM = rFPM
        self.wvl0 = wvl0
        self.fno = fno
        self.f = Dpup * fno
        self.Nimg = Nimg
        self.image_oversampling = image_oversampling

        self.iwa = iwa
        self.owa = owa
        self.start_az = start_az
        self.end_az = end_az

        self.dx = Dpup / Npup
        lamD = wvl0 * fno
        fpm_dx = lamD / fpm_oversampling
        self.lamD = lamD
        self.fpm_dx = fpm_dx

        img_dx = lamD / image_oversampling
        self.img_dx = img_dx

        self.data_root = data_root
        self.ifn_fn = ifn_fn
        self.dm = None

        # p = pupil
        self.xp  = None
        self.yp  = None
        self.rp  = None
        self.tp  = None
        self.rpn = None
        # f = fpm
        self.xf  = None
        self.yf  = None
        self.rf = None
        self.tf = None
        self.rfn = None

        # i = image
        self.xi = None
        self.yi = None
        self.ri = None
        self.ti = None
        self.rin = None

        self.lyot = None
        self.pu = None
        self.fpm = None
        self.fpm_babinet = None

        self.dh_mask = None

        self.dm_wfe = None

        self.setup()

    def setup_dm(self, data_root, ifn_fn=DEFAULT_IFN_FN):
        dm_act_pitch = 0.9906  # mm
        ifn_sampling_factor = 10  # dimensionless
        ifn_pitch = dm_act_pitch / ifn_sampling_factor  # mm
        dm_angle_deg = (0, 0, 0)
        nact = self.Nact
        dm_diam_mm = dm_act_pitch * nact
        dm_diam_px = dm_diam_mm * ifn_sampling_factor
        dm_model_res = int(dm_diam_px + 4 * ifn_sampling_factor)

        ifn = fits.getdata(data_root/ifn_fn).squeeze()
        ifn = fttools.pad2d(ifn, out_shape=dm_model_res)
        mag = ifn_pitch / self.dx
        dm1 = DM(ifn, Nact=nact, sep=ifn_sampling_factor, rot=dm_angle_deg, upsample=mag)
        self.dm = dm1

    def setup(self):
        self.setup_dm(self.data_root, self.ifn_fn)

        # pupil
        x, y = coordinates.make_xy_grid(self.Nmodel, dx=1)
        r, t = coordinates.cart_to_polar(x, y)
        self.xp = x
        self.yp = y
        self.rp = r
        self.tp = r

        pu = geometry.circle(self.Npup/2, r)
        pu = pu / np.sqrt(pu.sum())
        self.pu = pu

        # lyot
        lyot = geometry.circle(self.Nlyot/2, r).astype(float)
        self.lyot = lyot

        # fpm
        x2, y2 = coordinates.make_xy_grid(self.Nfpm, dx=self.fpm_dx)
        r2, t2 = coordinates.cart_to_polar(x2, y2)
        fpm_babinet = geometry.circle(self.rFPM*self.lamD, r2)
        fpm = 1 - fpm_babinet
        self.xf = x2
        self.yf = y2
        self.rf = r2
        self.tf = t2
        self.fpm = fpm
        self.fpm_babinet = fpm_babinet
        self.norm_ = self.norm(None)


        x3, y3 = coordinates.make_xy_grid(self.Nimg, dx=self.img_dx)
        r3, t3 = coordinates.cart_to_polar(x3, y3)
        iwa = self.iwa * self.lamD
        owa = self.owa * self.lamD
        rs = np.radians(self.start_az)
        re = np.radians(self.end_az)
        self.xi = x3
        self.yi = y3
        self.ri = r3
        self.ti = t3
        mask1 = geometry.circle(iwa, r3)
        mask2 = geometry.circle(owa, r3)
        mask3 = t3 > rs
        mask4 = t3 < re
        self.dh_mask = (mask2 ^ mask1) & (mask3 & mask4)
        return

    def update_dm(self, actuators):
        self.dm.actuators[:] = actuators.reshape(self.dm.actuators.shape)[:]
        self.dm_wfe = fttools.pad2d(self.dm.render(wfe=True), out_shape=self.Nmodel)
        return

    def norm(self, wvl):
        Ea = WF.from_amp_and_phase(self.pu, self.dm_wfe, self.wvl0, self.dx)
        focused_no_coronagraph = Ea.focus_fixed_sampling(self.f, self.img_dx, self.Nimg)
        return 1/np.sqrt(focused_no_coronagraph.intensity.data.max())

    def fwd(self, wvl):
        Ea = WF.from_amp_and_phase(self.pu, self.dm_wfe, self.wvl0, self.dx)
        field_after_lyot = Ea.babinet(self.f, self.lyot, self.fpm_babinet, self.fpm_dx)
        out = field_after_lyot.focus_fixed_sampling(self.f, self.img_dx, self.Nimg)
        out.data *= self.norm_
        out.dx /= self.lamD
        return out


data_root = Path('~/Downloads').expanduser()
_tmp_lc_kwargs = dict(
    Nmodel=512,
    Npup=300,
    Nlyot = 300 * 0.8,
    Nfpm=128,
    Nimg=256,
    Dpup = 30,
    fpm_oversampling=8,
    image_oversampling=8,
    rFPM=2.7,
    wvl0=.550,
    fno=40,
    data_root=data_root,
    ifn_fn=DEFAULT_IFN_FN,
    iwa=3.5,
    owa=10,
    start_az=-90,
    end_az=90
)
