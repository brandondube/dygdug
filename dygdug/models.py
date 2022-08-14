from pathlib import Path

from astropy.io import fits

from prysm import (
    propagation,
    coordinates,
    geometry,
    fttools,
)
from prysm.experimental.dm import DM
from prysm.mathops import np
WF = propagation.Wavefront

DEFAULT_IFN_FN = 'influence_dm5v2.fits'


class LyotCoronagraphSingleDM:
    def __init__(self, Nmodel, Npup, Nlyot, Nfpm, Nimg, Dpup, fpm_oversampling, image_oversampling, rFPM, wvl0, fno, data_root, ifn_fn, iwa, owa, start_az, end_az):
        self.Nmodel = Nmodel
        self.Npup = Npup
        self.Nlyot = Nlyot
        self.Nfpm = Nfpm
        self.Dpup = Dpup
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
        nact = 32
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
