"""Specialized propagations for optical phase vortices."""
from prysm import (
    geometry,
    coordinates,
    propagation
)
from prysm.mathops import np


def compute_partitions_fixed_samplings(n_in, dx_in, efl, wavelength,
                                       partitions='auto', divisor=4, zoomincr=2,
                                       padsamples=1):
    """Compute the partitions for an FFT+MFT hybrid propagation.

    Parameters
    ----------
    n_in : int
        the number of samples across the input pupil
    dx_in : float
        the inter-sample spacing in the input pupil
    efl : float
        focal length to propagate to the focus, mm
    wavelength : float
        the wavelength of light, microns
    partitions : int
        the number of successive zoomed partitions to use when performing
        the zoomed propagations.  Auto selects this number such that the
        innermost partition samples only the core of the airy disk, a region in
        which there are no more sign inversions in the real part of the nominal
        E-field, and beyond which no further partitioning will bring benefit
    divisor : int
        the fraction of the *diameter* of an array to use as the region
        for the next zoom.  The default value of 4 keeps half the *radius*, and
        is analagous to Q=2 propagations in terms of padding/cropping
    zoomincr : float
        ratio of sampling between successive zooms.  Should be chosen in concert
        with divisor, so that the required window size in the next zoom does
        not grow exponentially.  divisor=4, zoomincr=2 (defaults) do not
        require modification in almost all cases
    padsamples : int
        the number of additional samples of zero padding used in each window,
        to guarantee that the partition's boundary exists
        (protects against rounding errors)

    Returns
    -------
    float
        FL/D
    float
        the boundary between the FFT and first fixed sampling window
    list
        (n, res, outer partition, [inner partition]) for each fixed sampling
        propagation; to be passed via *args to make_zoomed_mask

    """
    # compute the region for the FFT
    # we keep Nfft/divisor samples, then MFT the interior of that separately
    natural_res = propagation.pupil_sample_to_psf_sample(dx_in, n_in, wavelength, efl)

    fft_divisor = divisor
    n = n_in  # = fft_n, since we do a Q=1 FFT propagation
    partition_samples = n // fft_divisor
    partition = partition_samples * natural_res

    if partitions == 'auto':
        # choose number of partitions such that final partition is 0.5 L/D in diameter
        # just do the loop instead of trying to do this analytically
        partitions = 0
        terminal_fov = natural_res
        res = natural_res
        outer_partition = partition

        while True:
            res = res / zoomincr
            n = int((outer_partition + padsamples*res) / res * 2)
            fov = n*res
            inner_partition = (n // divisor) * res
            partitions += 1
            outer_partition = inner_partition
            if fov < terminal_fov:
                del inner_partition
                del outer_partition
                break

    outer_partition = partition
    res = natural_res
    n_res_parts = []
    for i in range(partitions):
        # basic loop is to keep zooming
        # now figure out the region for the MFTs;
        # zoom in by zoomincr, then calculate the required number of samples
        # so that we can see the outer boundary
        res = res / zoomincr
        n = int((outer_partition + padsamples*res) / res * 2)
        inner_partition = (n // divisor) * res
        if i == (partitions-1):
            # final partition, no inner cut
            n_res_parts.append((n, res, outer_partition))
        else:
            # non-final partition, include inner cut
            n_res_parts.append((n, res, outer_partition, inner_partition))

        outer_partition = inner_partition

    return natural_res, partition, n_res_parts


def make_fft_mask(charge, n, dx, seam):
    """Make the masked vortex mask for the outermost FFT propagation.

    Parameters
    ----------
    charge : int
        vortex charge (even numbers are better than odd)
    n : int
        number of samples per side of the nxn array
    dx : float
        inter-sample spacing for the FFT propagation (FL/D)
    seam : float
        the coordinate at which the seam between FFT and first zoomed propagation
        exists

    Returns
    -------
    numpy.ndarray
        complex mask containing a vortex and high-pass window

    """
    x, y = coordinates.make_xy_grid(n, dx=dx, grid=False)
    r, t = coordinates.cart_to_polar(x, y)
    # mask = ~geometry.rectangle(seam, x, y)
    mask = ~geometry.circle(seam, r)
    # mask =  1 - geometry.truecircle(seam, r)

    vortex = np.exp(1j*charge*t)
    return vortex * mask


def make_zoomed_mask(charge, n, dx, seam_outer, seam_inner=None):
    """Make the masked vortex mask for zoomed propagations.

    Parameters
    ----------
    charge : int
        vortex charge (even numbers are better than odd)
    n : int
        number of samples per side of the nxn array
    dx : float
        inter-sample spacing for the FFT propagation (FL/D)
    seam_outer : float
        the coordinate at which the prior propagation hands off to this one
    seam_inner : float, optional
        the coordinate at which this propagation hands off to the next
        (None in the case where this is the final propagation)

    Returns
    -------
    numpy.ndarray
        complex mask containing a vortex and high-pass window

    """
    x, y = coordinates.make_xy_grid(n, dx=dx, grid=False)
    r, t = coordinates.cart_to_polar(x, y)

    # mask = geometry.rectangle(seam_outer, x, y)
    mask = geometry.circle(seam_outer, r)
    # mask = geometry.truecircle(seam_outer, r)
    if seam_inner is not None:
        # mask ^= geometry.rectangle(seam_inner, x, y)
        mask ^= geometry.circle(seam_inner, r)
        # mask -= geometry.truecircle(seam_inner, r)

    vortex = np.exp(1j*charge*t)
    return vortex * mask


def vortex_prop(input_field, efl, fft_mask, mft_list, mft_masks, dbg=False, meth='mdft'):
    """Propagate the input field through a phase vortex.

    Parameters
    ----------
    input_field : prysm.propagation.Wavefront
        the input complex E-field
    efl : float
        focal length between the pupil and FPM, mm
    fft_mask : numpy.ndarray
        mask computed with make_fft_mask
    mft_list : list
        the final return from compute_partitions_fixed_sampling
    mft_masks : Iterable
        the sequence of masks computed by [make_zoomed_mask(charge, *e) for e in mft_list]
    dbg : bool, optional
        debug, returns many additional fields (see Returns)
    meth : str
        method of performing zoomed propagations, either mdft or czt

    Returns
    -------
    prysm.propagation.Wavefront
        field at the Lyot plane (pupil after the FPM)
    list of prysm.propagation.Wavefront
        each MFT field at the Lyot plane, if dbg=True
    list of list of prysm.propagation.Wavefront
        (field at FPM, field after FPM), if dbg=True

    """
    inp = input_field
    at_fft = inp.focus(efl, Q=1)
    after_fft = at_fft * fft_mask
    at_lyot_fft = after_fft.unfocus(efl, Q=1)

    mft_fields = []
    debug_fields = []
    for (n, res, *_), mask in zip(mft_list, mft_masks):
        if dbg:
            at_lyot_mft, *dbgf = inp.to_fpm_and_back(efl, mask, res, method=meth, return_more=True)
            debug_fields.append(dbgf)
        else:
            at_lyot_mft = inp.to_fpm_and_back(efl, mask, res, method=meth)

        mft_fields.append(at_lyot_mft.data)

    if dbg:
        return at_lyot_fft, mft_fields, debug_fields

    total_mft_field = sum(mft_fields)
    total_field = at_lyot_fft + total_mft_field
    return total_field
