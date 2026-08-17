"""Coronagraph optimizer."""

from prysm import coordinates, geometry
from prysm.mathops import ndimage, np
from prysm.x.optym.governors import (
    AnyGovernor,
    FunctionTolerance,
    GradientTolerance,
    StepRecord,
)

from dygdug.backend import sync_coronagraph
from dygdug.cost_functions import (
    FieldBoxConstraint,
    IntensityConstraint,
    MeanSquaredErrorQuadratic,
)
from dygdug.masks import FPM, Pupil
from dygdug.truecircle_grad import truecircle_fwd


class VariablePupil(Pupil):
    """A Pupil whose active pixels can be updated from a flat parameter vector.

    Constructed identically to :class:`Pupil` (same classmethods: ``circle``,
    ``annular``, ``hexagonal_segmented``), with an additional *mode* argument
    that selects between amplitude and phase apodization:

    ``'amplitude'`` (default)
        ``x`` contains real transmission values.  ``pupil.data`` is stored as
        floating-point data so continuous amplitude updates are not coerced
        back to a binary mask.

    ``'phase'``
        ``x`` contains real phase values φ.  ``pupil.data`` is complex
        (``exp(1j·φ)``), which sets ``Coronagraph.PUPIL_IS_COMPLEX = True``.
    """

    def __init__(self, data, dx, mode="amplitude"):
        if mode not in ("amplitude", "phase"):
            raise ValueError(f"mode must be 'amplitude' or 'phase', got {mode!r}")

        # Capture the active region before any optimization updates.  Some
        # geometries (e.g. CompositeHexagonalAperture.amp) are boolean; keep
        # the mask binary, but cast the stored pupil to a continuous dtype.
        self.mask = data == 1
        self._mask_idx = np.flatnonzero(self.mask.ravel())
        self._n_params = int(self._mask_idx.size)
        if mode == "phase":
            # Cast to complex so Coronagraph.PUPIL_IS_COMPLEX is set correctly.
            data = data.astype(complex)
        else:
            data = data.astype(float)

        super().__init__(data, dx)
        self.mode = mode

    @property
    def n_params(self):
        """Number of optimizable parameters (active pixel count)."""
        return int(self.mask.sum())

    def update(self, x):
        """Scatter parameter vector *x* into the active (mask == True) pixels.

        Parameters
        ----------
        x : array_like
            1-D array of length ``self.mask.sum()``.
            *Amplitude mode*: real transmission values written directly.
            *Phase mode*: real phase values φ; stored as ``exp(1j·φ)``.
        """
        if self.mode == "phase":
            self.data[self.mask] = np.exp(1j * x)
        else:
            self.data[self.mask] = x

    # ------------------------------------------------------------------
    # Classmethods — thin wrappers that add the *mode* parameter
    # ------------------------------------------------------------------
    # The parent classmethods call cls(data=data, dx=dx), which hits
    # VariablePupil.__init__ with mode defaulting to 'amplitude'.  These
    # overrides let callers explicitly request phase mode without needing
    # to construct the geometry themselves.

    @classmethod
    def circle(cls, Dpup, Npup, mode="amplitude"):
        if mode not in ("amplitude", "phase"):
            raise ValueError(f"mode must be 'amplitude' or 'phase', got {mode!r}")
        inst = super().circle(Dpup, Npup)
        if mode == "phase":
            inst.data = inst.data.astype(complex)
            inst.mode = "phase"
        return inst

    @classmethod
    def annular(cls, Dpup, Npup, inner_radius, outer_radius, mode="amplitude"):
        if mode not in ("amplitude", "phase"):
            raise ValueError(f"mode must be 'amplitude' or 'phase', got {mode!r}")
        inst = super().annular(Dpup, Npup, inner_radius, outer_radius)
        if mode == "phase":
            inst.data = inst.data.astype(complex)
            inst.mode = "phase"
        return inst

    @classmethod
    def hexagonal_segmented(
        cls,
        Dpup,
        Npup,
        rings,
        segment_diameter,
        segment_separation,
        segment_angle=90,
        exclude=(),
        mode="amplitude",
    ):
        if mode not in ("amplitude", "phase"):
            raise ValueError(f"mode must be 'amplitude' or 'phase', got {mode!r}")
        inst = super().hexagonal_segmented(
            Dpup,
            Npup,
            rings,
            segment_diameter,
            segment_separation,
            segment_angle=segment_angle,
            exclude=exclude,
        )
        if mode == "phase":
            inst.data = inst.data.astype(complex)
            inst.mode = "phase"
        return inst


class VariableFPM(FPM):
    """A FPM whose active pixels can be updated from a flat parameter vector.

    This class is in progress, and doesnt support wavelength-dependent optimization
    just yet. This is mostly out of a lack of need, but could be relevant for
    designing Hybrid-Lyot Coronagraphs in the future.

    Constructed identically to :class:`FPM` (same classmethods: ``lyot``,
    ``annular``, ``unity``).  On construction the underlying function is
    evaluated once (these masks are all wavelength-independent in practice)
    to snapshot both the mutable :attr:`data` array and the boolean
    :attr:`mask` of active pixels.  :meth:`__call__` then returns
    :attr:`data` directly, bypassing the original function.
    """

    def __init__(self, func_lam, dx):
        super().__init__(func_lam, dx)
        # Evaluate once to get the initial spatial array.
        # None is a safe sentinel: all built-in fpmfunc closures ignore wvl.
        initial = func_lam(None)
        self.data = initial.copy()  # ensure a mutable, owned array
        self.mask = initial == 1

    def __call__(self, wvl):
        # Return the mutable data array; wvl is accepted for API compatibility.
        return self.data

    @property
    def n_params(self):
        """Number of optimizable parameters (active pixel count)."""
        return int(self.mask.sum())

    def update(self, x):
        """Scatter parameter vector *x* into the active (mask == True) pixels.

        Parameters
        ----------
        x : array_like
            1-D array of length ``self.mask.sum()``.  Values are written
            in row-major order into every pixel where the original FPM
            data equalled 1.
        """
        self.data[self.mask] = x


class VariableLyotStop(Pupil):
    """A Lyot stop whose inner/outer radii are the optimizable parameters.

    Unlike :class:`VariablePupil`, the parameter vector passed to
    :meth:`update` does not address individual pixels — it addresses the
    geometric radii of the stop, and :attr:`data` is recomputed from the
    stored radial grid on every call.

    Use :meth:`circular` for a solid disk stop (1 DOF: outer radius) or
    :meth:`annular` for a ring-shaped stop (2 DOFs: inner and outer radius).
    """

    def __init__(self, data, dx, r, annular):
        super().__init__(data, dx)
        self._r = r  # radial coordinate grid, computed once
        self._annular = annular

    @classmethod
    def circular(cls, Dpup, Npup, outer_radius):
        """Circular (solid disk) Lyot stop.

        Parameters
        ----------
        Dpup : float
            Pupil diameter in physical units.
        Npup : int
            Array side length in pixels.
        outer_radius : float
            Initial radius of the stop in physical units.
        """
        dx = Dpup / Npup
        x, y = coordinates.make_xy_grid(Npup, dx=dx)
        r = np.hypot(x, y)
        data = truecircle_fwd(outer_radius, r)[0]
        return cls(data=data, dx=dx, r=r, annular=False)

    @classmethod
    def annular(cls, Dpup, Npup, inner_radius, outer_radius):
        """Annular (ring-shaped) Lyot stop.

        Parameters
        ----------
        Dpup : float
            Pupil diameter in physical units.
        Npup : int
            Array side length in pixels.
        inner_radius : float
            Initial inner radius of the annulus in physical units.
        outer_radius : float
            Initial outer radius of the annulus in physical units.
        """
        dx = Dpup / Npup
        x, y = coordinates.make_xy_grid(Npup, dx=dx)
        r = np.hypot(x, y)
        data = truecircle_fwd(outer_radius, r)[0] * (
            1 - truecircle_fwd(inner_radius, r)[0]
        )
        return cls(data=data, dx=dx, r=r, annular=True)

    @property
    def n_params(self):
        """Number of optimizable parameters (1 for circular, 2 for annular)."""
        return 2 if self._annular else 1

    def update(self, x):
        """Recompute the Lyot stop from new radius parameters.

        Parameters
        ----------
        x : array_like
            For a circular stop: ``[outer_radius]``.
            For an annular stop: ``[inner_radius, outer_radius]``.
        """
        if self._annular:
            inner_r, outer_r = x
            self.data = truecircle_fwd(outer_r, self._r)[0] * (
                1 - truecircle_fwd(inner_r, self._r)[0]
            )
        else:
            self.data = truecircle_fwd(x[0], self._r)[0]


class ThroughputOptimizer:
    def __init__(self, coro, alpha=1.0):
        """Coronagraph optimizer that sums the values in the apodizer
        only meaningful for Amplitude-apodized coronagraphs as a proxy
        for core throughput, which is an application of CoronagraphOptimizer

        Parameters
        ----------
        coro: `dygdug.models.Coronagraph`
            Coronagraph model that contains a VariablePupil
        alpha: float
            optimization weight to multiply the throughput by
        """

        # --- Discover variable elements in pupil → fpm → lyot_stop order ---
        self._variable_elems = []  # [(attr_name, element), ...]
        self._slices = []  # [slice into flat x, ...]
        offset = 0
        self.alpha = alpha
        self.coro = coro

        # One-time backend sync so a GPU pupil/Lyot/executor stay on-device.
        sync_coronagraph(coro)

        # Scan for pupil
        for attr in ("pupil", "fpm", "lyot_stop"):
            elem = getattr(coro, attr)
            if hasattr(elem, "n_params"):
                n = elem.n_params
                self._variable_elems.append((attr, elem))
                self._slices.append(slice(offset, offset + n))
                offset += n
        self.n_params = offset

    def _push(self, x):
        """Distribute *x* slices to their respective variable elements."""
        for sl, (_, elem) in zip(self._slices, self._variable_elems):
            elem.update(x[sl])

    def fg(self, x):
        """Evaluate cost *J* and gradient *dJ/dx* for parameter vector *x*."""
        self._push(x)

        # simply sum the values in the pupil
        f = -1 * np.sum(self.coro.pupil.data) * self.alpha
        g = -1 * self.coro.pupil.data[self.coro.pupil.mask == 1] * self.alpha

        return f, g


class CoronagraphOptimizer:
    """Coronagraph optimizer with automatic variable-element discovery.

    On construction, ``coro.pupil``, ``coro.fpm``, and ``coro.lyot_stop``
    are inspected for objects that expose an ``n_params`` attribute
    (:class:`VariablePupil`, :class:`VariableFPM`, :class:`VariableLyotStop`).
    A contiguous flat parameter layout is built from those elements in
    ``pupil → fpm → lyot_stop`` order so that :meth:`fg` can distribute
    a single vector *x* to the right place.

    Parameters
    ----------
    dark_hole : ndarray of bool
        2-D boolean mask selecting the dark-hole pixels in the final image
        plane.
    coro : Coronagraph
        Coronagraph model to optimise.
    wvl : float or sequence of float
        Wavelength(s) to include.  Cost and gradient are summed over all
        wavelengths.
    cost : cost-function class or instance, optional
        Must expose ``.forward`` → scalar and ``.reverse`` → array.  The
        ``domain`` class attribute (see :mod:`dygdug.cost_functions`) selects
        what the cost consumes:

        ``"intensity"`` (default)
            ``forward(I)`` / ``reverse(I)`` operate on the dark-hole intensity
            ``I = |E|^2``; ``reverse`` returns ``dJ/dI``.

        ``"field"``
            ``forward(E)`` / ``reverse(E)`` operate on the complex dark-hole
            field, so the real and imaginary parts can be constrained
            independently (e.g. :class:`FieldMeanSquaredError`).  ``reverse``
            returns the Wirtinger seed ``dJ/dE*``.

        Defaults to :class:`MeanSquaredErrorQuadratic`.
    """

    def __init__(self, dark_hole, coro, wvl, cost=MeanSquaredErrorQuadratic):
        # Accept 0/1 mask arrays, but always index with a boolean mask.
        # Integer masks would otherwise trigger NumPy advanced indexing.
        self.dh = np.asarray(dark_hole).astype(bool)
        self.coro = coro

        # Accept a scalar or a sequence of wavelengths.
        self.wvl = [wvl] if np.ndim(wvl) == 0 else list(wvl)

        # One-time backend sync (see AugmentedLagrangian): keep the inner
        # forward/reverse loop entirely on the active prysm backend.  No-op on CPU.
        sync_coronagraph(coro, self.wvl)

        # Accept the cost-function class or an already-constructed instance.
        self.cost_fn = cost() if isinstance(cost, type) else cost
        # Domain decides whether the cost consumes intensity |E|^2 or the
        # complex field E directly (see dygdug.cost_functions).  Unmarked costs
        # are treated as intensity for backward compatibility.
        self.cost_domain = getattr(self.cost_fn, "domain", "intensity")
        if self.cost_domain not in ("intensity", "field"):
            raise ValueError(
                f"cost domain must be 'intensity' or 'field', got {self.cost_domain!r}"
            )

        # --- Discover variable elements in pupil → fpm → lyot_stop order ---
        self._variable_elems = []  # [(attr_name, element), ...]
        self._slices = []  # [slice into flat x, ...]
        offset = 0

        # coro is our forward mode
        for attr in ("pupil", "fpm", "lyot_stop"):
            elem = getattr(coro, attr)
            if hasattr(elem, "n_params"):
                n = elem.n_params
                self._variable_elems.append((attr, elem))
                self._slices.append(slice(offset, offset + n))
                offset += n
        self.n_params = offset

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fg(self, x):
        """Evaluate cost *J* and gradient *dJ/dx* for parameter vector *x*.

        Variable elements are updated first, then the coronagraph is
        propagated forward.  Gradients are computed analytically for
        :class:`VariablePupil` via the adjoint model, and with central
        finite differences for :class:`VariableLyotStop` (1–2 DOFs, cheap).
        :class:`VariableFPM` gradient is not yet implemented.

        Parameters
        ----------
        x : array_like, shape ``(n_params,)``
            Flat parameter vector.  Layout is ``pupil | fpm | lyot_stop``
            (only present variable elements are included).

        Returns
        -------
        J : float
            Scalar cost summed over all wavelengths.
        grad : ndarray, shape ``(n_params,)``
            Gradient of *J* w.r.t. *x*.
        """

        # Updates all optimizeable optics
        self._push(x)

        # Accumulates cost and gradient over all wavelengths
        J = 0.0
        grad = np.zeros(self.n_params)

        for wvl in self.wvl:
            E_focal = self.coro.forward(wvl, include_fpm=True)
            E_dh = E_focal[self.dh]
            Ebar = np.zeros_like(E_focal)

            if self.cost_domain == "field":
                # Field-domain cost: forward/reverse consume the complex field
                # directly and reverse returns the Wirtinger seed dJ/dE*.
                J += float(self.cost_fn.forward(E_dh))
                Ebar[self.dh] = self.cost_fn.reverse(E_dh)
            else:
                # Intensity-domain cost on the dark-hole intensity I = |E|^2.
                # Wirtinger gradient at focal plane: dJ/dE* = (dJ/dI) · E.
                I_dh = np.abs(E_dh) ** 2
                J += float(self.cost_fn.forward(I_dh))
                Ebar[self.dh] = self.cost_fn.reverse(I_dh) * E_dh

            # Back-propagate gradient to pupil plane (return value is 2D;
            # relevant slices are extracted below by the per-element helpers).
            self.coro.reverse(Ebar, wvl, include_fpm=True)

            for (attr, elem), sl in zip(self._variable_elems, self._slices):
                if isinstance(elem, VariablePupil):
                    grad[sl] += self._pupil_grad(elem)
                # VariableFPM: TODO (needs intermediate FPM-plane field)
                # VariableLyotStop: handled below via finite differences until
                # analytical gradient is available

        # Finite-difference gradient for VariableLyotStop.
        # Done outside the wavelength loop so it samples the full summed cost.
        for (attr, elem), sl in zip(self._variable_elems, self._slices):
            if isinstance(elem, VariableLyotStop):
                grad[sl] = self._fd_grad(x, sl)

        return J, grad

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------

    def _push(self, x):
        """Distribute *x* slices to their respective variable elements."""
        for (attr, elem), sl in zip(self._variable_elems, self._slices):
            elem.update(x[sl])

    def _pupil_grad(self, elem):
        """Analytical gradient w.r.t. the real optimisation variables of a
        :class:`VariablePupil`, dispatched on ``elem.mode``.

        ``coro.reverse`` stores the Wirtinger adjoint
        ``adj = dJ/d(pupil.data)*``.  Because the optimisation variables are
        real, the directional derivative is
        ``dJ = 2 Re(conj(adj) · d(pupil.data))``.

        *Amplitude mode* — ``x`` are real transmissions, so
            ``d(pupil.data)/dx = 1`` and ``dJ/dx = 2 Re(adj)``.

        *Phase mode* — ``x`` are real phases φ, ``pupil.data = exp(iφ)``, so
            ``d(pupil.data)/dφ = i exp(iφ)`` and
            ``dJ/dφ = 2 Im(adj · conj(pupil.data))``.
        """
        adj = self.coro.adjoint_at_entrance_pupil[elem.mask]
        if elem.mode == "phase":
            field = elem.data[elem.mask]
            return 2 * np.imag(adj * field.conj())
        else:
            return 2 * np.real(adj)

    def _total_cost(self, x):
        """Forward-only cost evaluation used by the finite-difference helper."""
        self._push(x)
        J = 0.0
        for wvl in self.wvl:
            E_dh = self.coro.forward(wvl, include_fpm=True)[self.dh]
            if self.cost_domain == "field":
                J += float(self.cost_fn.forward(E_dh))
            else:
                J += float(self.cost_fn.forward(np.abs(E_dh) ** 2))
        return J

    def _fd_grad(self, x, sl, eps=1e-7):
        """Central-difference gradient for the parameters in slice *sl*.

        Only the elements in *sl* are perturbed; all variable elements
        are restored to *x* before returning.
        """
        g = np.empty(sl.stop - sl.start)
        x_work = x.copy()
        for i, idx in enumerate(range(sl.start, sl.stop)):
            x_work[idx] = x[idx] + eps
            J_plus = self._total_cost(x_work)
            x_work[idx] = x[idx] - eps
            J_minus = self._total_cost(x_work)
            x_work[idx] = x[idx]  # restore before next iteration
            g[i] = (J_plus - J_minus) / (2.0 * eps)
        self._push(x)  # restore all elements to original x
        return g


class AugmentedLagrangian:
    """Augmented-Lagrangian optimizer for amplitude-apodized Lyot coronagraphs.

    This implements the outer-loop structure used for constrained coronagraph
    design problems such as Por, Proc. SPIE 12180, 121805J (2022): maximize an
    amplitude apodizer throughput while imposing dark-hole contrast constraints.

    For fixed Lagrange multipliers ``lambda`` and penalty ``rho``, the inner
    optimizer minimizes

    ``-throughput + 1/(2*rho) * sum(max(0, lambda + rho*c(x))**2 - lambda**2)``

    where each inequality constraint defaults to

    ``c_i(x) = intensity_i(x) - contrast_i <= 0``.

    The constrained quantity is pluggable via ``constraint`` (or the
    ``constraint_kind`` shorthand): pass any object following the constraint
    protocol in :mod:`dygdug.cost_functions` to bound, e.g., the real and
    imaginary parts of the dark-hole field instead of its intensity.

    After each inner solve, multipliers are updated with

    ``lambda_i <- max(0, lambda_i + rho*c_i(x))``.

    Notes
    -----
    This class intentionally supports only the current amplitude-apodized pupil
    Lyot use case: ``coro.pupil`` must be a :class:`VariablePupil` in
    ``'amplitude'`` mode, and FPM/Lyot-stop variables are not included in the
    flat optimization vector.
    """

    def __init__(
        self,
        coro,
        optimizer,
        dark_hole,
        wvl,
        contrast,
        x0=None,
        lower_bounds=None,
        upper_bounds=None,
        penalty=1.0,
        penalty_growth=10.0,
        constraint_reduction=0.25,
        constraint_tolerance=0.0,
        throughput_weight=1.0,
        normalize_throughput=False,
        binarize_weight=0.0,
        binarize_kind="quadratic",
        constraint_kind="intensity",
        constraint=None,
        multipliers=None,
        optimizer_kwargs=None,
        include_fpm=True,
        relaxation_sigma=None,
        ftol=None,
        gtol=None,
        governor_factory=None,
    ):
        """Create an augmented-Lagrangian optimization problem.

        Parameters
        ----------
        coro : dygdug.models.Coronagraph
            Coronagraph model.  Only ``coro.pupil`` is optimized, and it must
            be a real-valued :class:`VariablePupil` in amplitude mode.
        optimizer : callable
            Optimizer class/factory from ``prysm.x.optym``.  For example,
            pass ``PrysmLBFGSB`` rather than an already-stepped instance.
        dark_hole : ndarray
            Boolean or 0/1 mask selecting constrained focal-plane pixels.
        wvl : float or sequence of float
            Wavelength(s) included in the constraints.
        contrast : float or ndarray
            Maximum allowed intensity in the dark hole.  May be scalar, a
            vector of dark-hole length, a full focal-plane image, or a
            wavelength-indexed array of either form.
        x0 : ndarray, optional
            Initial pupil-amplitude vector.  Defaults to the current active
            values in ``coro.pupil``.
        lower_bounds, upper_bounds : ndarray or float, optional
            Bounds passed to the inner optimizer.  Defaults to ``0 <= x <= 1``.
        penalty : float, optional
            Initial augmented-Lagrangian penalty ``rho``.
        penalty_growth : float, optional
            Factor used to increase ``rho`` when constraints do not improve.
        constraint_reduction : float, optional
            Required fractional improvement in max violation to avoid
            increasing ``rho`` on the next outer iteration.
        constraint_tolerance : float, optional
            Violation below this value is considered feasible for penalty
            update purposes.
        throughput_weight : float, optional
            Weight multiplying the negative throughput objective.
        normalize_throughput : bool, optional
            If True, optimize mean active-pupil transmission instead of sum.
        binarize_weight : float, optional
            Weight of a double-well penalty ``beta * sum((x - l)(u - x))`` that
            drives pixels toward the bounds for a near-binary apodizer.  Default
            0 (off).  Mutable as ``model.binarize_weight`` so it can be annealed
            up across outer iterations; normalized like throughput, so it trades
            off against ``throughput_weight`` on a common scale.
        binarize_kind : {'quadratic', 'tent'}, optional
            Shape of the binarization penalty.  ``'quadratic'`` (default) is the
            smooth double well ``(x - l)(u - x)`` whose force vanishes at
            mid-gray.  ``'tent'`` is the nearest-bound distance
            ``min(x - l, u - x)``, which applies constant force ``+-beta``
            toward the nearest bound everywhere (including mid-gray), so it
            binarizes stuck mid-gray pixels far more aggressively.
        constraint_kind : {'intensity', 'field'}, optional
            Convenience selector for a built-in dark-hole constraint, used only
            when ``constraint`` is None.  ``'intensity'`` (default) bounds
            ``|E|^2 / norm <= contrast`` -- a convex quadratic whose optimum lies
            on a smooth boundary (gray).  ``'field'`` bounds the normalized field
            amplitude components linearly, ``|Re a| <= s`` and ``|Im a| <= s``
            with ``a = E / sqrt(norm)`` and ``s = sqrt(contrast / 2)``: a linear
            program whose feasible polytope has binary vertices, so throughput
            maximization drives a binary apodizer.  ``contrast`` keeps the same
            meaning (max intensity relative to the direct peak) in both modes.
            In field mode the multipliers have shape ``(n_wvl, 2, n)`` (axis 1
            indexes real/imag).  Maps to :class:`IntensityConstraint` /
            :class:`FieldBoxConstraint` respectively.
        constraint : constraint object or class, optional
            Pluggable dark-hole constraint that *replaces* ``constraint_kind``
            when given.  Must follow the constraint protocol in
            :mod:`dygdug.cost_functions` -- ``multiplier_shape(n_wvl, n)``,
            ``residual(E_dh, norm, contrast)`` (per-pixel values, ``<= 0``
            feasible), and ``grad_seed(E_dh, norm, contrast, positive)`` (the
            Wirtinger seed ``dJ/dE*`` of the active constraints).  This lets the
            constrained quantity itself be defined on the electric field -- e.g.
            bounding the real and imaginary parts -- rather than on intensity.
            The multiplier shape and ``constraint_values`` shape follow the
            object's ``multiplier_shape``.
        multipliers : ndarray, optional
            Initial nonnegative Lagrange multipliers.  Defaults to zero.
        optimizer_kwargs : dict, optional
            Extra keyword arguments forwarded to the inner optimizer.
        include_fpm : bool, optional
            Whether the coronagraph propagation includes the FPM.
        relaxation_sigma : float or None, optional
            Standard deviation in pixels of the Gaussian kernel applied for
            progressive relaxation (Por, Proc. SPIE 12180, 121805J, 2022).
            After each inner solve (except the final outer iteration) the
            apodizer amplitude map is blurred by this kernel to smooth sharp
            features and perturb the solution out of local minima before the
            next inner solve begins.  The blur preserves the total active-pixel
            sum and clips the result to the configured amplitude bounds.
            Set to ``None`` (default) to disable progressive relaxation.
        ftol : float, optional
            If set, end an inner solve early once consecutive inner-objective
            values change by less than this (relative) tolerance, via a
            ``prysm.x.optym.governors.FunctionTolerance``.  ``None`` (default)
            disables this stop condition.
        gtol : float, optional
            If set, end an inner solve early once the inner-objective gradient
            infinity-norm falls below this value, via a ``GradientTolerance``.
            ``None`` (default) disables this stop condition.
        governor_factory : callable, optional
            Zero-argument callable returning a fresh
            ``prysm.x.optym.governors.Governor`` for each inner solve.  Use
            this for full control over the stop conditions (e.g. ``StepTolerance``,
            ``MaxEvaluations``, or a custom ``AllGovernor``).  When given, it
            takes precedence over ``ftol``/``gtol``.  The factory is called once
            per inner solve so per-solve governor state does not leak across
            outer iterations.
        """
        if not isinstance(coro.pupil, VariablePupil):
            raise TypeError(
                "AugmentedLagrangian requires coro.pupil to be VariablePupil"
            )
        if coro.pupil.mode != "amplitude" or np.iscomplexobj(coro.pupil.data):
            raise NotImplementedError(
                "AugmentedLagrangian currently supports amplitude pupils only"
            )
        if hasattr(coro.fpm, "n_params") or hasattr(coro.lyot_stop, "n_params"):
            raise NotImplementedError(
                "AugmentedLagrangian currently optimizes only the pupil plane"
            )

        self.coro = coro
        self.optimizer = optimizer
        # Inner-solve stop conditions (prysm governors).  Disabled by default
        # (ftol/gtol None and no factory) so the inner loop runs the full
        # ``inner_steps`` budget unless the optimizer self-terminates.  A fresh
        # governor is built per inner solve via ``_make_governor`` so tolerance
        # state does not leak across outer iterations.
        self.ftol = ftol
        self.gtol = gtol
        self._governor_factory = governor_factory
        self.include_fpm = include_fpm
        self.wvl = [wvl] if np.ndim(wvl) == 0 else list(wvl)

        # One-time: move the coronagraph's persistent arrays (pupil, Lyot stop,
        # FPM cache, executor matrices) onto the active prysm backend so the
        # inner forward/reverse loop runs entirely on-device.  No-op on CPU.
        sync_coronagraph(coro, self.wvl)

        self.dh = np.asarray(dark_hole).astype(bool)
        self._dh_idx = np.flatnonzero(self.dh.ravel())
        if self._dh_idx.size == 0:
            raise ValueError("dark_hole must select at least one focal-plane pixel")

        self.pupil = coro.pupil
        self._pupil_idx = getattr(
            self.pupil, "_mask_idx", np.flatnonzero(self.pupil.mask.ravel())
        )
        self.n_params = int(self._pupil_idx.size)

        if x0 is None:
            self.x = self.pupil.data.ravel()[self._pupil_idx].astype(float).copy()
        else:
            self.x = np.asarray(x0, dtype=float).copy()
        if self.x.size != self.n_params:
            raise ValueError(f"x0 has length {self.x.size}, expected {self.n_params}")

        self.lower_bounds = self._as_bound(lower_bounds, 0.0)
        self.upper_bounds = self._as_bound(upper_bounds, 1.0)

        # Resolve the dark-hole constraint object.  An explicit ``constraint``
        # (instance or class) takes precedence; otherwise ``constraint_kind``
        # selects a built-in.  The object owns the constraint residual, its
        # field-gradient seed, and the multiplier shape, so ``fg`` and
        # ``constraint_values`` stay agnostic to intensity vs. field.
        if constraint is None:
            if constraint_kind not in ("intensity", "field"):
                raise ValueError(
                    f"constraint_kind must be 'intensity' or 'field', got "
                    f"{constraint_kind!r}"
                )
            self.constraint = (
                FieldBoxConstraint()
                if constraint_kind == "field"
                else IntensityConstraint()
            )
        else:
            self.constraint = constraint() if isinstance(constraint, type) else constraint
        self.constraint_kind = constraint_kind

        self.contrast = self._prepare_contrast(contrast)
        # Multiplier (and constraint) shape is delegated to the constraint
        # object.  Intensity: one inequality per dark-hole pixel per wavelength,
        # (n_wvl, n).  Field (LP): two -- |Re a| <= s and |Im a| <= s -- so an
        # extra axis of size 2 (0 -> real, 1 -> imag), (n_wvl, 2, n).
        n_wvl, n_constraints = self.contrast.shape
        mult_shape = tuple(self.constraint.multiplier_shape(n_wvl, n_constraints))
        self._mult_shape = mult_shape
        if multipliers is None:
            self.multipliers = np.zeros(mult_shape, dtype=float)
        else:
            self.multipliers = np.asarray(multipliers, dtype=float).copy()
            if self.multipliers.shape != mult_shape:
                raise ValueError(
                    f"multipliers must have shape {mult_shape}, "
                    f"got {self.multipliers.shape}"
                )
            self.multipliers = np.maximum(self.multipliers, 0)

        self.penalty = float(penalty)
        self.penalty_growth = float(penalty_growth)
        self.constraint_reduction = float(constraint_reduction)
        self.constraint_tolerance = float(constraint_tolerance)
        self.throughput_weight = float(throughput_weight)
        self.throughput_scale = self.throughput_weight
        if normalize_throughput:
            self.throughput_scale /= self.n_params

        # Binarization (double-well) penalty weight.  Mutable so callers can
        # anneal it across outer iterations (graduated binarization): keep it 0
        # while the dark hole is dug, then ramp up.  The effective scale tracks
        # this weight via the binarize_scale property and is normalized the same
        # way as throughput, so the two objectives trade off on an O(1) scale.
        self.binarize_weight = float(binarize_weight)
        if binarize_kind not in ("quadratic", "tent"):
            raise ValueError(
                f"binarize_kind must be 'quadratic' or 'tent', got {binarize_kind!r}"
            )
        self.binarize_kind = binarize_kind
        self._normalize_throughput = bool(normalize_throughput)
        self.optimizer_kwargs = (
            {} if optimizer_kwargs is None else dict(optimizer_kwargs)
        )

        self.relaxation_sigma = relaxation_sigma

        self.outer_iter = 0
        self.last_violation = None
        self.last_inner_optimizer = None
        self.history = []
        self._Ebar = None

        # Starting point for the next inner solve.  Kept separate from
        # ``self.x`` (the converged solution) so progressive relaxation can
        # perturb the restart without corrupting the dual update (which must
        # see the unperturbed inner optimum).
        self._x_start = self.x.copy()

        self._push(self.x)

        # Contrast normalization: the dark-hole constraints are expressed as a
        # true contrast relative to the peak of the direct (no-FPM) PSF.  The
        # normalization is evaluated once here from the initial apodizer; it is
        # not recomputed every forward pass.  Over the course of optimization
        # the direct peak changes only at the ~10% level (it tracks throughput),
        # which is negligible for constraint scaling.
        self._normalization = np.empty(len(self.wvl), dtype=float)
        for iw, wvl_ in enumerate(self.wvl):
            E_direct = self.coro.forward(wvl_, include_fpm=False)
            peak = float(np.max(E_direct.real * E_direct.real
                                + E_direct.imag * E_direct.imag))
            self._normalization[iw] = peak if peak > 0 else 1.0

        # Restore the coronagraph state to the actual apodizer (forward above
        # ran with include_fpm=False purely to measure the direct peak).
        self._push(self.x)

    @property
    def binarize_scale(self):
        """Effective binarization weight (normalized like throughput)."""
        scale = self.binarize_weight
        if self._normalize_throughput:
            scale /= self.n_params
        return scale

    def _as_bound(self, value, default):
        """Return a length-``n_params`` bound vector."""
        if value is None:
            return np.full(self.n_params, default, dtype=float)
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(self.n_params, float(arr), dtype=float)
        if arr.size != self.n_params:
            raise ValueError(f"bound has length {arr.size}, expected {self.n_params}")
        return arr.copy()

    def _prepare_contrast(self, contrast):
        """Broadcast contrast input to ``(n_wavelengths, n_constraints)``."""
        arr = np.asarray(contrast, dtype=float)
        n_wvl = len(self.wvl)
        n_constraints = self._dh_idx.size

        if arr.ndim == 0:
            return np.full((n_wvl, n_constraints), float(arr), dtype=float)

        if arr.shape == self.dh.shape:
            selected = arr.ravel()[self._dh_idx]
            return np.tile(selected, (n_wvl, 1))

        if arr.ndim == 1:
            if arr.size == n_constraints:
                return np.tile(arr.copy(), (n_wvl, 1))
            if arr.size == n_wvl:
                return np.repeat(arr[:, None], n_constraints, axis=1)

        if arr.shape == (n_wvl, n_constraints):
            return arr.copy()

        if arr.ndim == self.dh.ndim + 1 and arr.shape[0] == n_wvl:
            if arr.shape[1:] == self.dh.shape:
                return np.asarray([plane.ravel()[self._dh_idx] for plane in arr])

        raise ValueError(
            "contrast must be scalar, full focal-plane shaped, dark-hole length, "
            "wavelength length, or wavelength-indexed focal/dark-hole shaped"
        )

    def _push(self, x):
        """Push an active-pupil vector into the coronagraph pupil."""
        self.pupil.update(x)

    def _zeroed_Ebar(self, like):
        """Return a reusable zero-filled adjoint seed shaped like ``like``."""
        if (
            self._Ebar is None
            or self._Ebar.shape != like.shape
            or self._Ebar.dtype != like.dtype
        ):
            self._Ebar = np.zeros_like(like)
        else:
            self._Ebar.fill(0)
        return self._Ebar

    def fg(self, x):
        """Evaluate the current augmented Lagrangian and gradient.

        This method is passed to the inner ``prysm.x.optym`` optimizer.  The
        Lagrange multipliers and penalty are fixed during each inner solve.
        """
        self._push(x)

        f = -self.throughput_scale * np.sum(x)
        g = np.full(self.n_params, -self.throughput_scale, dtype=float)

        # Binarization penalty: drives pixels toward the bounds for a near-binary
        # apodizer.  Concave (it un-convexifies the objective), so anneal beta up
        # from 0 once the dark hole is feasible rather than enabling it early.
        beta = self.binarize_scale
        if beta:
            l, u = self.lower_bounds, self.upper_bounds
            if self.binarize_kind == "tent":
                # Distance to nearest bound: beta * sum(min(x - l, u - x)).
                # Constant force toward the nearest bound everywhere (gradient
                # +-beta), including at mid-gray -- where the quadratic well's
                # force vanishes -- so it crystallizes stuck mid-gray pixels.
                f += float(beta * np.sum(np.minimum(x - l, u - x)))
                g += beta * np.sign(0.5 * (l + u) - x)
            else:
                # Quadratic double well: beta * sum((x - l)(u - x)).  Smooth, but
                # its force beta * (l + u - 2x) vanishes at mid-gray, so mid-gray
                # pixels barely move no matter how large beta is.
                f += float(beta * np.sum((x - l) * (u - x)))
                g += beta * (l + u - 2 * x)

        for iw, wvl in enumerate(self.wvl):
            E_focal = self.coro.forward(wvl, include_fpm=self.include_fpm)
            E_dh = E_focal.ravel()[self._dh_idx]
            norm = self._normalization[iw]
            contrast = self.contrast[iw]

            # The constraint object supplies the residual c (<= 0 feasible) and
            # the field-gradient seed; the AL penalty machinery below is identical
            # whether c bounds intensity, |Re a|/|Im a|, or any custom quantity.
            c = self.constraint.residual(E_dh, norm, contrast)
            lam = self.multipliers[iw]
            positive = np.maximum(lam + self.penalty * c, 0)
            f += float(
                (np.sum(positive * positive) - np.sum(lam * lam)) / (2 * self.penalty)
            )

            if np.any(positive > 0):
                # d(penalty)/dc = positive, so the focal-plane seed is
                # dJ/dE* = positive * dc/dE*, assembled by the constraint object.
                Ebar = self._zeroed_Ebar(E_focal)
                Ebar.ravel()[self._dh_idx] = self.constraint.grad_seed(
                    E_dh, norm, contrast, positive
                )
                self.coro.reverse(Ebar, wvl, include_fpm=self.include_fpm)
                adj = self.coro.adjoint_at_entrance_pupil.ravel()[self._pupil_idx]
                g += 2 * np.real(adj)

        return f, g

    def constraint_values(self, x=None):
        """Return the inequality constraint values (``<= 0`` is feasible).

        Intensity: ``intensity/norm - contrast`` per pixel, shape
        ``(n_wvl, n)``.  Field: the ``|Re a| - s`` and ``|Im a| - s`` constraints
        on the normalized amplitude, shape ``(n_wvl, 2, n)``.
        """
        if x is not None:
            self._push(x)

        constraints = np.empty(self._mult_shape, dtype=float)
        for iw, wvl in enumerate(self.wvl):
            E_dh = self.coro.forward(wvl, include_fpm=self.include_fpm).ravel()[
                self._dh_idx
            ]
            constraints[iw] = self.constraint.residual(
                E_dh, self._normalization[iw], self.contrast[iw]
            )
        return constraints

    def violation(self, constraints=None):
        """Return maximum positive contrast-constraint violation."""
        if constraints is None:
            constraints = self.constraint_values()
        return float(np.max(np.maximum(constraints, 0)))

    def update_multipliers(self, constraints=None):
        """Perform the nonnegative inequality multiplier update."""
        if constraints is None:
            constraints = self.constraint_values()
        self.multipliers = np.maximum(0, self.multipliers + self.penalty * constraints)
        return self.violation(constraints)

    def _apply_relaxation(self, x):
        """Blur the apodizer amplitude map with a Gaussian to escape local minima.

        The blurred solution is energy-preserving: the total sum of the
        active-pixel transmission values is unchanged.  Pixel values are then
        clipped to the configured ``[lower_bounds, upper_bounds]`` range so that
        the smoothed starting point remains feasible.

        Parameters
        ----------
        x : ndarray
            Current active-pupil amplitude vector (length ``n_params``).

        Returns
        -------
        x_smooth : ndarray
            Smoothed amplitude vector, same length as *x*.
        """
        # Reconstruct the full 2-D amplitude map (zeros outside the pupil).
        shape = self.pupil.data.shape
        img = np.zeros(shape, dtype=float)
        # reshape(-1) (a view on the contiguous array) rather than .flat:
        # cupy's flat iterator does not support fancy indexing.
        img.reshape(-1)[self._pupil_idx] = x

        # Apply isotropic Gaussian blur.  Values stay in [0, 1] because the
        # kernel is normalized and inputs are non-negative, so boundary pixels
        # can only decrease toward zero — never exceed the original maximum.
        # ndimage is the prysm backend shim: scipy.ndimage on CPU, cupyx.scipy.ndimage on GPU.
        blurred = ndimage.gaussian_filter(img, sigma=self.relaxation_sigma)

        # Extract active-pixel values from the blurred image.
        x_smooth = blurred.reshape(-1)[self._pupil_idx].copy()

        # Re-scale to preserve the total active-pixel transmission (energy).
        # Boundary bleed reduces the in-pupil sum slightly; undo that effect.
        s_orig = float(np.sum(x))
        s_smooth = float(np.sum(x_smooth))
        if s_smooth > 0:
            x_smooth = x_smooth * (s_orig / s_smooth)

        # Clip to configured amplitude bounds [0, 1] after rescaling.
        x_smooth = np.clip(x_smooth, self.lower_bounds, self.upper_bounds)

        return x_smooth

    def _make_inner_optimizer(self, x0):
        """Instantiate the configured inner optimizer for the current AL state."""
        kwargs = dict(self.optimizer_kwargs)
        kwargs.setdefault("lower_bounds", self.lower_bounds)
        kwargs.setdefault("upper_bounds", self.upper_bounds)
        return self.optimizer(self.fg, x0.copy(), **kwargs)

    def _make_governor(self):
        """Build a fresh stop-condition governor for one inner solve.

        Returns ``None`` when no stop conditions are configured (the inner
        loop then runs the full ``inner_steps`` budget unless the optimizer
        self-terminates).  Rebuilt every inner solve because tolerance
        governors carry per-solve state (e.g. ``FunctionTolerance``'s previous
        objective value) and the AL objective changes between outer iterations
        as the multipliers and penalty update, so f/g histories are not
        comparable across solves.
        """
        if self._governor_factory is not None:
            return self._governor_factory()
        conditions = []
        if self.ftol is not None:
            conditions.append(FunctionTolerance(self.ftol))
        if self.gtol is not None:
            conditions.append(GradientTolerance(self.gtol))
        if not conditions:
            return None
        return AnyGovernor(conditions)

    def step(self, inner_steps=50, apply_relaxation=True, progress=False):
        """Run one augmented-Lagrangian outer iteration.

        Parameters
        ----------
        inner_steps : int, optional
            Maximum number of inner optimizer ``step()`` calls for the current
            multiplier/penalty state.
        progress : bool, optional
            If True, display a ``tqdm`` progress bar over the inner optimizer
            steps.  Requires ``tqdm`` to be installed.
        apply_relaxation : bool, optional
            Whether to apply progressive relaxation (Gaussian smoothing) when
            forming the starting point for the *next* inner solve, if
            ``relaxation_sigma`` was set.  The converged solution returned by
            this call (``self.x``) and the dual update are never perturbed, so
            this only affects where the next inner solve begins.  Pass ``False``
            on the final outer iteration to skip the (then-pointless) blur.

        Returns
        -------
        x : ndarray
            Current pupil-amplitude vector after the inner solve.
        info : dict
            Diagnostics for the completed outer iteration.
        """
        inner = self._make_inner_optimizer(self._x_start)
        governor = self._make_governor()
        inner_iters = 0
        inner_status = "max_inner_steps"

        pbar = None
        if progress:
            from tqdm import tqdm

            pbar = tqdm(
                total=inner_steps,
                desc=f"inner (outer {self.outer_iter})",
                leave=False,
            )

        for _ in range(inner_steps):
            try:
                # PrysmLBFGSB.step() returns (x_pre, f, g) evaluated at the
                # pre-step iterate; inner.x is the post-step iterate.
                x_pre, f_pre, g_pre = inner.step()
                inner_iters += 1
            except StopIteration:
                inner_status = "optimizer_stopped"
                break
            if pbar is not None:
                pbar.update(1)

            # Governor-based early exit (function/gradient tolerance, etc.).
            if governor is not None:
                record = StepRecord(
                    optimizer=inner,
                    iteration=inner_iters,
                    x=x_pre,
                    f=f_pre,
                    g=g_pre,
                    x_next=inner.x,
                    metadata=getattr(inner, "last_step_metadata", None),
                )
                decision = governor.observe(record)
                if decision.stop:
                    inner_status = decision.message or "governor_stop"
                    break

        if pbar is not None:
            pbar.close()

        if not hasattr(inner, "x"):
            raise AttributeError(
                "inner optimizer must expose its current iterate as .x"
            )

        self.x = inner.x.copy()
        self._push(self.x)

        # Dual update first, on the *converged* inner solution.  Multipliers and
        # the penalty must reflect the true constraint values at the inner
        # optimum; computing them after a relaxation blur would feed the
        # perturbation artifacts (which raise dark-hole intensity) straight into
        # the multipliers and inflate the penalty.
        constraints = self.constraint_values()
        max_violation = self.update_multipliers(constraints)

        penalty_before = self.penalty
        # if (
        #     self.last_violation is not None
        #     and max_violation > self.constraint_tolerance
        #     and max_violation > self.constraint_reduction * self.last_violation
        # ):
        self.penalty *= self.penalty_growth

        # Progressive relaxation: blur the apodizer to perturb the *next* inner
        # solve out of local minima.  Only the starting point for the next solve
        # is perturbed; ``self.x`` (the converged/returned solution) and the dual
        # update above are left untouched.  Skipped on the final outer iteration.
        relaxation_applied = False
        if apply_relaxation and self.relaxation_sigma is not None:
            self._x_start = self._apply_relaxation(self.x)
            relaxation_applied = True
        else:
            self._x_start = self.x.copy()

        info = {
            "outer_iter": self.outer_iter,
            "inner_iters": inner_iters,
            "inner_status": inner_status,
            "max_violation": max_violation,
            "penalty_before": penalty_before,
            "penalty_after": self.penalty,
            "multiplier_norm": float(
                np.sqrt(np.sum(self.multipliers * self.multipliers))
            ),
            "throughput": float(np.sum(self.x)),
            "relaxation_applied": relaxation_applied,
        }

        self.last_violation = max_violation
        self.last_inner_optimizer = inner
        self.history.append(info)
        self.outer_iter += 1
        return self.x, info

    def solve(self, outer_steps=10, inner_steps=50, progress=False):
        """Run multiple augmented-Lagrangian outer iterations.

        Progressive relaxation (if ``relaxation_sigma`` is set) is applied
        after every inner solve *except* the last outer iteration, so the
        final returned solution is the direct optimizer output without any
        post-processing blur.

        Parameters
        ----------
        progress : bool, optional
            If True, display ``tqdm`` progress bars over both the outer
            iterations and each inner solve.  Requires ``tqdm`` installed.
        """
        outer_iterator = range(outer_steps)
        if progress:
            from tqdm import tqdm

            outer_iterator = tqdm(outer_iterator, desc="outer")

        for i in outer_iterator:
            is_last = i == outer_steps - 1
            self.step(
                inner_steps=inner_steps,
                apply_relaxation=not is_last,
                progress=progress,
            )
        return self.x, self.history


class JointOptimizer:
    def __init__(self, optlist):
        """
        For performing joint optimization of multiple parameters
        """
        self.optlist = optlist

    def fg(self, x):
        """
        Sums all of the functions and gradients
        """
        f = 0
        g = np.zeros(x.shape)

        for opt in self.optlist:
            _f, _g = opt.fg(x)
            f += _f
            g += _g

        return f, g
