from scipy.optimize import _lbfgsb

import numpy as np

class F77LBFGSB:
    def __init__(self, cost_grad_func, x0, memory=10, lower_bounds=None, upper_bounds=None):
        self.fg = cost_grad_func
        self.x0 = x0
        self.n = len(x0)  # n = n vars
        self.m = memory

        # create the work arrays Fortran needs
        fint_dtype = _lbfgsb.types.intvar.dtype
#         ffloat_dtype = x0.dtype  maybe can uncomment this someday, but probably not.
        ffloat_dtype = np.float64

        # todo: f77 code explodes for f32 dtype?
        if lower_bounds is None:
            lower_bounds = np.full(self.n, -np.Inf, dtype=ffloat_dtype)

        if upper_bounds is None:
            upper_bounds = np.full(self.n, np.Inf, dtype=ffloat_dtype)

        # nbd is an array of integers for Fortran
        #         nbd(i)=0 if x(i) is unbounded,
        #                1 if x(i) has only a lower bound,
        #                2 if x(i) has both lower and upper bounds, and
        #                3 if x(i) has only an upper bound.
        nbd = np.zeros(self.n, dtype=fint_dtype)
        self.l = lower_bounds  # NOQA
        self.u = upper_bounds
        finite_lower_bound = np.isfinite(self.l)
        finite_upper_bound = np.isfinite(self.u)
        # unbounded case handled in init as zeros
        lower_but_not_upper_bound = finite_lower_bound & ~finite_upper_bound
        upper_but_not_lower_bound = finite_upper_bound & ~finite_lower_bound
        both_bounds = finite_lower_bound & finite_upper_bound
        nbd[lower_but_not_upper_bound] = 1
        nbd[both_bounds]               = 2  # NOQA
        nbd[upper_but_not_lower_bound] = 3
        self.nbd = nbd

        # much less complicated initializations
        m, n = self.m, self.n
        self.x = x0.copy()
        self.f = np.array([0], dtype=ffloat_dtype)
        self.g = np.zeros([self.n], dtype=ffloat_dtype)
        # see lbfgsb.f for this size
        # error in the docstring, see line 240 to 252
        self.wa = np.zeros(2 * m * n + 11 * m ** 2 + 5 * n + 8 * m, dtype=ffloat_dtype)
        self.iwa = np.zeros(3*n, dtype=fint_dtype)
        self.task = np.zeros(1, dtype='S60')  # S60 = <= 60 character wide byte array
        self.csave = np.zeros(1, dtype='S60')
        self.lsave = np.zeros(4, dtype=fint_dtype)
        self.isave = np.zeros(44, dtype=fint_dtype)
        self.dsave = np.zeros(29, dtype=ffloat_dtype)
        self.task[:] = 'START'

        self.iter = 0

        # try to prevent F77 driver from ever stopping on its own
        # cannot use NaN or Inf, Fortran comparisons do not work
        # properly, so pick unreasonably small numbers.
        # TODO: would a negative number be better here?
        self.factr = 1e-999
        self.pgtol = 1e-999

        # other stuff to be added to the interface later
        self.maxls = 30
        self.iprint = 1

    def _call_fortran(self):
        _lbfgsb.setulb(self.m, self.x, self.l, self.u, self.nbd, self.f, self.g,
                       self.factr, self.pgtol, self.wa, self.iwa, self.task, self.iprint,
                       self.csave, self.lsave, self.isave, self.dsave, self.maxls)

    def _view_s(self):
        m, n = self.m, self.n
        # flat => matrix storage => truncate to only valid rows
        return self.wa[0:m*n].reshape(m, n)[:self._valid_space_sy]

    def _view_y(self):
        m, n = self.m, self.n
        # flat => matrix storage => truncate to only valid rows
        return self.wa[m*n:2*m*n].reshape(m, n)[:self._valid_space_sy]

    @property
    def _nbfgs_updates(self):
        return self.isave[30]

    @property
    def _valid_space_sy(self):
        return min(self._nbfgs_updates, self.m)

    def step(self):
        self.iter += 1  # increment first so that while loop is self-breaking
        while self._nbfgs_updates < self.iter:
            # call F77 mutates all of the class's state
            self._call_fortran()
            # strip null bytes/termination and any ASCII white space
            task = self.task.tobytes().strip(b'\x00').strip()
            if task.startswith(b'FG'):
                f, g = self.fg(self.x)
                self.f[:] = f
                self.g[:] = g
                self._call_fortran()

            if _fortran_died(task):
                msg = task.decode('UTF-8')
                raise ValueError("the Fortran L-BFGS-B driver thinks something is wrong with the problem and gave the message " + msg)

            if _fortran_converged(task):
                raise StopIteration

            if _fortran_major_iter_complete(task):
                break

        return self.x, self.f, self.g


def _fortran_died(task):
    return task.startswith(b'STOP')


def _fortran_converged(task):
    return task.startswith(b'CONV')


def _fortran_major_iter_complete(task):
    return task.startswith(b'NEW_X')
