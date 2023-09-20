""":class:`.FuzzyNumber`\\ s defined by a parameterized Python method."""

from __future__ import annotations

from math import log, exp, sqrt
from typing import Union, Tuple

import numpy as np
from scipy.stats import norm as gauss

from fuzzy.crisp import Interpolator
from fuzzy.number import Domain, Literal, _Numerical, default_interpolator
from fuzzy.truth import Truth, default_threshold


def _scale(data: Union[np.ndarray, Iterable[float], float],
           expected_range: Tuple[float, float] = (0, 1), intended_range: Tuple[float, float] = (0, 1),
           map: str = "lin") -> Union[np.ndarray, float]:
    """A helper for scaling the input to literals.

    This is intended for scaling truth-valued parameters onto some sub-range of [0,1].  With the defaults left as
    they are, it does nothing.  If you have real-world data in some units, you might need to scale it onto the range
    of truth to represent that data as a literal fuzzy number.  Handily, ``expected_range = None`` will normalize
    the input to ``expected_range``.

    Args:
        data:  Numbers that you want to scale from one range to another.
        expected_range:  The range, (min, max) that they might vary over.  If max<min, the sense of the data is flipped.
            If ``None``, it will be set to their actual range of variation (for this call), resulting in normalization.
        intended_range:  The range they will be mapped onto.    If max<min, the sense of the data is flipped.
            If ``None``, the ``default_threshold`` will be returned.
        map:  How the mapping will be done.  There are three options:

            * linear:  "lin" == "invlin"
            * logarithmic:  "log" == "invexp"
            * exponential:  "exp" == "invlog"

            (Data may be mapped and then recovered by a second mapping that
            exchanges the ranges and uses the inverse map.)

        Returns:
            The output type depends on the input type: a :class:`.ndarray` for any ``Iterable`` or a ``float``
            for a ``float``.  The data is scaled from ``expected_range`` to ``intended_range`` according
            to ``map`` with validation according to ``clip``."""
    t = np.array(data)
    if expected_range is None:
        expected_range = (np.min(data), np.max(data))
    if intended_range is None:
        return default_threshold
    t = Truth.scale(t, "in", expected_range, map, False)
    t = Truth.scale(t, "out", intended_range, "lin", True)
    return t


class Triangle(Literal):
    """Describes a fuzzy number as a triangular function."""

    def __init__(self, a: float, b: float, c: float, **kwargs):
        """
        Args:
            a, b, c:  The minimum, preferred, and maximum values.  The function is piecewise linear between
                these points.  The condition :math:`a \\le b \\le c` must hold
            range:  Default: (0,1).  The truths are:

                | ``'expected_range'[0]`` at ``a`` and ``c``,
                | ``'expected_range'[1]`` at ``b``.

        Other Parameters:

            kwargs: Keyword arguments relating to domain, range, discretization, and exceptional points:
                See :class:`.Literal`.
            'domain':  Default: [``a``, ``c``].
            'origin':  Default: ``b``.
            """
        super().__init__(**kwargs)
        if not (a <= b <= c):
            raise ValueError("a <= b <= c must hold.")
        self.a, self.b, self.c = a, b, c
        if self.d is None:
            self.d = Domain((a, c))
        if self.origin is None:
            self.origin = b
        # Oh, who cares?:
        # if (self.discrete is not None) and ((self.discrete < self.d[0]).any() or (self.discrete > self.d[1]).any()):
        #     raise ValueError("discrete points outside domain are redundant---they'd report the default anyway.")

    def __str__(self):
        s = super().__str__()
        dir = "peak" if (self.range[1] > self.range[0]) else "nadir"
        return str(f"triangle ({self.a},{self.b},{self.c}) "
                   f"with {dir} at ({self.b:,.4g}, {self.range[1]:,.4g}):---\n {s}")

    def _sample(self, v: np.ndarray) -> np.ndarray:
        """Returns the truth for every value in ``v``."""
        if self.b == self.d[0]:
            left = np.ones_like(v, self.range[1])
        else:
            left = (v - self.a) / (self.b - self.a)
        if self.d[1] == self.b:
            right = np.ones_like(v, self.range[1])
        else:
            right = (self.c - v) / (self.c - self.b)
        s = np.fmin(left, right)
        s = s * (self.range[1] - self.range[0]) + self.range[0]
        lo, hi = min(self.range[0], self.range[1]), max(self.range[0], self.range[1])
        return np.clip(s, lo, hi)


class Trapezoid(Literal):
    """Describes a fuzzy number as a trapezoidal function."""

    def __init__(self, a: float, b: float, c: float, d: float, **kwargs):
        """Args:
            a, b, c, d:  The extreme domain, [a,d], and, within it, the preferred domain [b,c].
                The function is piecewise linear between
                these points.  The condition :math:`a \\le b \\le c \\le d` must hold.
            range:  Default: (0,1).  The truths are:

                | ``range[0]`` at ``a`` and ``d``,
                | ``range[1]`` at ``b`` and ``c``.

        Other Parameters:

            kwargs: Keyword arguments relating to domain, range, discretization, and exceptional points:
                See :class:`.Literal`.
            origin:  Default: the center of the preferred region.
            """
        super().__init__(**kwargs)
        if not (a <= b <= c <= d):
            raise ValueError("a <= b <= c <= d must hold.")
        if self.d is None:
            self.d = Domain((a, d))
        if self.origin is None:
            self.origin = (c - b) / 2
        self.a = a
        self.b = b
        self.c = c
        self.f = d  # self.d and  self.e are already used

    def __str__(self):
        s = super().__str__()
        dir = "plateau" if (self.range[1] > self.range[0]) else "valley"
        return str(f"trapezoid with {dir} at {self.range[1]:,.4g} on ({self.b:,.4g}, {self.c:,.4g}); {s}")

    def _sample(self, v: np.ndarray) -> np.ndarray:
        """Returns the truth for every value in ``v``."""
        if self.b == self.a:
            left = np.ones_like(v)
        else:
            left = (v - self.a) / (self.b - self.a)
        if self.d[1] == self.c:
            right = np.ones_like(v)
        else:
            right = (self.f - v) / (self.f - self.c)
        middle = np.ones_like(v)
        triangle = np.fmin(left, right)
        s = np.fmin(triangle, middle)
        s = s * (self.range[1] - self.range[0]) + self.range[0]
        lo, hi = min(self.range[0], self.range[1]), max(self.range[0], self.range[1])
        return np.clip(s, lo, hi)


class Cauchy(Literal):
    """Describes a fuzzy number as the bell-shaped function due to Augustin-Louis Cauchy.

    This is a way of talking about a number as "``c`` ± ``hwhm``"."""

    def __init__(self, c: float, hwhm: float, **kwargs):
        """
        Args:
            c:  The most preferred value, at which the bell peaks.  It need not be in the ``domain``.
            hwhm: The half width at half maximum.

        Other Parameters:

            kwargs: Keyword arguments relating to domain, range, discretization, and exceptional points:
                See :class:`.Literal`.
            domain: Default:  the domain that covers the truth down to 1/1000 of its peak.
            origin:  Default: ``c``.
            """
        super().__init__(**kwargs)
        self.c = c
        if not (hwhm > 0):
            raise ValueError("hwhm must be greater than 0.")
        self.hwhm = hwhm
        if self.d is None:
            self.d = Domain((c - 31.6069612585582 * hwhm, c + 31.6069612585582 * hwhm))
        if self.origin is None:
            self.origin = c

    def __str__(self):
        s = super().__str__()
        dir = "peak" if (self.range[1] > self.range[0]) else "nadir"
        return str(f"Cauchy bell with {dir} at ({self.c:,.4g} ± {self.hwhm:,.4g}, {self.range[1]:,.4g}); {s}")

    def _sample(self, v: np.ndarray) -> np.ndarray:
        """Returns the truth for every value in ``v``."""
        s = (self.hwhm ** 2) / (np.square(v - self.c) + self.hwhm ** 2)
        s = s * (self.range[1] - self.range[0]) + self.range[0]
        return s


class Gauss(Literal):
    """Describes a fuzzy number as the bell-shaped function due to Carl Friedrich Gauss.

    This is a way of talking about a number as a normal distribution about an expectation value."""

    def __init__(self, c: float, sd: float, **kwargs):
        """
        Args:
            c:  The most preferred value, at which the bell peaks.  It need not be in the domain.
            sd: The size of one standard deviation---a larger value gives a wider bell.

        Other Parameters:

            kwargs: Keyword arguments relating to domain, range, discretization, and exceptional points:
                See :class:`.Literal`.
            domain: It can be given as a tuple, (min, max), or as ``float`` indicating the number of standard
                deviations about ``c``.  Default:  the domain that covers the truth down to 1/1000 of its peak.
            origin:  Default: ``c``.
            """
        super().__init__(**kwargs)
        self.c = c
        if not (sd > 0):
            raise ValueError("sd must be greater than 0.")
        self.sd = sd
        if self.d is None:  # if undefined:  .001 of peak
            self.d = Domain((c - 3.71692219 * sd, c + 3.71692219 * sd))
        if isinstance(self.d, float):
            d = self.d * sd
            self.d = Domain((c - d, c + d))
        if self.origin is None:
            self.origin = c

    def __str__(self):
        s = super().__str__()
        dir = "peak" if (self.range[1] > self.range[0]) else "nadir"
        return str(f"Gaussian bell with {dir} at ({self.c:,.4g} ± {1.17741 * self.sd:,.4g}, {self.range[1]:,.4g}) and "
                   f"standard deviation {self.sd:,.4g} \n {s}")

    def _sample(self, v: np.ndarray) -> np.ndarray:
        """Returns the truth for every value in ``v``."""
        s = self.sd * 2.50662827 * gauss.pdf(v, loc=self.c, scale=self.sd)
        s = s * (self.range[1] - self.range[0]) + self.range[0]
        return s


class Bell(Literal):
    """Describes a fuzzy number as a generalized bell membership function.

    This is a way of talking about a number as ``c`` ± ``hwhm`` with confidence ``slope``, or with
    vagueness ``transition_width``."""

    def __init__(self, c: float, hwhm: float, shape: float = 1, unit: str = "p", **kwargs):
        """
        Args:
            c: The center of the bell, the most preferred value.
            hwhm: The half width at half maximum of the bell.
            shape: A parameter affecting the shape of the sides.  Its significance depends on ``unit`` below.
                Default: 1, unit="p"---a plateau half the width.
            unit: The meaning of ``shape``.  It can have three values: {``s``, ``t``, ``p``}:
                * ``s``: **slope**---the steepness of the sides, the absolute slope at half maximum.
                * ``t``: **transition width**---the width, given in multiples of ``hwhm``, of the region
                  on either side of the bell where the truth varies on [.1, .9].
                * ``p``: **plateau width**---the width, given in multiples of ``hwhm``, of the region
                  in the middle of the bell where the truth varies on [.9, 1].

        Warning:
            It's fairly easy to set a moderately shallow ``slope`` and get a huge default ``domain``.
            It's wise to set ``domain`` manually if your ``slope < 1``.

        Other Parameters:

            kwargs: Keyword arguments relating to domain, range, discretization, and exceptional points:
                See :class:`.Literal`.
            domain: Default:  the domain that covers the truth down to 1/1000 of its peak.
            origin:  Default: ``c``.
            """
        super().__init__(**kwargs)
        self.c = c
        if hwhm == 0:
            raise ValueError("``a`` cannot equal zero.")
        self.hwhm = hwhm
        if unit == "s":  # shape is slope
            if not (shape > 0):
                raise ValueError("slope must be greater than 0.")
            b = shape * 2 * hwhm
        elif unit == "t":  # shape is transition_width
            if not (shape > 0):
                raise ValueError("transition width must be greater than 0.")
            b = 1.09861229 / log(.5 * (shape + sqrt(4 + (shape ** 2))))
        else:  # shape is plateau_width
            if (shape <= 0) or (shape >= 2 * hwhm):
                raise ValueError("plateau width must be greater than 0 and less than 2.")
            b = -1.09861228866811 / log(.5 * shape)
        self.b = b
        if self.d is None:  # if undefined:  .001 of peak
            w = hwhm * (3 ** (3 / (2 * b))) * (37 ** (1 / (2 * b)))
            self.d = Domain((c - w, c + w))
        if self.origin is None:
            self.origin = c

    def __str__(self):
        s = super().__str__()
        dir = "peak" if (self.range[1] > self.range[0]) else "nadir"
        tw = exp(1.09861 / self.b) - exp(-1.09861 / self.b)
        pw = 2 * self.hwhm * exp(-1.09861 / self.b)
        slope = .5 * self.b / self.hwhm
        sw = str(f"plateau width: {pw:,.4g}, transition width: {tw:,.4g}, slope: {slope:,.4g}")
        return str(f"bell with {dir} at ({self.c:,.4g} ± {self.hwhm:,.4g}, {self.range[1]:,.4g})---{sw}; \n{s}")

    def _sample(self, v: np.ndarray) -> np.ndarray:
        """Returns the truth for every value in ``v``."""
        s = 1 / (1 + np.abs((v - self.c) / self.hwhm) ** (2 * self.b))
        s = s * (self.range[1] - self.range[0]) + self.range[0]
        return s


class Inequality(Literal):
    """Describes a fuzzy inequality as piecewise linear.

    The "``sense``" of an inequality can be either ``">"`` or ``"<"``.
    An Inequality is a way of talking about an inequality as: "a value is ``sense c`` to within ``width``", or
    "with confidence ``slope``".  At ``c``, the truth is .5.  Surrounding this, there is a transition region where
    the truth varies linearly between (0,1).  On either side of this it is {0,1} depending on the ``sense``."""

    def __init__(self, sense: str, c: float, shape: float = 1, unit: str = "s", **kwargs):
        """
        Args:
            c:  The center of the transition, at which the truth is .5.  It need not be in the ``domain``.
            sense: Either ``">"`` (more suitable on the right---when greater than ``c``) or ``"<"``
                (more suitable on the left---when less than ``c``).
            shape: A parameter affecting the shape of the middle.  Its significance depends on ``unit`` below.
                Default: 1, unit="s"---a slope of 1.
            unit: The meaning of ``shape``.  It can have two values: {``s``, ``w``}:
                * ``s``: **slope**---the slope of the transition region.
                * ``w``: **width**---the width of the transition region.

        Other Parameters:

            kwargs: Keyword arguments relating to domain, range, discretization, and exceptional points:
                See :class:`.Literal`.
            domain: Default:  the domain that covers the transition region---where the range is (0,1).
                You may well wish to override this, extending the side of interest---probably the ``True`` side.
            origin:  Default: ``c``.

        Warning:
            It's fairly easy to set a moderately shallow ``slope`` and get a huge default ``domain``.
            It's wise to set ``domain`` manually if your ``slope < 1``.

        Caution:
            * The :meth:`.Inequality.t` method has been overridden to give the extremes of ``range``
              according to the ``sense`` of the inequality.  Consequently, there is no ``v`` for which it
              automatically returns ``elsewhere``.
            * Making ``range[1] > range[0]`` will reverse the ``sense`` of the inequality.

            """
        super().__init__(**kwargs)
        self.c = c
        if unit == "w":  # width
            if not (shape > 0):
                raise ValueError("width must be greater than 0.")
            half_width = shape / 2
        else:  # slope
            if not (shape > 0):
                raise ValueError("slope must be greater than 0.")
            half_width = .5 / shape
        self.sense = -1 if sense == "<" else 1
        self.half_width = half_width
        if self.d is None:  # if undefined:  s = (0,1)
            self.d = Domain((c - half_width, c + half_width))
        if self.origin is None:
            self.origin = c

    def __str__(self):
        s = super().__str__()
        sense = "<" if (self.sense < 0) else ">"
        return str(f"Inequality: x {sense} {self.c:,.4g} ± {self.half_width:,.4g}, "
                   f"slope: {1 / (2 * self.half_width):,.4g} \n {s}")

    def _sample(self, v: np.ndarray) -> np.ndarray:
        """Returns the truth for every value in ``v``."""
        # s = 1 / (1 + np.exp(-self.sense * self.a * (v - self.c)))
        # s = s * (self.range[1] - self.range[0]) + self.range[0]

        left = np.zeros_like(v)
        right = np.ones_like(v)
        slope = self.sense / (2 * self.half_width)
        middle = (v - self.c + self.sense * self.half_width) * slope
        # middle = (-.5 * v) / self.half_width + (.5 + (.5 * self.c) / self.half_width)
        slm = np.fmax(left, middle)
        s = np.fmin(slm, right)
        s = s * (self.range[1] - self.range[0]) + self.range[0]
        return s

    def t(self, v: float) -> float:
        """Returns the truth of any real number, ``v``.

        To follow the behavior expected of an inequality, it returns an extreme of ``range`` according
        to the ``sense`` of the inequality (``">"`` or ``"<"``).    I.e., it's a shelf function.
        N.B.: it does not automatically return ``elsewhere`` for any ``v``. """
        if v < self.d[0]:
            return self.range[0] if self.sense > 0 else self.range[1]
        elif v > self.d[1]:
            return self.range[1] if self.sense > 0 else self.range[0]
        else:
            return super().t(v)


class Sigmoid(Literal):
    """Describes a fuzzy inequality as a sigmoid curve.

    The "``sense``" of an inequality can be either ``">"`` or ``"<"``.
    A Sigmoid, then, is a way of talking about an inequality as: "a value is ``sense c`` to within ``width``", or
    "with confidence ``slope``"."""

    def __init__(self, sense: str, c: float, shape: float = 1, unit: str = "s", **kwargs):
        """
        Args:
            c:  The center of the transition, at which the truth is .5.  It need not be in the ``domain``.
            sense: Either ``">"`` (more suitable on the right---when greater than ``c``) or ``"<"``
                (more suitable on the left---when less than ``c``).
            shape: A parameter affecting the shape of the middle.  Its significance depends on ``unit`` below.
                Default: 1, unit="s"---a slope of 1 at ``c``.
            unit: The meaning of ``shape``.  It can have two values: {``s``, ``w``}:
                * ``s``: **slope**---the  slope at ``c``.
                * ``w``: **width**---the width of the transition region, where the truth varies on [.1, .9].

        Other Parameters:

            kwargs: Keyword arguments relating to domain, range, discretization, and exceptional points:
                See :class:`.Literal`.
            domain: Default:  the domain that covers the range on [.001, .999].
            origin:  Default: ``c``.

        Warning:
            It's fairly easy to set a moderately shallow ``slope`` and get a huge default ``domain``.
            It's wise to set ``domain`` manually if your ``slope < 1``.

        Caution:
            * The :meth:`.Sigmoid.t` method has been overridden to give the extremes of ``range``
              according to the ``sense`` of the inequality.  Consequently, there is no ``v`` for which it
              automatically returns ``elsewhere``.
            * Making ``range[1] > range[0]`` will reverse the ``sense`` of the inequality.

            """
        super().__init__(**kwargs)
        self.c = c
        if unit == "w":  # width
            if not (shape > 0):
                raise ValueError("width must be greater than 0.")
            a = 4.39444915467244 / shape
        else:  # slope
            if not (shape > 0):
                raise ValueError("slope must be greater than 0.")
            a = 4 * shape
        self.sense = -1 if sense == "<" else 1
        self.a = a
        if self.d is None:  # if undefined:  s = [.001, .999]
            w = 6.906754778648554 / self.a
            self.d = Domain((c - w, c + w))
        if self.origin is None:
            self.origin = c

    def __str__(self):
        s = super().__str__()
        sense = "<" if (self.sense < 0) else ">"
        return str(f"sigmoid: x {sense} {self.c:,.4g} ± {4.39444915467244 / self.a:,.4g} "
                   f"---width: {4.39444915467244 / self.a:,.4g}, slope: {self.a / 4:,.4g} \n {s}")

    def _sample(self, v: np.ndarray) -> np.ndarray:
        """Returns the truth for every value in ``v``."""
        s = 1 / (1 + np.exp(-self.sense * self.a * (v - self.c)))
        s = s * (self.range[1] - self.range[0]) + self.range[0]
        return s

    def t(self, v: float) -> float:
        """Returns the truth of any real number, ``v``.

        To follow the behavior expected of an inequality, it returns an extreme of ``range`` according
        to the ``sense`` of the inequality (``">"`` or ``"<"``).    I.e., it's a shelf function.
        N.B.: it does not automatically return ``elsewhere`` for any ``v``. """
        if v < self.d[0]:
            return self.range[0] if self.sense > 0 else self.range[1]
        elif v > self.d[1]:
            return self.range[1] if self.sense > 0 else self.range[0]
        else:
            return super().t(v)


class CPoints(Literal):
    """A fuzzy number defined as knots for interpolation.

    Note the similarity of its interface to that of :class:`DPoints`---it is easy to convert the calls.
    """

    def __init__(self, knots: Iterable[Tuple[float, float]], interp: Union[str, Tuple] = None,
                 expected_range: Tuple[float, float] = (0, 1), map: str = "lin", **kwargs) -> None:
        """
        Args:
            knots:  A collection of (value, truth) pairs---knots to be interpolated between, producing t(v).
                Values need not be unique (discontinuities are all right) and all "truths" are not checked for validity
                since they will be scaled to (keyword argument) ``range``---i.e., you can use (value, raw data) pairs
                to represent empirical (or imaginary) information.
            interp: Interpolator type used to construct the t(v) function.  See :class:`.Interpolator`.  Some
                interpolators may define curves that stray outside [0,1], but these will be clipped automatically.
                Default: ``linear``.  See :class:`.Interpolator`.
            expected_range: The expected range of input data.  Default: [0,1].

                * The default allows direct entry of data.
                * If ``expected_range==None``, the input will be normalized to fill ``range``.

                This provides a convenient way to make empirical data into a fuzzy number.

            map:  How the ``expected_range`` is mapped to ``range``---"lin", "log", or "exp".  See :meth:`._scale`.

        Other Parameters:

            kwargs: Keyword arguments relating to domain, range, discretization, and exceptional points:
                See :class:`.Literal`.
            domain:  Default: the domain that just covers the knots.

            """
        super().__init__(**kwargs)
        # TODO: make a better checker --> helper
        p = np.atleast_2d(np.array(knots))
        if (self.__class__ is CPoints) and (len(p) < 2):
            raise ValueError("CPoints needs at least two knots.")
        p = p[p[:, 0].argsort()]
        self.points_v = p[:, 0]
        self.points_t = p[:, 1]
        self.points_t = _scale(self.points_t, expected_range, self.range, map)
        if self.d is None:
            self.d = Domain((np.min(self.points_v), np.max(self.points_v)))
        if self.origin is None:
            self.origin = self.d.center()
        if interp is None:
            self.interp = default_interpolator
        else:
            self.interp = Interpolator(kind=interp)

    def __str__(self):
        s = super().__str__()
        i = self.interp.__str__()
        knots = np.dstack((self.points_v, self.points_t))[0]
        return str(f"CPoints: interpolation by {i} of knots:\n  {knots} \n {s}")

    def _sample(self, v: np.ndarray) -> np.ndarray:
        """Returns the truth for every value in ``v``."""
        return self.interp.interpolate(v, self.points_v, self.points_t)


class DPoints(CPoints):
    """A fuzzy number defined as discrete points.

    Note the similarity of its interface to that of :class:`CPoints`---it is easy to convert the calls."""

    def __init__(self, points: Iterable[Tuple[float, float]],
                 expected_range: Tuple[float, float] = (0, 1), map: str = "lin", **kwargs) -> None:
        """Args:

            points:  A collection of (value, truth) pairs---discrete points.
                All values must be unique and all truths will be scaled to (keyword argument) ``range``.
            expected_range: The expected range of input data.  Default: [0,1].

                * The default allows direct entry of data.
                * If ``expected_range==None``, the input will be normalized to fill ``range``.

                This provides a convenient way to make empirical data into a fuzzy number.

        Other Parameters:

            kwargs: Keyword arguments relating to domain, range, discretization, and exceptional points:
                See :class:`.Literal`."""
        if (self.__class__ is DPoints) and (len(points) < 1):
            raise ValueError("DPoints needs at least one point to have meaning.  Maybe you want a Truthy.")
        super().__init__(points, None, expected_range, map, **kwargs)
        self.xv, self.xt = self.points_v, self.points_t
        self.d, self.points_v, self.points_t = None, None, None

    def __str__(self):
        if self.xv is None:
            return str(f"DPoints: empty = {self.e} everywhere.")
        n = len(self.xv)
        points = np.dstack((self.xv, self.xt))[0]
        return str(f"DPoints: {n} discrete points: \n {points}, elsewhere: {self.e}.")

    def _sample(self, v: np.ndarray) -> np.ndarray:
        """Returns the truth for every value in ``v``."""
        return np.full_like(v, self.e)


class Exactly(DPoints):
    """A fuzzy number exactly equivalent to a crisp number.

    This enables crisp numbers to be used in fuzzy calculations.  (Fuzzy math operators conveniently promote
    number operands to :class:`.Exactly` objects.)

    """

    def __init__(self, value: float) -> None:
        """
        Args:
            value:  The value where truth is 1---it is 0 at all other points.
            """
        super().__init__(points=(value, 1), elsewhere=0)

    def __str__(self):
        return str(f"Exactly {self.xv[0]}")


class Truthy(_Numerical):
    """A fuzzy number equivalent to a Truth.

    This enables :class:`.Truth` to be used in fuzzy calculations.  (Fuzzy logic operators conveniently promote
    number operands to :class:`.Truthy` objects.)
    """

    def __init__(self, t: Union[Truth, float, int, bool]) -> None:
        """
        Args:
            t:  The truth for all values.
            """
        if not Truth.is_valid(Truth(t)):
            raise ValueError("A truth must be on [0,1]")
        t = float(t)
        super().__init__(None, 0, None, None, None, None, t)

    def __str__(self):
        # s = super().__str__()
        return str(f"Truthy: truth = {self.e} everywhere.")
