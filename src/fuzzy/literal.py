"""creates :class:`.FuzzyNumber`\\ s defined by parameters.

The representation of fuzzy numbers as :math:`t(v)` defined for all real numbers is extremely flexible and
allows for infinite possibilities.   Users need a practical mechanism for creating fuzzy numbers to work with.
Ideally, they should be able to describe their ideas with a few intuitive parameters.  Then they may combine them with
operators into expressions and built up complex, highly structured fuzzy numbers to model real-life situations
and acts of judgement.  There are two ways of creating literal fuzzy numbers via intuitive parameters:

    * The general-purpose classes provided in this module, descended from :class:`.Literal`.
    * Classes that you make yourself, designed to represent things from your own field of
      discourse, operating via inheritance or composition.  I.e., they could be descended from :class:`.Literal`,
      or its descendants, or assemble within themselves expressions of literals.

The basic literals provided here fall into a few categories:

    * Approximate numbers:  :class:`.Triangle`, :class:`.Trapezoid`, :class:`.Cauchy`, :class:`.Gauss`,
      and :class:`.Bell`.
    * Inequalities:  :class:`.Inequality`, and :class:`.Sigmoid`.
    * Empirical data or a desired shape:  :class:`.CPoints`.
    * Discrete empirical data or desired points:  :class:`.DPoints`.
    * Fuzzy versions of other types:  :class:`.Exactly`, and :class:`.Truthy`.

Approximate values are the most obvious application of fuzzy numbers, but the real power comes with representing more
sophisticated ideas in a number.  Fuzzy inequalities allow one to reason about the grey areas of categories and
decisions.  They are like one-sided approximate numbers.    The class :class:`.CPoints` allows one to make a fuzzy
number out of empirical data.  Perhaps there are only a few points of :math:`t(v)` you are certain about---this class
uses various kinds of interpolation to make them into a continuous function.  This can also be used to create functions
of whatever shape you imagine.

In all of the above, it is possible to define a continuous function with a few parameters.  With a few more, the
domain can be restricted to a finite set of discrete points, e.g., if the truth of a number can be
defined by a function but only integer values are sensible.  This can be done by defining uniformly-spaced values or
by giving an explicit list of values or both. The class :class:`.DPoints` allows you to input an explicit list
of :math:`(t,v)` pairs, similar to :class:`.CPoints`, but without connecting the points.

Finally, :class:`.Exactly`, and :class:`.Truthy` allow
one to bring crisp numbers and fuzzy truths into the world of fuzzy mathematics.  They may be used directly, but they
are also used in the system of automatic type promotions that makes it possible to mix together ``float``, ``int``,
``bool``, :class:`.Truth` and :class:`.FuzzyNumber` objects in expressions.

More powerful than any of these, it is possible to create new types of fuzzy number.  You can, with some parameters,
create literals that describe objects and concepts that you are modeling in your reasoning.  Since this is being done
in a Python program, the parameters needn't be numbers---they could be objects of any type, instances of your own
classes, perhaps.  By the same token, the definition of :math:`t(v)` is a Python program---an algorithm---and so can
be much more complex than an ordinary mathematical function.

To create new types of literal, you need only subclass :meth:`.Literal` and implement its :meth:`.Literal._sample`
method, adding whatever parameters you need to describe your concept of the number.  You might do this to represent
some real phenomena that vary between established limits.  For example, there is a very good model of sensory
dissonance---the unpleasantness listeners report when hearing combinations of sounds.  One could create a
literal that indicated dissonance vs. pitch interval (as truth vs. value), given the parameters: central pitch, and
two sound spectra objects.  The calculation of the resulting dissonance curve depends on a numerical algorithm.
It is not an analytical formula, but it can be easily represented by a Python method.  You define custom parameters
by implementing your class's ``__init__(self, **kwargs)`` method.  Begin it with a
call ``super().__init__(**kwargs)``, which handles the basic keywords and sets up the basic data members.

You should observe two cautions when subclassing :meth:`.Literal`.  They have to do with the domain and origin
members, ``self.d`` and ``self.origin``, which users may or may not set with the ``domain=`` and ``origin=`` keyword
arguments---by default, these are set to ``None`` which may cause errors---so you should define them if the user did
not.  Set ``self.d`` to the domain where the number is sensible and the function varies interestingly, say, over the
truth range [.001, .999].  But: beware of vast domains that might require a vast number of samples.  If ``self.origin``
has not been set, set it to the value where the function peaks, inflects, or has some other critical feature.
Origin is only used in cases where uniform discretization is called for.

How It Works
============

In the larger scheme of the algorithm described in the last module (:mod:`.fuzzy.number`), :meth:`._get_numerical`
calls eventually reach literal objects ("leaves" on the "tree" of an expression).  When this method is
called on a :class:`.Literal`, it creates a numerical representation of itself by choosing the values of
sample points and calling its own ``_sample`` method with them.  If discretization has been called for, it does the
same sampling for the given values and returns a :class:`._Numerical` with them as its explicit points.  In the case
of :class:`.DPoints` no sampling is necessary, as the points have been given literally.  The class :class:`.Exactly`
does this for a single point, and :class:`.Truthy` simply sets the "elsewhere" default truth.

The next module, (:mod:`.fuzzy.operation`), describes the operators that combine literals into expressions.
"""

from __future__ import annotations

from abc import abstractmethod
from math import log, exp, sqrt, pi, isfinite
from typing import Union, Tuple

import numpy as np
from scipy.stats import norm as gauss

from fuzzy.crisp import Interpolator
from fuzzy.number import Domain, _Numerical, default_interpolator, default_sampling_method, _guard
from fuzzy.operator import Operator
from fuzzy.truth import Truth, default_threshold


def _scale(data: Union[np.ndarray, Iterable[float], float],
           expected_range: Tuple[float, float] = (0, 1), intended_range: Tuple[float, float] = (0, 1),
           map: str = "lin") -> Union[np.ndarray, float]:
    """A helper for scaling the input to literals.

    This is intended for scaling truth-valued parameters onto some sub-range of [0,1].  With the defaults left as
    they are, it does nothing.  If you have real-world data in some units, you might need to scale it onto the range
    of truth to represent that data as a literal fuzzy number.  Handily, ``expected_range = None`` will normalize
    the input to fill ``intended_range``.

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
            for a ``float``.  The data is scaled from ``expected_range`` to ``intended_range`` according to ``map``."""
    t = np.array(data)
    if expected_range is None:
        expected_range = (np.min(data), np.max(data))
    if intended_range is None:
        if isinstance(data, float):
            return default_threshold
        else:
            return np.full((len(data)), default_threshold)
    t = Truth.scale(t, "in", expected_range, map, False)
    t = Truth.scale(t, "out", intended_range, "lin", True)
    return t


def _check_points(points: np.ndarray) -> (np.ndarray, np.ndarray):
    """Checks a set of (v,t) points for validity and returns them as separate v,t arrays.
    For validity, all truths must be on [0,1] and all values must be unique.  Exceptions are raised otherwise."""
    xv = np.atleast_2d(points)[:, 0]
    xt = np.atleast_2d(points)[:, 1]
    if (np.max(xt) > 1) or (np.min(xt) < 0):
        too_high = xv[np.where(xt > 1)]
        too_low = xv[np.where(xt < 0)]
        raise ValueError(f"Truths must be on [0,1].  "
                         f"Yours are too high at v=={too_high}; and too low at v=={too_low}.")
    all_values, counts = np.unique(xv, return_counts=True)
    dup_values = all_values[np.where(counts > 1)]
    if len(dup_values) > 0:
        raise ValueError(f"You cannot define two truths for the same value.  "
                         f"You have more than one truth at v={dup_values}.")
    return xv, xt


class Literal(Operator):  # user only implements _sample
    """An abstract base for fuzzy numbers defined by a mathematical function given as a Python method.

    To create new types of fuzzy number, one may subclass this class and implement its :meth:`.Literal._sample`
    method, as has been done below to represent several varieties of approximate numbers.

    The fuzzy number may be continuous (the default) or discretized.  This is useful if the truth may be
    easily defined by a mathematical function, but only valid for discrete values.
    There are three ways of doing this:

        * For an arbitrary collection of values, given by ``discrete=``.
        * For a set of uniformly-spaced values, by setting ``uniform=True`` and,
          optionally, ``step=`` and ``origin=``.
        * Both of the above together---the discrete points will be the union of the two sets.

    Subclasses must implement :meth:`.Literal._sample` to be the :math:`t(v)` function that defines the fuzzy number.
    Its input and output are :class:`numpy.ndarray`\\ s, so it should be done using vectorized Numpy mathematics
    (i.e., with `*ufunc*\\ s <https://numpy.org/doc/stable/reference/ufuncs.html#available-ufuncs>`_).  This
    method can depend on members of the object that you initialize by defining the ``__init__(self, **kwargs)``
    method of your subclass.  Begin it with a call to ``super().__init__(**kwargs)``.
    """

    def __init__(self, **kwargs):
        """
        Keyword Arguments:

            domain= (Tuple[float, float]):
                The domain of values over which :meth:`.Literal._sample` defines a continuous function
                (i.e., not considering the exceptional points). The full, continuous domain will be
                used, unless it is discretized by defining ``discrete=`` or by setting ``uniform=True``.
                Values will return the ``elsewhere`` truth if they are neither explicitly defined as discrete points
                nor on the (possibly discretized) domain.  If only exceptional points are intended (as
                in :class:`.DPoints`), ``domain`` *should be* ``None``.  Default: ``None``.
                N.B.: if a continuous function is intended and users do not set the domain, subclasses should
                choose a default---probably the domain where the truth has most of its variation (say, 99.9%).
            range= (Tuple[float, float]):
                The extremes of truth that will be reached.  Default: [0,1].  If ``range[0] > range[1]`` the sense of
                the scaling is reversed and peaks become valleys (so, e.g., an approximate number becomes an
                *exclusion* of values).
            elsewhere= (float):
                The truth returned for values that are otherwise undefined.  Default: 0.
            discrete= (Iterable[float]):
                If defined, the domain is restricted to these values, given explicitly
                (and to any defined by ``uniform``).
            uniform= (bool):
                If ``True``, the domain is restricted to a set of uniformly-spaced values, according
                to ``step`` and ``origin`` (and to any defined by ``discrete``).
            step= (float):
                Uniform points are spaced at these intervals over ``domain``.  Default: 1.
            origin= (float):
                Uniform points will be measured from ``origin``.  It needn't be in ``domain``.
                Default:  ``None``---the subclass must choose a default,
                and would do well to use the value where the function "peaks" (i.e., reaches ``range[1]``).
            points= (Iterable[Tuple[float, float]]):
                  A container of (v, t) pairs to be added to the continuous definition.  Default: ``None``.

        Caution:
            Authors of subclasses, take note.
            Two of the keyword arguments, ``domain`` and ``origin`` are set to ``None`` by default.  This will cause
            errors unless the subclass replaces them with a default.  The superclass (here) can't
            because it doesn't know the intentions of the subclass.  The best practices are as follows:

                * If ``domain == None``:  set it to the domain where the function varies interestingly,
                  e.g., over the truth range [.001, .999].
                * If ``origin == None``:  set it to the value where the function peaks, inflects, or has some other
                  critical feature.
                """
        # domain-related:
        super().__init__(None, **kwargs)
        domain = kwargs.get('domain', None)  # Where _sample is defined, even if cd will be None due to discretization
        self.d = None if domain is None else Domain(domain) if isinstance(domain, tuple) else domain
        # If None, subclasses must supply a default
        self.range = kwargs.get('range', (0, 1))
        if not (Truth.is_valid(self.range[0]) and Truth.is_valid(self.range[1])):
            raise ValueError("Truths like those in ``range`` must be on [0,1].")
        # discretization-related:
        self.discrete = kwargs.get('discrete', None)
        self.uniform = kwargs.get('uniform', False)
        self.step = kwargs.get('step', 1)
        self.origin = kwargs.get('origin', None)  # If None, subclasses must supply a default
        if self.discrete is not None:
            self.discrete = np.array(self.discrete)
        if self.uniform:
            if self.step <= 0:
                raise ValueError("Step must be > 0.")
        # exceptional points
        points = kwargs.get('points', None)
        if (points is not None) and (len(points) != 0):
            xv, xt = _check_points(points)
            self.xv, self.xt = xv, xt
        else:
            self.xv, self.xt = None, None  # They can hold exceptional points if they are assigned.

    def __str__(self):
        s = str(f"domain: {self.d}, range: ({self.range[0]:.3g}, {self.range[1]:.3g}), "
                f"elsewhere: {self.e:.4g}")
        if self.uniform or self.discrete is not None:
            i = str(f"\n discretization: ")
        else:
            i = str(f"no discretization. ")
        u, d, c = "", "", ""
        if self.discrete is not None:
            d = str(f"at values: {self.discrete}")
        if self.uniform and self.discrete is not None:
            c = str(f" and at ")
        if self.uniform:
            u = str(f"uniform {self.step:g} steps from {self.origin:g}")
        if self.xv is None:
            x = str(f"no exceptional points.")
        else:
            x = str(f"exceptional points: \n {np.dstack((self.xv, self.xt))[0]}")
        return str(f"{s}; {i}{d}{c}{u}; {x}")

    @abstractmethod
    def _sample(self, v: np.ndarray) -> np.ndarray:
        """This is where you implement the t(v) function that defines your fuzzy number.
        It should return a truth on [0,1] for any real number in the domain, ``self.d``,
        using vectorized Numpy functions.

        You can call ``super()._sample(v) for values outside the defined domain, ``self.d``---or simply return
        the "elsewhere" value, ``self.e``, for them.

        If you intend for your number to have a continuous domain you should check to see if the value of ``self.d``
        is ``None``.   This indicates that the caller has not requested one.  The best choice in this case is
        probably the domain over which the truth has most of its variation (say, 99.9%), or is above some threshold
        of significance (say, .001).  You might even consider using :class:`.Domain.intersection` with self.d to
        restrict it to whatever you consider reasonable.

        If ``self.uniform==True`` you should also make sure that ``self.origin`` is set (and not ``None``).
        It should probably be the point where the function peaks or has some other critical point."""

    def _get_domain(self) -> Domain:
        return self.d

    def _get_numerical(self, precision: int, allowed_domain: Domain = None) -> _Numerical:
        """Returns a numerical version of itself.  This is where the Literal object samples itself, and where
        the details of discretization are determined."""
        domain = self.d  # The domain on which _sample() is defined (and a bit beyond, we can hope).
        discrete_points = self.discrete
        if allowed_domain is not None:
            domain = domain.intersection(allowed_domain)
        if (domain is None) or (precision <= 0):
            num = _Numerical(None, 0, None, None, self.xv, self.xt, self.e)
            if allowed_domain is not None:
                num = _Numerical._impose_domain(num, allowed_domain)
            return num
        xv, xt = self.xv, self.xt  # explicitly-defined exceptional points
        if discrete_points is not None:  # Exclude discrete points outside the continuous domain.
            if domain is None:
                discrete_points = None
            else:
                discrete_points = discrete_points[(discrete_points >= domain[0]) & (discrete_points <= domain[1])]
                discrete_points = None if len(discrete_points) == 0 else discrete_points
        discrete = (discrete_points is not None) and (len(discrete_points) > 0)
        if discrete or self.uniform:  # Discretization has been called for.
            cd, cn, cv, ct = None, 0, None, None
            xv = np.empty(0)
            if self.uniform:  # Find the uniformly-spaced sample points inside domain.
                n0 = ceil((domain[0] - self.origin) / self.step)  # Extreme sample numbers relative to origin.
                n1 = floor((domain[1] - self.origin) / self.step)
                v0 = self.origin + n0 * self.step  # extreme values
                v1 = self.origin + n1 * self.step
                xv = np.linspace(v0, v1, n1 - n0 + 1)  # Calculate the values of the points.
            if discrete:
                xv = np.unique(np.concatenate((xv, discrete_points)))
            xv = np.setdiff1d(xv, self.xv, assume_unique=True)  # Remove from xv any elements that are in self.xv
            xt = self._sample(xv)  # Sample the Literal at the exceptional points.
            xt = _guard(xt)
            if self.xv is not None:
                xv = np.concatenate((xv, self.xv), axis=None)
                xt = np.concatenate((xt, self.xt), axis=None)
        else:  # A continuous function has been called for. Find cv cleverly.
            cd = domain
            cn = 3 if (precision < 3) else precision  # I.e., if it's 1 or 2, make it 3.
            # below: 2 to guard the secant & interval from exploding; 3 allows possible linear extrapolation.
            if default_sampling_method == "Chebyshev":  # Find the values to sample at, Chebyshev or uniform.
                m = cn + 2
                cv = np.arange(1, m + 1, 1)
                cv = .5 * cd.span() * (-np.cos((cv - .5) * pi / m)) / np.cos(1.5 * pi / m) + cd.center()
            else:
                interval = cd.span() / (cn - 1)
                cv = np.linspace(cd[0] - interval, cd[1] + interval, num=(cn + 2))
            ct = self._sample(cv)  # Sample the Literal on the continuous domain.
            if (len(ct) >= 4) and (not isfinite(ct[-2])):  # because Akima interp. produces two NaNs at the end.
                ct[-2] = 2 * ct[-3] - ct[-4]
            ct = np.concatenate((ct[0:1], _guard(ct[1:-1]), [ct[-1]]))  # Make the pts. on domain safe.
            # Literals aren't guaranteed at their guard points (beyond self.d).  So, if they're funny...
            # linear extrapolate to them---excursions beyond [0,1] allowed here---to preserve the *shape* on domain:
            if not isfinite(ct[0]):
                ct[0] = 2 * ct[1] - ct[2]
            if not isfinite(ct[-1]):
                ct[-1] = 2 * ct[-2] - ct[-3]
        return _Numerical._impose_domain(_Numerical(cd, cn, cv, ct, xv, xt, self.e), allowed_domain)

    def t(self, v: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Returns the truth of given values.

        Args:
            v: a ``float`` value, or an array of them.

        Returns:
            The truth(s) of the given value(s)."""
        a = isinstance(v, np.ndarray)
        v = np.atleast_1d(v)
        e = np.full_like(v, self.e)
        c = self._sample(v)
        if self.d is None:
            c_e = e
        else:
            c_e = np.where((v < self.d[0]) | (v > self.d[1]), e, c)  # in an Exactly, d is None.
        for i, value in enumerate(v):
            j = np.where(value == self.xv)[0]
            if j is None or len(j) == 0:
                continue
            c_e[i] = self.xt[j]
            c_e = _guard(c_e)
        return c_e if a else c_e[0]

    # Because of the inheritance structure, Literals must be Operators, but they don't really need implementations
    # of the superclass's abstract methods ``_operate`` and ``_op``.

    def _operate(self, precision: int, allowed_domain: Domain = None):
        pass

    def _op(self, *args) -> Union[np.ndarray, float]:
        pass

    def d_op(self, inv: bool, *d: Domain) -> Domain:
        """:class:`.Literal`\\ s simply return the intersection of continuous domain for which they are defined,
        with that of the called-for domain.

        They are never called for an inverse domain, so ``inv`` is irrelevant."""
        if d[0] is None:
            return self.d
        return self.d.intersection(d[0])


class Triangle(Literal):
    """Describes a fuzzy number as a triangular function."""

    def __init__(self, a: float, b: float, c: float, **kwargs):
        """
        Args:
            a, b, c:  The minimum, preferred, and maximum values.  The function is piecewise linear between
                these points.  The condition :math:`a \\le b \\le c` must hold
            range:  Default: (0,1).  The truths are:

                | ``range[0]`` at ``a`` and ``c``,
                | ``range[1]`` at ``b``.

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

    This is a way of talking about a number as "``c`` ± ``hwhm``" (half width at half maximum)."""

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

    This is a way of talking about a number as ``c`` ± ``hwhm`` (half width at half maximum)
    with confidence ``slope``, or with vagueness ``transition_width``."""

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

    I.e., :math:`t(v)` looks like a shelf function with a linear segment at the transition.
    The "``sense``" of an inequality can be either ``">"`` or ``"<"``.

    An :class:`.Inequality` is a way of talking about an inequality as: "a value is ``sense c`` to within ``width``",
    or "with confidence ``slope``".  At ``c``, the truth is .5.  Surrounding this, there is a transition region where
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
                You may well wish to override this, extending the side of interest---probably the ``True`` side
                (since, beyond the domain, :math:`t(v)` will revert to the "elsewhere" value, which is probably 0).
            origin:  Default: ``c``.

        Warning:
            It's fairly easy to set a moderately shallow ``slope`` and get a huge default ``domain``.
            It's wise to set ``domain`` manually if your ``slope < 1``.

        Caution:
            * The :meth:`.Inequality.t` method has been overridden to give the extremes of ``range``
              according to the ``sense`` of the inequality.  Consequently, there is no ``v`` for which it
              automatically returns ``elsewhere``.  E.g. ``Inequality(">", 0, domain=(-2, 2)).t(3)`` returns 1.
            * The other behaviors of the class cannot do this because they depend on an array of samples that cannot
              practically extend to infinity, so the fuzzy number has the "elsewhere" truth adjoining its domain on
              both sides---even though this does not agree with the concept of "inequality".
              E.g. ``Inequality(">", 0, domain=(-2, 2)).display(None, Domain((-8,8)))`` reveals a drop to 0
              to the right of ``v == 2``.
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
        left = np.zeros_like(v)
        right = np.ones_like(v)
        slope = self.sense / (2 * self.half_width)
        middle = (v - self.c + self.sense * self.half_width) * slope
        slm = np.fmax(left, middle)
        s = np.fmin(slm, right)
        s = s * (self.range[1] - self.range[0]) + self.range[0]
        return s

    def t(self, v: float) -> float:
        """Returns the truth of any real number, ``v``.

        To follow the behavior expected of an inequality, it returns an extreme of ``range`` according
        to the ``sense`` of the inequality (``">"`` or ``"<"``).    I.e., it's a shelf function.
        N.B.: there is no ``v`` for which it automatically returns ``elsewhere``. """
        if v < self.d[0]:
            return self.range[0] if self.sense > 0 else self.range[1]
        elif v > self.d[1]:
            return self.range[1] if self.sense > 0 else self.range[0]
        else:
            return super().t(v)


class Sigmoid(Literal):
    """Describes a fuzzy inequality as a sigmoid curve.


    I.e., :math:`t(v)` looks like a shelf function with an S-shaped segment at the transition.
    The "``sense``" of an inequality can be either ``">"`` or ``"<"``.

    A :class:`.Sigmoid`, then, is a way of talking about an inequality as: "a value is ``sense c``
    to within ``width``", or "with confidence ``slope``"."""

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
              automatically returns ``elsewhere``.  E.g. ``Sigmoid(">", 0, domain=(-2, 2)).t(3)`` returns 1.
            * The other behaviors of the class cannot do this because they depend on an array of samples that cannot
              practically extend to infinity, so the fuzzy number has the "elsewhere" truth adjoining its domain on
              both sides---even though this does not agree with the concept of "inequality".
              E.g. ``Sigmoid(">", 0, domain=(-2, 2)).display(None, Domain((-8,8)))`` reveals a drop to 0
              to the right of ``v == 2``.
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
        N.B.: there is no ``v`` for which it automatically returns ``elsewhere``. """
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
            knots:  A collection of (value, truth) pairs---knots to be interpolated between, producing :math:`t(v)`.
                Values need not be unique (discontinuities are all right) and all "truths" are not checked for validity
                since they will be scaled to (keyword argument) ``range``---i.e., you can use (value, raw data) pairs
                to represent empirical (or imaginary) information.
            interp: Interpolator type used to construct the t(v) function.  See :class:`.Interpolator`.  Some
                interpolators may define curves that stray outside [0,1], but these will be clipped automatically.
                Default: ``linear``.
            expected_range: The expected range of input data.  Default: [0,1].

                * The default allows direct entry of truths.
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

            points:  A collection of (value, truth) pairs---discrete points (an iterable of tuples).
                All values must be unique and all truths will be scaled to (keyword argument) ``range``.
                For all other values, the number will report its "elsewhere" truth.
            expected_range: The expected range of input data.  Default: [0,1].

                * The default allows direct entry of truths.
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


class Truthy(Literal):
    """A fuzzy number equivalent to a Truth.

    This enables :class:`.Truth` objects to be used in fuzzy calculations.  (Fuzzy logic operators conveniently promote
    number operands to :class:`.Truthy` objects.)
    """

    def __init__(self, t: Union[Truth, float, int, bool], **kwargs) -> None:
        """
        Args:
            t:  The truth for all values.
            """
        if not Truth.is_valid(Truth(t)):
            raise ValueError("A truth must be on [0,1]")
        super().__init__(**kwargs)
        self.e = float(t)

    def __str__(self):
        # s = super().__str__()
        return str(f"Truthy: truth = {self.e} everywhere.")

    def _sample(self, v: np.ndarray) -> np.ndarray:
        """Returns the truth for every value in ``v``."""
        return np.full_like(v, self.e)
