"""provides methods for defuzzification, interpolation, and mapping.

Subclasses of :class:`Crisper` provide a :meth:`.Crisper.defuzzify` method, which takes the numerical representation
of a fuzzy number (a :class:`numerical` object), and finds the crisp real number (a ``float``) that best represents it.
Since there are many ways this can be done, there are many subclasses.  There is a further complication in that the
numerical representation might include both a sampled continuous function and a set of discrete exceptional points.

The :class:`Interpolator` class provides a selection of interpolation algorithms in one handy interface.

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from scipy.interpolate import CubicSpline, barycentric_interpolate, \
    krogh_interpolate, pchip_interpolate, Akima1DInterpolator

from fuzzy.truth import Truth



class Interpolator:
    """Performs interpolations for arrays of points.

    Instances hold the parameter.  The work is done in the :meth:`.interpolate` method."""

    def __init__(self, kind: Union[str, tuple] = "linear"):
        """Args:
            kind: A string or tuple---((*left*), (*right*))---indicating the type of interpolation:

        +--------------------+-------------------------------------------------------------------+
        |  ``kind=``         | note                                                              |
        +====================+===================================================================+
        | ``"linear"``       | linear                                                            |
        +--------------------+-------------------------------------------------------------------+
        | ``"akima"``        | due to Hiroshi Akima                                              |
        +--------------------+-------------------------------------------------------------------+
        | ``"bary"``         | barycentric                                                       |
        +--------------------+-------------------------------------------------------------------+
        | ``"krogh"``        | due to Fred T. Krogh                                              |
        +--------------------+-------------------------------------------------------------------+
        |  ``"pchip"``       | piecewise cubic Hermite interpolating polynomial                  |
        +--------------------+-------------------------------------------------------------------+
        |  cubic splines     |                                                                   |
        +--------------------+-------------------------------------------------------------------+
        | ``"not-a-knot"``   | Last two segments at each end are the same polynomial.            |
        +--------------------+-------------------------------------------------------------------+
        | ``"periodic"``     | The first and last suitabilities must be identical.               |
        +--------------------+-------------------------------------------------------------------+
        | ``"clamped"``      | ((1, 0), (1, 0))                                                  |
        +--------------------+-------------------------------------------------------------------+
        | ``"natural"``      | ((2, 0), (2, 0))                                                  |
        +--------------------+-------------------------------------------------------------------+
        | ``((a,b), (c,d))`` | (*left, right*): (*derivative order* {1,2}, *derivative value*)   |
        +--------------------+-------------------------------------------------------------------+

        For ``"linear"``,
        see `Numpy <https://numpy.org/doc/stable/reference/generated/numpy.interp.html>`_.
        For all others,
        see `Scipy <https://docs.scipy.org/doc/scipy/reference/interpolate.html#univariate-interpolation>`_.

        """
        if kind is None:
            kind = "linear"
        if (kind == "linear" or "bary" or "krogh" or "pchip" or "akima" or
                "not-a-knot" or "periodic" or "clamped" or "natural") or isinstance(kind, tuple):
            self.kind = kind
        else:
            raise NameError(f"{kind} isn't a type of interpolator")

    def __str__(self):
        if isinstance(self.kind, tuple):
            l = "1st" if self.kind[0][0]==1 else "2nd"
            r = "1st" if self.kind[1][0]==1 else "2nd"
            return str(f"cubic interpolator with derivative boundary conditions "
                       f"({l} = {self.kind[0][1]}, {r} = {self.kind[1][1]})")
        else:
            return str(f"{self.kind} interpolator")

    def interpolate(self, value: Union[np.ndarray, float], v: np.ndarray, s: np.ndarray) -> Union[np.ndarray, float]:
        """Interpolates to perform a numerical function.

        The arrays ``v`` and ``s`` describe a function s(v).  In :class:`.Value.CPoints` this is used to construct
        a function from a few data points.  In :meth:`.Value.Numerical.suitability` it is used to find suitabilities
        for arbitrary values on the domain.

        Args:
            v: The values for the knots.
            s: The suitabilities for the knots.
            value: The v for which to obtain an interpolated s.

        Returns:
            The suitabilities for the given ``value``\\ s.
        """
        if self.kind == "linear":
            satv = np.interp(value, v, s)  # This is only linear.  Good enough?
        elif self.kind == "bary":
            satv = barycentric_interpolate(v, s, value)
        elif self.kind == "krogh":
            satv = krogh_interpolate(v, s, value)
        elif self.kind == "pchip":
            satv = pchip_interpolate(v, s, value)
        elif self.kind == "akima":  # This gives me an anomalous Truth.default_threshold for value == d[1].  Why?
            ai = Akima1DInterpolator(v, s)
            satv = ai(value)
        else:  # cubic, with ``kind`` giving boundary conditions:
            cs = CubicSpline(v, s, bc_type=self.kind, extrapolate=True)
            satv = cs(value)
        return satv


class Map:
    """A callable object: a function mapping a fuzzy number's suitability onto real numbers.

    The :class:`.NumericalMap` provides the same functionality, is less accurate, but may be more efficient, depending
    on the complexity of the expression.

    The most familiar way of using a fuzzy number is to "defuzzify" it to obtain a single, definite, crisp number.
    That is accomplished by the :meth:`.FuzzyNumber.crisp` method.  Another way is to use a function to map its
    truth onto the crisp numbers.  I.e., a :class:`.FuzzyNumber` object is :math:`t(v)`,
    a :class:`Map` object (obtainable from the :class:`FuzzyNumber` by the :meth:`.map` method)
    is :math:`f(v) = g(t(v))`.

    I can't provide every possible :math:`g(x)`, but this is a good start:  a simple scaling onto a given ``range``,
    linearly, or with the option to invert a log mapping by ``exponential=True``.  This function-as-a-callable-object
    can be used in whatever expressions you need.  I.e., since, internally, a fuzzy
    number *is* a function, it is sometimes useful to use it *as* a function.

    Consider the utility of an input variable that has been mapped onto [0,1], modified by fuzzy work, and then
    reconstituted into its input units by a :class:`Map`.  Consider also the possibility of a variable
    conceived in relative units, within the fuzzy world, as a :class:`.FuzzyNumber`, then reified into practical
    units by a :class:`Map`.

    Example:
        | my_map = my_value.map(range=(-100,100), map = "lin")
        | y = my_map(x)
    """

    def __init__(self, expression: 'Operator', range: Tuple[float, float], map: str = "lin"):
        """I expect that the constructor will only ever be called by the :meth:`.FuzzyNumber.map` method.

        Args:

            expression:  The operator at the root of a tree of :class:`.Operator`\\ s and :class:`.Literal`\\ s
                that describe a fuzzy number as an expression.

        Other Parameters:
            range, map:  relate to mapping fuzzy units.  See :meth:`.Truth.scale`."""
        self.expression = expression
        self.range = range
        self.map = map

    def __str__(self):
        if self.map == "exp" or "invlog":
            map = "exponatially"
        elif self.map == "log" or "invexp":
            map = "logarithmically"
        else:
            map = "linearly"
        return str(f"a callable object {map} mapping to {self.range} the suitability "
                   f"of the following fuzzy number:\n {self.expression}")

    def __call__(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Performs function mapped from the :math:`t(v)` of a fuzzy number.

        The fuzzy number used and details of the mapping are determined when the :class:`Map`
        object was created by :meth:`.FuzzyNumber.map`.

        Args:
            x: The independent variable (any float or an array of them).

        Returns:
            The dependent variable (a float or an array of them, corresponding to ``x``).
        """
        return  Truth.scale(self.expression.t(x), "out", self.range, self.map)


class NumericalMap:
    """A callable object: a function mapping a fuzzy number's suitability onto real numbers.

    A raw expression (as used by :class:`.Map`) is more accurate, but this type may be more efficient, depending
    on the complexity of the expression.

    The most familiar way of using a fuzzy number is to "defuzzify" it to obtain a single, definite, crisp number.
    That is accomplished by the :meth:`.FuzzyNumber.crisp` method.  Another way is to use a function to map its
    truth onto the crisp numbers.  I.e., a :class:`.FuzzyNumber` object is :math:`t(v)`,
    a :class:`NumericalMap` object (obtainable from the :class:`FuzzyNumber` by the :meth:`.numerical_map` method)
    is :math:`f(v) = g(t(v))`.

    I can't provide every possible :math:`g(x)`, but this is a good start:  a simple scaling onto a given ``range``,
    linearly, or with the option to invert a log mapping by ``exponential=True``.  This function-as-a-callable-object
    can be used in whatever expressions you need.  I.e., since, internally, a fuzzy
    number *is* a function, it is sometimes useful to use it *as* a function.

    Consider the utility of an input variable that has been mapped onto [0,1], modified by fuzzy work, and then
    reconstituted into its input units by a :class:`NumericalMap`.  Consider also the possibility of a variable
    conceived in relative units, within the fuzzy world, as a :class:`.FuzzyNumber`, then reified into practical
    units by a :class:`NumericalMap`.

    Example:
        | my_map = my_value.numerical_map(range=(-100,100), map = "lin", resolution = .001, interp = "cubic")
        | y = my_map(x)
    """

    def __init__(self, numerical: 'Numerical', range: Tuple[float, float], map: str = "lin",
                 interp: Interpolator = None):
        """I expect that the constructor will only ever be called by the :meth:`.FuzzyNumber.numerical_map` method.

        Args:

            numerical:  The numerical representation of a fuzzy number.
            interp:  The interpolator used on the numerical representation.

        Other Parameters:
            range, map:  relate to mapping fuzzy units.  See :meth:`.Truth.scale`."""
        self.numerical = numerical
        self.range = range
        self.map = map
        if interp is None:
            from fuzzy.number import default_interpolator
            self.interp = default_interpolator
        else:
            self.interp = interp
        # I think only linear interpolation never strays from the valid range:
        if self.interp.kind == "linear":
            self.clip = False
        else:
            self.clip = True

    def __str__(self):
        if self.map=="exp" or "invlog":
            map = "exponatially"
        elif self.map=="log" or "invexp":
            map = "logarithmically"
        else:
            map = "linearly"
        n = self.numerical.__str__()
        return str(f"a callable object {map} mapping to {self.range} the truth "
                   f"of the following fuzzy number:\n {n}")

    def __call__(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Performs function mapped from the :math:`t(v)` of a fuzzy number.

        The fuzzy number used and details of the mapping are determined when the :class:`NumericalMap`
        object was created by :meth:`.FuzzyNumber.numerical_map`.

        Args:
            x: The independent variable (any float or an array of them).

        Returns:
            The dependent variable (a float or an array of them, corresponding to ``x``).
        """
        return  Truth.scale(self.numerical.t(x, self.interp), "out", self.range, self.map, self.clip)


class Crisper(ABC):
    """Subclasses implement :meth:`.Crisper.defuzzify` to provide a selection of algorithms
    used by :meth:`.Value.crisp`."""

    @staticmethod
    @abstractmethod
    def defuzzify(numerical: 'Numerical') -> float:
        """Finds the crisp number equivalent to a fuzzy number represented numerically.

        Arg:
            numerical:  any fuzzy number---a numerical representation of a :class:`.Value`, obtainable
                via the :class:`.Value`'s :meth:`.Value.evaluate` method.

        Returns:
            The crisp equivalent of the fuzzy input."""


class MedMax(Crisper):
    """Provides median-of-the-maxima defuzzification."""

    @staticmethod
    def defuzzify(numerical: 'Numerical') -> float:
        # for each, v s and xp, find maxima.  find median of them.  xp might have two---chose the one where
        # s is greater, or, if they are equal...?  then choose the more suitable of the two.
        return (numerical.d[0] + numerical.d[1]) / 2  # do the real one later

    def __str__(self):
        return str(f"a median-of-the-maxima crisper---options?")