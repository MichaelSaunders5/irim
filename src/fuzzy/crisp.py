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

from fuzzy.truth import Truth



class Interpolator:
    """Some interpolation routines and data indicating which to use and how."""

    def __init__(self, **kwargs):
        """Parameters for the :meth:`interpolate` method.

        Kwargs:
            ``key=value`` pairs in one of these combinations:

            * type="linear": simple linear interpolation between adjacent knots.

        """
        if not kwargs:
            kwargs = {'type': "linear"}
        self.parameters = kwargs

    def interpolate(self, value: Union[np.ndarray, float], v: np.ndarray, s: np.ndarray) -> Union[np.ndarray, float]:
        """Performs an interpolation based on ``self.parameters``.

        Args:
            v: The values for the knots taken as (v,s) pairs.
            s: The suitabilities for the knots taken as (v,s) pairs.
            value: The v for which to obtain an interpolated s.

        Returns:
            The interpolated suitability at
        """
        satv = 0
        if self.parameters['type'] == "linear":
            satv = np.interp(value, v, s)  # This is only linear.  Good enough?
        # cs = CubicSpline(self.v, self.s, bc_type="not-a-knot", extrapolate=None)
        # satv = cs(value)
        # TODO: this is where the interpolation happens
        return satv


default_interpolator = Interpolator(type="linear")
"""The one to use, usually"""


class Map:
    """A function mapping a fuzzy number's suitability onto real numbers.

    The most familiar way of using a fuzzy number is to "defuzzify" it to obtain a single, definite, crisp number.
    That is accomplished by the :meth:`.Value.crisp` method.  Another way is to use a function to map its
    suitability onto the crisp numbers.  I.e., a fuzzy number (a :class:`Value` object) is :math:`s(v)`,
    a :class:`Map` object (obtainable from the :class:`Value` by the :meth:`.map` method) is :math:`f(v) = g(s(v))`.

    I can't provide every possible :math:`g(x)`, but this is a good start:  a simple scaling onto a given ``range``,
    linearly, or with the option to invert a log mapping by ``exponential=True``.  This function-as-an-object
    with a :meth:`.Map.get` method can be used in whatever expressions you need.  I.e., since, internally, a fuzzy
    number *is* a function, it is sometimes useful to use it *as* a function.

    Consider the utility of an input variable that has been mapped onto [0,1], modified by fuzzy work, and then
    reconstituted into its input units by a :class:`Map`.  Consider also the possibility of a variable conceived
    in relative units, within the fuzzy world, as a :class:`.Value`, then reified into practical units
    by a :class:`Map`.

    Example:
        | my_map = my_value.map(range=(-100,100), exponential = False, resolution = .001, interp = "cubic")
        | y = my_map(x)
    """

    def __init__(self, numerical: 'Numerical', range: Tuple[float, float], map: str = "lin",
                 interp: Interpolator = None):
        """I expect that the constructor will only ever be called by the :meth:`.Value.map` method.

        Args:

            numerical:  The numerical representation of a fuzzy number.
            interp:  The interpolator used on the numerical representation.

        Other Parameters:
            range, map:  relate to mapping fuzzy units.  See :meth:`.Truth.scale`."""
        self.numerical = numerical
        self.range = range
        self.map = map
        if interp is None:
            self.interp = default_interpolator
        else:
            self.interp = interp
        # I think only linear interpolation never strays from the valid range:
        if self.interp.parameters['type'] == "linear":
            self.clip = False
        else:
            self.clip = True

    def __call__(self, x: float) -> float:
        """Performs function mapped from the :math:`s(v)` of a fuzzy number.

        The fuzzy number used and details of the mapping are determined when the :class:`Map` object was created
        by :meth:`.Value.map`.

        Args:
            x: The independent variable.

        Returns:
            The dependent variable.
        """
        return Truth(self.numerical.suitability(x, self.interp)).to_float(self.range, self.map, self.clip)



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
