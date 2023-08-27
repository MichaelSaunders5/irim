"""Value
The :class:`.Value` class applies the idea of fuzzy truth to the representation of numbers.  A :class:`Value`
object is a function of truth vs. value.  We may think of the function as describing the suitability of each
possible value for some purpose.  E.g., we might describe room temperature as a symmetrical triangular function
on (68, 76)°F or (20, 24)°C.  In this way, we might state our opinions as "preference curves".  We might also
represent empirical knowledge as fuzzy numbers.  E.g., sensory dissonance vs. interval is a highly structured
function with many peaks and valleys across a wide domain, but all of this information can be encapsulated
in a single :class:`Value` object.

and...

"""

# Here are the classes for fuzzy value and arithmetic:
# Value --- base class;  Numerical --- "working" class of evaluation and defuzzification;
# Triangle, Trapezoid, Bell, Cauchy, Gauss, Points --- "atoms" defining a value;
# ValueNot, ValueAnd, ValueOr --- logic on values;
# Sum, Difference, Prod, Quotient, Focus, Abs, Inverse --- arithmetic on values.

from __future__ import annotations

from abc import ABC, abstractmethod
from math import floor, ceil
from typing import Tuple  # , ClassVar,

import numpy as np

from fuzzy.truth import Truth


class Interpolator:
    """Some interpolation routines and data to indicate which to use and how."""

    def __init__(self, **kwargs):
        """Parameters for :meth:`interpolate` to follow."""
        if not kwargs:
            kwargs = dict(type="linear")
        self.parameters = kwargs

    def interpolate(self, value: Union[np.ndarray, float], v: np.ndarray, s: np.ndarray) -> Union[np.ndarray, float]:
        satv = 0
        if self.parameters['type'] == "linear":
            satv = np.interp(value, v, s)  # This is only linear.  Good enough?
        # cs = CubicSpline(self.v, self.s, bc_type="not-a-knot", extrapolate=None)
        # satv = cs(value)
        # TODO: this is where the interpolation happens
        return satv


class Crisper(ABC):  # ?

    @staticmethod
    @abstractmethod
    def defuzzify(numerical: Numerical) -> float:
        pass


class MedMax:
    @staticmethod
    def defuzzify(numerical: Numerical) -> float:
        # for each, v s and xp, find maxima.  find median of them.  xp might have two---chose the one where
        # s is greater, or, if they are equal...?  then choose the more suitable of the two.
        print(numerical)
        return 0.


# @dataclass
class Value(ABC):
    """Represents a generally fuzzy real number (as a function of suitability (on [0,1]) vs. value).
    It may be obtained (defuzzified) as a crisp value, along with that value's suitability.
    """
    # First:  deal with the data every Value will need:

    global_crisper: Crisper = MedMax()  # a thing that does .defuzzify(). I guess there is a family of them.
    default_resolution: float = .001  # Needed when calling float(Value).
    default_interpolator = Interpolator()

    def __init__(self, domain: Tuple[float, float], default_suitability: float = 0):
        """Args:
            domain:  The domain of values over which the suitability is defined as a continuous function.
            default_suitability:  The suitability for values that are otherwise undefined."""
        self._d = domain
        self._ds = default_suitability

    @property
    def d(self) -> Tuple[float, float]:
        """The domain on which a function of suitability vs. value is defined."""
        return self._d

    @d.setter
    def d(self, d: Tuple[float, float] = (0, 0)) -> None:
        if d is not None:
            if d[0] > d[1]:
                raise ValueError("This domain is ill-defined: the upper bound must be above the lower.")
        self._d = d

    @property
    def ds(self) -> float:
        """The suitability reported for any value not in xp or d.
        If there *is* any reporting."""
        return self._ds

    @ds.setter
    def ds(self, ds: float = 0) -> None:
        if not Truth.is_valid(ds):
            raise ValueError("Suitabilities like this default suitability, ``ds``, must be on [0,1].")
        self._ds = ds

    # Second: deal with the required behaviors:  suitability, sample?, evaluate, crisp

    def suitability(self, v: float) -> float:
        """Returns the suitability at the given point.

        It refers, in order, to:  the exceptional points, the continuous function, and the default suitability."""

    @abstractmethod
    def evaluate(self, resolution: float) -> Numerical:
        """Obtains and returns a numerical representation of itself.
        This is where the work is done in each subclass, possibly by evaluating its :class:`Value` members
        and operating on them numerically."""

    def crisp(self, resolution: float, extreme_domain: (float, float) = None,
              crisper: Crisper = None) -> float:
        """Returns a crisp value that is equivalent to its fuzzy value.

        Returns:
            The crisp equivalent of this fuzzy number.

        Arguments:
            resolution: the maximum distance between values that will be considered in the numerical representation.
                This controls the accuracy of the result (a smaller resolution is better).
                Also, consider that a coarse mesh in the numerical representation might miss narrow peaks.
            extreme_domain: bounds the domain of the result in case the answer must be limited,
                e.g., if tempo must be on [30, 200] bpm, or a parameter must be on [0,100].
            crisper:  An object with a defuzzify method to do the actual work.
        """
        # Obtain a numerical representation of the fuzzy number at the appropriate resolution:
        numerical = self.evaluate(resolution)
        # Impose an extreme domain, if required:
        if extreme_domain is not None:
            numerical.impose_domain(extreme_domain)
        # Defuzzify the fuzzy number to obtain its crisp value:
        if crisper is None:
            crisper = Value.global_crisper
        v = crisper.defuzzify(numerical)
        return v

    def __float__(self) -> float:
        """returns the crisp float value using some default defuzzification parameters.
        but what about , extreme_domain: (float, float), resolution: float??"""
        value, _ = self.crisp(Value.default_resolution)  # what default resolution is appropriate???
        return value

    @staticmethod
    def _scale(p: np.ndarray, scale: Tuple[float, float]) -> np.ndarray:
        """A helper function to scale the suitabilities of a set of (v,s) pairs according to ``scale``
        taken as (min,max)."""
        v = p[:, 0]
        s = p[:, 1]
        raw_min, raw_max = np.min(s), np.max(s)
        if raw_min == raw_max:
            s = (scale[0] + scale[1]) / 2
        else:
            s = scale[0] + (scale[1] - scale[0]) * (s - raw_min) / (raw_max - raw_min)
        return np.dstack((v, s))


class Numerical(Value):
    """The numerical representation of a fuzzy value.

    To represent the suitabilities of all real numbers, it uses:

        * Discrete part: a 2D :class:`numpy.ndarray` of exceptional points :math:`(v_i, s_i)`.
        * Continuous part: Two :class:`numpy.ndarray`: ``v``, ``s``
          for (value, suitability) pairs over the domain ``d``.
        * A default suitability, ``ds``, that is assumed for any value otherwise undefined.

    """

    xp: np.ndarray  # exceptional points as (value, suitability) pairs
    v: np.ndarray  # values        } v & s:  a numerical representation of a function, s(v),
    s: np.ndarray  # suitabilities } over ``d`` + guard points.
    ds: float  # suitability elsewhere

    @classmethod
    def _setup(cls, domain, resolution):
        """A helper function to determine the sample points needed for a numerical representation over a domain."""
        if domain is None:
            return 0, 0, 0
        else:
            v_0 = floor(domain[0] / resolution)  # lowest sample number (or guard point)
            v_n = ceil(domain[1] / resolution)  # highest sample number (or guard point)
            # N.B.: use v_0 - 1 and v_n + 1 to ensure there is at least one guard point beyond the required domain.
            # As it is, I allow zero guard points if the domain bound coincides with a sample point.
            # That's fine for linear interpolation, but any more sophisticated interpolation would benefit from more.
            # TODO: make interpolation type an option with a global default using the above.
            n = v_n - v_0 + 1  # total number of samples
            return v_0, v_n, n

    def __init__(self, resolution: float = Value.default_resolution, domain: (float, float) = None,
                 points: np.ndarray = None, default_suitability: float = 0) -> None:
        """Initialization prepares the continuous domain ``d`` with sample points that are
        integer multiples of the resolution (so that all arrays in the calculation will line up),
        covering the stated domain plus guard points on either end (for future interpolation).

        The set of exceptional points (the discrete domain) is a 2D array of (value, suitability) pairs.
        Otherwise, undefined points in the domain default to a suitability of ``ds``.

        Args:
            resolution: the separation between sample points (a smaller resolution is better).
            domain: values over which the continuous domain will be defined.
            points: exceptional points as (v, s) pairs.
        """
        # the domain of the continuous function and the suitability elsewhere, outside the defined domains.
        super().__init__(domain, default_suitability)
        if resolution <= 0:
            raise ValueError("resolution must be >= 0.")
        self.resolution = resolution
        if domain is not None:  # Only bother with sampling the continuous domain if necessary.
            v_0, v_n, n = Numerical._setup(domain, resolution)  # v_0, v_n = range of sample nos.;  n = no. of samples
            v_0, v_n = v_0 * resolution, v_n * resolution  # value bounds of sampled domain (may exceed d for guard pts)
            # create sample points on the continuous domain, s to be calculated by subclasses:
            self.v = np.linspace(v_0, v_n, n)
            self.s = np.ones_like(self.v) * self.ds
        if points is None:  # the discrete domain:  exceptional points:
            self.xp = np.empty((2, 0))  # directions:  axis=0 columns, axis=1 rows; col. 0=values, col. 1 = suits.
        else:
            if (np.max(points, axis=0)[1] > 1) or (np.min(points, axis=0)[1] < 0):
                raise ValueError("Suitabilities like ``xp[v,1]`` must be on [0,1].")
            sorted = points[points[:, 0].argsort()]
            s = sorted[:, 0]
            if s[:-1][s[1:] == s[:-1]]:
                raise ValueError("You cannot define two suitabilities for the same value in ``xp``.")
            self.xp = sorted  # Pleasant side-effect: this leaves xp sorted by ascending values.

    def suitability(self, value: float, interp: Interpolator = None) -> float:
        """Returns the suitability of a given value.

        The exceptional points of the discrete domain override the definition of the continuous domain,
        which is generally found by interpolation.  Points outside these domains return a default value."""
        exceptional = np.where(value == self.xp)[0]  # returns the [row index] where condition is true...
        if exceptional.shape[0] != 0:  # ...as an array!  So, if it's not empty, that's the answer:
            return self.xp[exceptional[0]][1]
        else:
            if (self.d is None) or ((value < self.d[0]) or (value > self.d[1])):
                return self.ds
            else:
                if interp is None:
                    interp = Value.default_interpolator
                return interp.interpolate(value, self.v, self.s)

    def evaluate(self, resolution: float) -> Numerical:
        """It returns itself because it is the evaluation.

        In any other subclass of Value, this is where the work would be done."""
        return self

    def impose_domain(self, imposed_domain: (float, float)) -> Numerical:
        """Returns a copy of ``self``without exceptional points or continuous domain outside ``imposed_domain``."""
        if self.d is None:
            d = None
        else:
            min_d = max(self.d[0], imposed_domain[0])  # Find the intersection of self.d and imposed_domain.
            max_d = min(self.d[1], imposed_domain[1])
            d = (min_d, max_d) if (max_d >= min_d) else None  # Make sure that it is not the empty set.
        trimmed = Numerical(self.resolution, d)  # Begin building the trimmed version of self.
        if (self.xp is None) or (self.xp.size == 0):
            xp = None
        else:
            xp = self.xp
            if xp is not None:  # Remove the exceptional points outside imposed_domain.
                xp = xp[xp[:, 0] >= imposed_domain[0]]
            if xp is not None:
                xp = xp[xp[:, 0] <= imposed_domain[1]]
            if xp.size == 0:
                xp = None
        # Trim v and s as if they were being set up here:
        v_0_o, v_n_o, n_o = Numerical._setup(self.d, self.resolution)  # The original sample numbers
        v_0_i, v_n_i, n_i = Numerical._setup(d, self.resolution)  # The restricted sample numbers
        trim_left = v_0_i - v_0_o  # New v and s with the difference trimmed off:
        trim_right = trim_left + n_i
        cv = self.v[trim_left:trim_right]
        cs = self.s[trim_left:trim_right]
        trimmed.xp = xp
        trimmed.v = cv
        trimmed.s = cs
        trimmed.ds = self.ds
        return trimmed


# # Triangle, Trapezoid, Bell, Cauchy, Gauss, DPoints, CPoints --- "atoms" defining a value;
# # ValueNot, ValueAnd, ValueOr --- logic on values;
# # Sum, Difference, Prod, Quotient, Focus, Abs, Inverse --- arithmetic on values.


class DPoints(Numerical):
    """A fuzzy number defined as singular points---discrete (v,s) pairs."""

    def __init__(self, singular_points: Iterable[Tuple[float, float]],
                 scale: Tuple[float, float] = None, default_suitability: float = 0) -> None:
        """
        Args:
            points:  A collection of (*value*, *suitability*) pairs.  All *values* must be unique
                and all *suitabilities* must be on [0,1].
            scale:  If defined, the range of ``points`` is linearly scaled to ``scale``, taken as (*min*, *max*).
                If *min*>*max*, the sense of the suitabilities is flipped.
                If *min*==*max*, all suitabilities are the average of *min* and *max*.
            default_suitability:  The suitability outside the defined domain, defaults to 0.
            """
        p = np.array(singular_points)
        if scale is not None:
            p = Value._scale(p, scale)
        super().__init__(resolution=Value.default_resolution, domain=None,
                         points=p, default_suitability=default_suitability)


class Atom(Value):
    """The ancestor of classes which describe single fuzzy numbers as continuous functions which may be discretized.
    Much of the initialization is handled here.  This is an effectively abstract class.

    Subclasses need only define an s(v) function in their _sample method."""

    def __init__(self, domain: Tuple[float, float],
                 peak: float = 1., default_suitability: float = 0.,
                 discrete: bool = False, step: float = 1, origin: float = None):
        """
        Args:
            domain:  The domain, [min, max], over which the suitability varies.
            peak:  The suitability at one extreme, defaults to 1.
            dl:  The suitability where the :class:`Value` is undefined
                (perhaps at the opposite extreme), defaults to 0.
            discrete:  If ``True``, the function is not continuous, but a set of singular points.
            step:  If ``discrete``, the points are spaced at these intervals over ``domain``.
            origin: if defined, the singular points will be spaced about ``origin``
                rather than the value where ``peak`` is reached."""
        # the domain of the continuous function and the suitability elsewhere, outside the defined domains.
        super().__init__(domain, default_suitability)
        if not Truth.is_valid(peak):
            raise ValueError("Suitabilities like peak must be on [0,1].")
        self.peak = peak
        if step <= 0:
            raise ValueError("Step must be > 0.")
        self.step = step
        self.discrete = discrete
        if discrete:
            self.origin = origin
        else:
            self.origin = 0

    # @abstractmethod
    # def evaluate(self, resolution: float) -> Numerical:
    #     """Apparently, this must be here to keep the class abstract."""
    @abstractmethod
    def _sample(self, v: np.ndarray) -> np.ndarray:
        """Returns the value of the continuous function at all values in v."""

    def evaluate(self, resolution: float):
        if self.discrete:
            n = Numerical(resolution, domain=None, default_suitability=self.ds)
            n0 = ceil((self.d[0] - self.origin) / self.step)  # extreme sample numbers relative to origin
            n1 = floor((self.d[1] - self.origin) / self.step)
            v0 = self.origin + n0 * self.step  # extreme values
            v1 = self.origin + n1 * self.step
            v = np.linspace(v0, v1, n1 - n0 + 1)  # calculate the pairs
            s = self._sample(v)
            xp = np.dstack((v, s))  # fold them together
            n.xp = xp
        else:
            n = Numerical(resolution, domain=self.d, default_suitability=self.ds)
            n.s = self._sample(n.v)

        return n

    def suitability(self, v: float) -> float:
        """Returns the suitability of a given value as usual.

        This uses the _sample() method, which must be defined to accept numpy.ndarray.
        The alternative would be to implement this method for each subclass with a calculation shadowing its _sample().
        """
        if self.discrete:
            if ((v - self.origin) / self.step).is_integer():
                return self._sample(np.array([v]))[0]
            else:
                return self.ds
        else:
            if (self.d is None) or ((v < self.d[0]) or (v > self.d[1])):
                return self.ds
            else:
                return self._sample(np.array([v]))[0]


class Triangle(Atom):
    """Describes a fuzzy number as a triangular function with a peak (maximum s) and extreme limits (s==0)"""

    def __init__(self, a: float, b: float, c: float,
                 peak: float = 1., default_suitability: float = 0,
                 discrete: bool = False, step: float = 1, origin: float = None):
        """
        Args:
            a, b, c:  The minimum, preferred, and maximum values.
                The function is piecewise linear between these points.
            peak:  The suitability at ``b``, defaults to 1.
            dl:  The suitability at ``a`` and ``c``, defaults to 0.
            discrete:  If True, the function is not continuous, but a set of singular points.
            step:  If ``discrete``, the points are spaced at these intervals about ``b``, between ``a`` and ``c``.
            origin: if defined, the singular points will be spaced about ``origin`` rather than ``b``.
                N.B.: it needn't be in the domain [a,c].
            """
        if (b < a) or (b > c):
            raise ValueError("b must be in the domain [a,c].")
        self.b = b
        if origin is None:
            origin = b
        super().__init__((a, c), peak, default_suitability, discrete, step, origin)

    def _sample(self, v: np.ndarray) -> np.ndarray:
        """Returns the suitability for every value in ``v``."""
        if self.b == self.d[0]:
            left = np.ones_like(v)
        else:
            left = (v - self.d[0]) / (self.b - self.d[0])
        if self.d[1] == self.b:
            right = np.ones_like(v)
        else:
            right = (self.d[1] - v) / (self.d[1] - self.b)
        s = np.fmin(left, right)
        s = s * (self.peak - self.ds) + self.ds
        return s


class CPoints(Atom):
    """A fuzzy number defined as knots for interpolation---(v,s) pairs.

    The resulting function may be taken as continuous or defined only at uniformly spaced points."""

    def __init__(self, knots: Iterable[Tuple[float, float]], interp: Interpolator = None,
                 scale: Tuple[float, float] = None, default_suitability: float = 0,
                 discrete: bool = False, step: float = 1, origin: float = None) -> None:
        """
        Args:
            knots:  A collection of (*value*, *suitability*) pairs---knots to be interpolated between, producing s(v).
                All *values* must be unique and all *suitabilities* must be on [0,1].
            interp: The Interpolator used to construct the s(v) function.  The default is linear.
            scale:  If defined, the range of ``points`` is linearly scaled to ``scale``, taken as (*min*, *max*).
                If *min*>*max*, the sense of the suitabilities is flipped.
                If *min*==*max*, all suitabilities are the average of *min* and *max*.
            default_suitability:  The suitability outside the defined domain, defaults to 0.
            discrete:  If ``True``, the continuous s(v) function describes the *suitabilities*
                of a set of singular points, the *values* of which are determined by ``step`` and ``origin``.
            step:  If ``discrete``, the singular points are spaced at these intervals about ``origin``.
            origin: If ``discrete``, the value about which the singular points are spaced.
                If undefined, the default is the midpoint value of ``points``.
                N.B.: it can be outside ``points``.
            """
        p = np.array(knots)
        p = p[p[:, 0].argsort()]
        if scale is not None:
            p = Value._scale(p, scale)
        self.points_v = p[:, 0]
        self.points_s = p[:, 1]
        domain = (np.min(self.points_v), np.max(self.points_v))
        origin = (domain[0] + domain[1]) / 2 if origin is None else origin
        if interp is None:
            interp = Value.default_interpolator
        super().__init__(domain, 1, default_suitability, discrete, step, origin)
        self.interp = interp

    def _sample(self, v: np.ndarray) -> np.ndarray:
        """Returns the suitability for every value in ``v``."""
        return self.interp.interpolate(v, self.points_v, self.points_s)

# USA: balance, Labcorps, kroger: ?, commit-push, Anthelion, Jangwa
