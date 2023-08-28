"""The :class:`.Value` class applies the idea of fuzzy truth to the representation of numbers.

Introduction
------------

A :class:`Value` object is a function of truth vs. value.  We may think of the function as describing the suitability
of each possible value for some purpose.  E.g., we might describe room temperature as a symmetrical triangular function
on (68, 76)°F or (20, 24)°C.  In this way, we might state our opinions as "preference curves".  Notice how the fuzzy
number includes information about the ideal (the maximum), our tolerance for its variation (the width) and the limits
of what we will accept (the support, or non-zero domain).

We might also represent empirical knowledge as fuzzy numbers.  E.g., sensory dissonance vs. interval is a highly
structured function with many peaks and valleys across a wide domain, but all of this information can be encapsulated
in a single :class:`Value` object.

Fuzzy solutions to complex problems can look much like the crisp solutions.  Suppose a model is described by crisp
equations.  When the operators of those equations are made fuzzy, and the independent variables can be fuzzy, every
contingency can be planned for, and every subtle effect can be taken into account.

Fuzzy solutions can also use the dimension of truth and suitability---the certainty, strength, likelihood,
desirability of the inputs and our opinions about the goodness of the outputs are encoded in the numbers.
This gives meaning to logical operations between variables.  Suppose your opinion on room temperature
is different that mine---we can combine them by ANDing them together to find a compromise.  Enrich the model
with equations for energy prices, budget concerns, expected temperatures, time spent in the building, etc.,
and we can automate our argument about the thermostat.

Make all the natural-language statements you can about a system, its problems and heuristics.  They are already
very close to fuzzy statements, and are easily converted.  Your knowledge of a domain can easily become a very
sophisticated computer program automating your own judgment.

How to Use It
-------------

There are two domains in play.  There is the familiar world of absolute truths and precise numbers that we will
call "crisp".  There is also the world of partial truths and indefinite numbers that we will call "fuzzy".
In the fuzzy domain, the vagueness is measured with precision, so we can compute with it.  The intricacies happen
largely out of sight, so our calculations and reasoning retain their familiar form.  The main differences are:

    * We enter the fuzzy domain by defining fuzzy numbers, usually by intuitive parameters; or, by using informative
      functions about quantities varying between established limits *as* fuzzy numbers.
    * Within the fuzzy domain, we can use not only the familiar arithmetic operators in the usual way, but we can also
      combine numbers with the logical operators (and, or, not, and so on) to compute statements of reasoning.
    * The notion of weighting terms in a equation still exists, in two forms: trinary (by :meth:`.Value.weight`) and
      binary (by :meth:`.Value.focus`).
    * We reenter the crisp domain by the method :meth:`crisp`, to render a :class:`.Value` as a single float; or, by
      mapping the suitability of the fuzzy number to an output variable.  (All right.  I use "crisp" as a verb
      because the usual word, "defuzzify", is impossibly ugly.)

One may state one's ideas in the usual way, translate the statements into algorithms using logic and math operators
(overloaded to be fuzzy), plug in fuzzy numbers to the inputs, receive the fuzzy outputs, crisp them if desired, and
judge their quality by the indicated suitability of the results.

The :mod:`value` module is designed to work with the :mod:`truth` module and has some basic functionality provided
by the :mod:`norm` module.  Just as :class:`Truth` objects are analogous to ``bool``, :class:`Value` objects
are analogous to ``float``.

The abstract class :class:`.Value` provides the method:

    * :meth:`.Value.crisp`,

and guarantees the provision of:

    * :meth:`.Value.suitability`, and
    * :meth:`.Value.evaluate`.

The :meth:`.Value.crisp` method *defuzzifies* a fuzzy number, resulting in a crisp number, in our case
a single `float`.  It is usually used at the end of an algorithm, to produce a final result.  (Consider, though,
that the function of suitability vs. value that represents a fuzzy number might be more useful to you if you map
the suitability to some output variable.)  Since there are many opinions about how best to perform defuzzification,
this is handled by the class :class:`Crisper` and its descendants.  One may indicate a choice in the call or rely
on a global default.

The :meth:`.Value.suitability` method returns the suitability (the "truth" on [0,1]) of a given value (any real
number).  This is most useful for determining the goodness of a crisp result.  It may also be used by mapping the
suitability to an output variable, therefore mapping one value to another by a function determined by a fuzzy
reasoning process.  E.g., we may combine fuzzy information about sound spectra, dissonance, and plausible melodic
motions to calculate the frequency response of a filter.

The :meth:`.Value.evaluate` method returns a numerical representation of any :class:`Value` object, given a required
resolution.  This is used internally and probably won't be needed by end-users, unless they want a fuzzy result.
The purpose is to allow fuzzy numbers to be described by precise, analytical expressions.  When needed for calculation
(the algorithms for most arithmetic operators are numerical), they are called up in the required form.  Since all
of the operators are also of type :class:`Value` (in the sense that their result is a :class:`.Value`), and since
they hold their operands, which are also :class:`Value` objects, an expression builds up a tree of objects.
When the root is asked to evaluate itself, it asks its operands, and so on, resulting in a cascade of recursive calls
sending numerical representations of operands up the tree to the root.  Most often this is done by the call
to :meth:`.Value.crisp`.  In any case, users never see the underlying complexity.

These numerical representations have a standard form in the class :class:`Numerical`.  It represents a fit-valued
function over three domains, in order of decreasing priority:

* A set of exceptional points, discrete :math:`(v,s)` pairs.
* A continuous function, :math:`s(v)`, over a given domain, represented by an array of uniformly-spaced values and an
  array of corresponding suitabilities.  Calls requesting a suitability may, therefore, require interpolation, so there
  is a system of selectable interpolators like the ones for :class:`.Norm`, and :class:`.Crisper`,
* A default suitability (usually 0) to be reported for values not included in the above continuous or discrete domains.

The class also includes an :meth:`.impose_domain` method in case one needs to impose an extreme domain on a result,
e.g., if values outside a certain range are nonsensical.

TODO: I need to be able to apply the .to_float method from Truth to Numericals
---some kind of convenient mapping function.

What happens when :meth:`.Value.evaluate` calls reach the "leaves" of the expression "tree"?  What are they?
They are input variables holding well-defined fuzzy numbers.  There are several classes for creating these
"literals" by intuitive parameters.
described by

* Methods for defining fuzzy numbers parametrically:

    * :class:`.Triangle`
    * :class:`.Trapezoid`
    * :class:`.Cauchy`
    * :class:`.Gauss`
    * :class:`.Bell`
    * :class:`.DPoints`
    * :class:`.CPoints`
    * :class:`.Exactly` TODO: Exactly!


Most of these are subclasses of the abstract class , which provides the apparatus for declaring the

(:class:`.DPoints` is an exception---it is a subclass of :class:`.Numerical`)

Subclassing important functions: analytical Numpy function in the _sample method.


    All of these, except for  , subclass the abstract :class:

* Three basic logical connectives (the underscores are to difference them from Python keywords; the overloaded
  operators are rubricated as code; truth tables are shown in brackets):

    * :meth:`.Truth.not_` (¬, ``~``) [10],
    * :meth:`.Truth.and_` (∧, ``&``) [0001], and
    * :meth:`.Truth.or_` (∨, ``|``) [0111].

* The other eight non-trivial logical connectives familiar from propositional calculus and electronic logic
  gates:

    * :meth:`.imp` (→, "implies", ``>>``) [1101],
    * :meth:`.con` (←, *converse*, ``<<``) [1011],
    * :meth:`.iff` (↔, *equivalence*, "if and only if", "xnor", ``~(a @ b)``) [1001], and its inverse---
    * :meth:`.xor` (⨁, *exclusive or*, ``@``) [0110]; and the other inverses:
    * :meth:`.nand` (↑, non-conjunction, ``~(a & b)``) [1110],
    * :meth:`.nor` (↓, non-disjunction, ``~(a | b)``) [1000],
    * :meth:`.nimp` (:math:`\\nrightarrow`, non-implication, ``~(a >> b)``) [0010], and
    * :meth:`.ncon` (:math:`\\nleftarrow`, non-converse implication, ``~(a << b)``) [0100].

  All eleven logical connective methods can be defined by a :class:`Norm` optionally given in the call;
  otherwise, they use the :attr:`fuzzy.norm.global_norm` by default.

* The six comparisons: ``<``, ``>`` , ``<=``, ``>=``, ``==``, ``!=``.

* Methods, for emphasizing or deëmphasizing the contribution of a truth to whatever expression
  it is in---equivalent to partially defuzzifying or refuzzifying it.

    * :meth:`.weight`
    * :meth:`.focus`


* Two methods for making final decisions---"defuzzifying" the *fuzzy* :class:`Value` to a *crisp* ``float``:

    * :meth:`.Value.crisp`, which allows the crisper to be given in the call, or
    * :func:`.float`, which refers to a global default:  :attr:`.Value.global_crisper`.

* "Crispers"---classes that implement defuzzification routines:

* Interpolator, a class for implementing the interpolation routines needed for some descriptions.



How It Works
------------



and...
"literals" defining a value go here: Triangle, Trapezoid, Cauchy, Gauss, Bell, DPoints, CPoints:
Practical fuzzy numbers representing real things, e.g., Dissonance,
are probably descended from Literal (if generally continuous) or Numerical (if discrete and non-uniform).

"""

# Here are the classes for fuzzy value and arithmetic:
# Value --- base class;  Numerical --- "working" class of evaluation and defuzzification;
# "literals" defining a value go here: Triangle, Trapezoid, Cauchy, Gauss, Bell, DPoints, CPoints;
# logic on Values:  the same ops as for Truth
# arithmetic on Values: Sum, Difference, Prod, Quotient, Focus, Abs, Inverse, Negative
# Overloaded operators
# Interpolators and Crispers

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
        """Returns a copy of ``self``without exceptional points or continuous domain outside ``imposed_domain``.

        Caution:
            This doesn't change ds.  Should it?  How do you avoid trouble?"""
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


# "literals" defining a value go here: Triangle, Trapezoid, Cauchy, Gauss, Bell, DPoints, CPoints:
# Practical fuzzy numbers representing real things, e.g., Dissonance,
# are probably descended from Literal (if generally continuous) or Numerical (if discrete and non-uniform).


class Literal(Value):
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


class Triangle(Literal):
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


class CPoints(Literal):
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



# # Triangle, Trapezoid, Cauchy, Gauss, Bell, DPoints, CPoints --- "literals" defining a value;
# # ValueNot, ValueAnd, ValueOr --- logic on values;
# # Sum, Difference, Prod, Quotient, Focus, Abs, Inverse, Negative --- arithmetic on values.

# doc; Trapezoid, Cauchy, Gauss, Bell; interps; logic ops; arithmetic ops; overloads; crispers; review; test; parser


