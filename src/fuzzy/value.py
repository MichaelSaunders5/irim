"""applies the idea of fuzzy truth to the representation of numbers.

Introduction
------------

A :class:`.Value` object is a function of truth vs. value.  We may think of the function as describing the *suitability*
of each possible value for some purpose.  E.g., we might describe room temperature as a symmetrical triangular function
on (68, 76)°F or (20, 24)°C.  In this way, we might state our opinions as "preference curves".  Notice how the fuzzy
number includes information about the ideal (the maximum), our tolerance for its variation (the width), and the limits
of what we will accept (the *support*, or non-zero domain).  An arbitrary function can contain an enormous amount
of detail.

Fuzzy numbers might also represent empirical knowledge.  E.g., sensory dissonance vs. interval is a highly
structured function with many peaks and valleys across a wide domain, but all of this information can be encapsulated
in a single :class:`.Value` object.

Fuzzy solutions to complex problems can look much like the crisp solutions.  Suppose a model is described by crisp
equations.  When the operators of those equations are made fuzzy, and the independent variables can be fuzzy, every
contingency can be planned for, and every subtle effect can be taken into account.

Fuzzy solutions can also use the dimension of truth and suitability---the certainty, strength, likelihood, or
desirability of the inputs and our opinions about the goodness of the outputs are encoded in the numbers.
This gives meaning to logical operations between numbers.  Suppose your opinion on room temperature
is different than mine---we can combine them by ANDing them together to find a compromise.  Enrich the model
with equations for energy prices, budget concerns, expected temperatures, time spent in the building, etc.,
and we can automate our argument about the thermostat.

Make all the natural-language statements you can about a system, its problems and heuristics.  They are already
very close to fuzzy statements, and are easily converted.  Your knowledge of a domain can easily become a very
sophisticated computer program automating your own judgment.

How to Use It
-------------

There are two worlds in play.  There is the familiar world of absolute truths and precise numbers that we will
call "crisp".  There is also the world of partial truths and indefinite numbers that we will call "fuzzy".
In the fuzzy world, the vagueness is measured with precision, so we can compute with it.  The intricacies happen
largely out of sight, so our calculations and reasoning retain their familiar form.  The main differences are:

    * We enter the fuzzy world by defining fuzzy numbers, either:

        * Literally---via intuitive parameters (as in the room temperature example); or,
        * Analytically---by adopting *as* fuzzy numbers informative functions that vary between established limits
          (as in the dissonance example).

    * Within the fuzzy world, we can use not only the familiar arithmetic operators in the usual way, but we can also
      combine numbers with the logical operators (and, or, not, and so on) to compute statements of reasoning.
    * The notion of weighting terms in a equation still exists, but in two forms: trinary (by :meth:`.Value.weight`)
      and binary (by :meth:`.Value.focus`).
    * We reënter the crisp world by the methods:

        * :meth:`.Value.crisp`, to render a :class:`.Value` as a single, most suitable, float; or
        * :meth:`.map`, to make the suitability vs. value function of the fuzzy number a callable object,
          usable in crisp expressions.

      (All right.  I use "crisp" as a verb because the usual word, "defuzzify", is impossibly ugly.)

One may state one's ideas in the usual way, translate the statements into algorithms using logic and math operators
(overloaded to be fuzzy), plug in fuzzy numbers to the inputs, receive the fuzzy outputs, crisp them if desired, and
judge their quality by the indicated suitability of the results.

The :mod:`value` module is designed to work with the :mod:`truth` module and has some basic functionality provided
by the :mod:`norm` and :mod:`crisp` modules.  Just as :class:`Truth` objects are analogous
to ``bool``, :class:`.Value` objectsare analogous to ``float``.

The abstract class :class:`.Value` provides the methods:

    * :meth:`.Value.crisp`, and
    * :meth:`.Value.map`,

and guarantees the provision of:

    * :meth:`.Value.suitability`, and
    * :meth:`.Value.evaluate`.

The :meth:`.Value.crisp` method *defuzzifies* a fuzzy number, resulting in a crisp number, in our case
a single ``float``.  That is, among all the real numbers, it chooses the best answer, the crisp number that best
represents the fuzzy number. It is usually used at the end of an algorithm, to produce a final result.
There are many opinions about how best to perform defuzzification.  They are represented by descendants of the
class :class:`.Crisper`.  As with norms, thresholds, and interpolators, one may rely on a global default, or
choose one in the call.

The :meth:`.Value.map` method creates a callable object usable as a mathematical function---the same suitability vs.
value function that defines the :class:`.Value` object.  Consider that the best solution to a problem may not be
a crisp number.  Often it is a mathematical function that preserves the details of the reasoning and can itself
implement a nuanced behavior.  E.g., you may combine fuzzy information about sound spectra, dissonance, and plausible
melodic motions to calculate the frequency response of a filter.  The :class:`.Map` object produced by the method
is usable in any math expression as easily as ``y = name(x)``.  When it is created, you may choose a mapping
(linear, logarithmic, or exponential) from the suitability's [0,1] range to whatever range you require.

The :meth:`.Value.suitability` method returns the suitability (the "truth" on [0,1]) of a given value (any real
number).  This is most useful for determining the goodness of a crisp result.

To put it dramatically:---

    | ``give_me_a_yes_or_no_answer: bool = my_truth.crisp()``
    | ``give_me_a_hard_cold_figure: float = my_value.crisp()``
    | ``is_that_so: float = my_value.suitability(a_hard_cold_figure)``
    | ``give_me_the_perfume_of_the_thing: Map = my_value.map()``

The :meth:`.Value.evaluate` method returns a numerical representation of any :class:`.Value` object, given a required
resolution.  This is used internally and probably won't be needed by end-users.  The purpose is to allow fuzzy numbers
to be defined by precise, analytical expressions.  Yet, since the algorithms for operators are numerical, fuzzy numbers
must be cast in numerical form for calculation.  Since all the operators are also of type :class:`Value` (because
their result is a :class:`.Value`), and since they hold their operands, which are also :class:`.Value` objects,
an expression builds up a tree of objects.  When the root of the tree is asked to evaluate itself, it asks its
operands, and so on, resulting in a cascade of recursive calls sending numerical representations of operands up
the tree to the root.  Most often this is done by the calls to :meth:`.Value.crisp`, or :meth:`.Value.map`.
In any case, users never see the underlying complexity.

The class :class:`.Numerical` provides a standard form for these numerical representations.  It represents
a fuzzy number as a fit-valued function over three domains, in order of decreasing priority:

* A set of exceptional points, discrete :math:`(v,s)` pairs.
* A continuous function, :math:`s(v)`, over a domain :math:`D`, represented by an array of uniformly-spaced values
  and an array of corresponding suitabilities.  Calls requesting a suitability may, therefore, require interpolation,
  so there is a system of selectable :class:`.Interpolator`\\ s like the ones for :class:`.Norm`, and :class:`.Crisper`,
* A default suitability, :math:`s_d`, (usually 0) to be reported for values not included in the above continuous
  or discrete domains.

The class also includes an :meth:`.impose_domain` method in case one needs to impose an extreme domain on a result,
e.g., if values outside a certain range are nonsensical.

What happens when :meth:`.Value.evaluate` calls reach the "leaves" of the expression "tree"?  What are they?
They are input variables holding well-defined fuzzy numbers.  There are several classes for creating them, as
"literals", by giving intuitive parameters.

* Methods for defining fuzzy numbers parametrically:

    * :class:`.Triangle`
    * :class:`.Trapezoid`
    * :class:`.Cauchy`
    * :class:`.Gauss`
    * :class:`.Bell`
    * :class:`.DPoints`
    * :class:`.CPoints`
    * :class:`.Exactly`


Most of these are subclasses of the abstract class :class:`.Literal`, which provides the apparatus for declaring the
fuzzy number conveniently and precisely:  as a continuous, analytical function (by implementing a private
method, :meth:`._sample`).  They include the option to restrict the domain further to a set of uniformly-spaced,
discrete values.  This is useful if the solutions one seeks must be integer multiples of some unit, e.g., if you need
to reason about feeding your elephants, you may leave the oats continuous, but please, make sure your elephants
are discrete.  The classes :class:`.DPoints` and :class:`.Exactly` are exceptions---though used as "literals" they
describe only discrete sets and so descend from :class:`.Numerical`, which makes their implementation much simpler.

The classes :class:`.Triangle` and :class:`.Trapezoid` are the simple linear functions one usually encounters in
simpler fuzzy logic implementations.  (However, I like how they have constant variability over their linear segments,
and so have relatively large variabilities near their extrema---this keeps things exciting.)

The classes :class:`.Cauchy`, :class:`.Gauss`, and :class:`.Bell` all represent bell-shaped functions defined by their
peak and width, a very natural way for talking vaguely about a number.

The classes :class:`.DPoints` and :class:`.CPoints` allow one ot describe a fuzzy number as a set of :math:`(v,s)`
pairs.  In the former, this is simply a discrete set.  In the latter, it implies a continuous function interpolated
between the points (the choice of interpolator is another parameter).  The class :class:`.Exactly` is for talking about
crisp number in the fuzzy world---it defines a fuzzy number with suitability 1 at the crisp value and 0 elsewhere.

The above are fine for describing numbers in the ordinary sense---and literal numbers are often needed---but it is
most practical and interesting to use fuzzy numbers that represent real things---physical objects, mental experiences,
and relations among them.  This is done by using arbitrary functions as fuzzy numbers.  The only requirement is that
their range be restricted to [0,1].  For example: Is the sun shining?  It depends on the time of day, and, for
a significant part of the day, a fuzzy answer is appropriate---and may bear on the thermostat argument.  Another:
sensory dissonance is a very predictable experience.  Usually, we are interested in its amount vs. the pitch interval
between two tones.  That curve, scaled to [0,1], becomes a very useful fuzzy number if you want to tune a musical
instrument.  Both examples may depend on many factors, but may be boiled down to useful functions and used in
reasoning as fuzzy numbers.

No doubt you can think of many examples in whatever fields you are expert in.  To bring your knowledge into
the :mod:`fuzzy` package, you simply create a subclass of :class:`.Literal` (if the function is generally continuous)
or :class:`.Numerical` (if it is discrete and non-uniform).  Alternatively, you might subclass *their*
subclasses, :class:`.CPoints` or :class:`.DPoints`, to take advantage of their interpolation and scaling behaviors.
Then you add appropriate parameters to your class's ``__init__`` method (remembering to call ``super().__init__()``);
and, in the case of :class:`.Literal` subclasses, implement the :meth:`._sample` method with an analytical definition
of your function.  (This method must be compatible with `Numpy <https://numpy.org/doc/stable/index.html>`_, that is,
able to take arrays as well as ``float``\\ s, but this will normally have little effect on your syntax.)  In this way,
a concept from the real world becomes a computable object in your algorithms.

With fuzzy numbers defined, you may wish to qualify them.......



* Methods, for emphasizing or deëmphasizing the contribution of a :class:`Value` to whatever expression
  it is in---equivalent to partially defuzzifying or refuzzifying it.

    * :meth:`.weight`
    * :meth:`.focus`


With fuzzy numbers defined, and possibly qualified, the next step is to reason with them.  This is done by putting
them into expressions where numbers are operated upon---transmuted or combined into different numbers.


* The eleven logical connectives, familiar from :mod:`truth` but here applied to the suitabilities of :class:`Value`
  objects (the underscores are to difference them from Python keywords; the overloaded operators are rubricated
  as code; truth tables are shown in brackets):

    * :meth:`.Value.not_` (¬, ``~``) [10], the only unary; the two associatives:
    * :meth:`.Value.and_` (∧, ``&``) [0001], and
    * :meth:`.Value.or_` (∨, ``|``) [0111]; and the eight that are purely binary:
    * :meth:`.Value.imp` (→, "implies", ``>>``) [1101],
    * :meth:`.Value.con` (←, *converse*, ``<<``) [1011],
    * :meth:`.Value.iff` (↔, *equivalence*, "if and only if", "xnor", ``~(a @ b)``) [1001], and its inverse---
    * :meth:`.Value.xor` (⨁, *exclusive or*, ``@``) [0110]; and the other inverses:
    * :meth:`.Value.nand` (↑, non-conjunction, ``~(a & b)``) [1110],
    * :meth:`.Value.nor` (↓, non-disjunction, ``~(a | b)``) [1000],
    * :meth:`.Value.nimp` (:math:`\\nrightarrow`, non-implication, ``~(a >> b)``) [0010], and
    * :meth:`.Value.ncon` (:math:`\\nleftarrow`, non-converse implication, ``~(a << b)``) [0100].

  All eleven logical connective methods can be defined by a :class:`Norm` optionally given in the call;
  otherwise, they use the :attr:`fuzzy.norm.default_norm` by default.

* The N arithmetic operators:

    * :meth:`.add_` (:math:`+`, (associative) addition, ``+``),
    * :meth:`.sub_` (:math:`-`, (binary) subtracttion, ``-``),
    * :meth:`.mul_` (:math:`\\times`, (associative) multiplication, ``-``),
    * :meth:`.div_` (:math:`\\div`, (binary) division, ``+``),
    * :meth:`.abs_` (:math:`|x|`, (unary) absolute value, ``+``),
    * :meth:`.neg_` (:math:`-`, (unary), negative (flipping the sign, not equivalent to logical negarion) ``-x``),
    * :meth:`.inv_` (:math:`1/x`, (unary) inversion, ``-``),
    * :meth:`.pow_` (:math:`x^a`, (binary) exponentiation, ``**``), do I dare pow? exp? log?


* The six comparisons: ``<``, ``>`` , ``<=``, ``>=``, ``==``, ``!=``. ??? defined by crisping with ``float()``?


* Two methods for making final decisions---"defuzzifying" the *fuzzy* :class:`Value` to a *crisp* ``float``:

    * :meth:`.Value.crisp`, which allows the crisper to be given in the call, or
    * :func:`.float`, which refers to a global default:  :attr:`.Value.default_crisper`.

* "Crispers"---classes that implement defuzzification routines:

* Interpolator, a class for implementing the interpolation routines needed for some descriptions.


"""

# Here are the classes for fuzzy value and arithmetic:
# Value --- base class;  Numerical --- "working" class of evaluation and defuzzification;
# "literals" defining a value go here: Triangle, Trapezoid, Cauchy, Gauss, Bell, DPoints, CPoints, Exactly;
# logic on Values:  the same ops as for Truth
# arithmetic on Values: Sum, Difference, Prod, Quotient, Focus, Abs, Inverse, Negative;
# Overloaded operators
# Map, Interpolators, and Crispers go in crisp.py

from __future__ import annotations

from abc import ABC, abstractmethod
from math import floor, ceil, inf, nan
from typing import Tuple  # , ClassVar,

import numpy as np

from fuzzy.crisp import Crisper, MedMax, Interpolator, Map
from fuzzy.truth import Truth


class Value(ABC):
    """A fuzzy real number.

    The representation is a function, :math:`s(v)`, of "suitability" or "truth" (on [0,1]) vs. value
    (on the real numbers).

    It implements:

        * :meth:`map`, which turns the function into a callable object;
        * :meth:`crisp`, which finds the best real number equivalent for it (by defuzzification); and
        * ``float()``, which does the same using only default parameters.

    Its subclasses implement:

        * :meth:`suitability`, which returns the suitability of a given value; and
        * :meth:`evaluate`, which returns a numerical representation of :math:`s(v)`, a :class:`Numerical`.

    """

    default_resolution: float = .001  # Needed when calling float(Value).
    """The minimum difference in value that is considered by:
    
    * The :meth:`crisp` method (as the default) and by ``float()``;
    * All comparisons, e.g., :meth:`__gt__` (as the default) and by their operators, e.g., ``>``.
    * The constructor of :class:`Numerical` (as the default), including when :meth:`.Value.evaluate` is called.
    
    """
    default_interpolator: Interpolator = Interpolator()
    """The interpolator that is used when:
    
    * Constructing a :class:`CPoints` fuzzy number (to interpolate the sample points between the knots).
    * Calculating the suitability in the continuous domain of a :class:`Numerical` 
      (to interpolate between the sample points).
    """
    default_crisper: Crisper = MedMax()
    """The :class:`.Crisper` (defuzzifier) that is used by the methods :meth:`.Value.crisp` (by default) 
    and ``float()``."""

    def __init__(self, domain: Tuple[float, float], default_suitability: float = 0):
        """Args:
            domain:  The domain of values over which the suitability is defined as a continuous function
                 (i.e., not including any exceptional points).
            default_suitability:  The suitability for values that are otherwise undefined (the default is 0).

        Note:
            * Subclasses might also define exceptional, discrete points.
            * The subclass :class:`Numerical` provides that functionality.
            * Subclasses needn't define a continuous function."""
        self._d = domain
        self._ds = default_suitability

    @property
    def d(self) -> Tuple[float, float]:
        """The domain on which a function of suitability vs. value is defined by a continuous function."""
        return self._d

    @d.setter
    def d(self, d: Tuple[float, float] = (0, 0)) -> None:
        """Ensures that the continuous domain has a width of at least 0."""
        if d is not None:
            if d[0] > d[1]:
                raise ValueError("This domain is ill-defined: the upper bound must be above the lower.")
        self._d = d

    @property
    def ds(self) -> float:
        """The default suitability, reported for any value not on ``d`` or defined as an exceptional point."""
        return self._ds

    @ds.setter
    def ds(self, ds: float = 0) -> None:
        """Ensures that the default suitability is on [0,1]."""
        if not Truth.is_valid(ds):
            raise ValueError("Suitabilities like this default suitability, ``ds``, must be on [0,1].")
        self._ds = ds

    @abstractmethod
    def suitability(self, v: float) -> float:
        """Returns the suitability at the given value.

        It should refer to, in order of priority:  the exceptional points, the continuous function,
        and the default suitability.

        Arg:
            v: any real number, a proposed value.

        Return:
            The suitability of the proposed value, a measure of truth in the range [0,1].
        """

    @abstractmethod
    def evaluate(self, resolution: float) -> Numerical:
        """Obtains and returns a numerical representation of itself.

        This is where the work is done in each subclass.  In a :class:`Literal` number, the work is to sample its
        analytical function appropriately.  In an :class:`Operator`, the work is to call for evaluations
        of its operands, operate on them, and return the numerical result.

        Arguments:
            resolution: The spacing in value between sample points of the continuous function.  This determines how
                accurately the result represents the original.

        Return:
            A numerical representation of ``self``, i.e., of the :math:`s(v)` that defines the fuzzy number.
        """

    def crisp(self, resolution: float = default_resolution, extreme_domain: Tuple[float, float] = None,
              crisper: Crisper = None) -> float:
        """Returns a crisp value that is equivalent to ``self``\\ s fuzzy value.

        Arguments:
            resolution: The distance between sample values in the numerical representation.
                This controls the accuracy of the result (a smaller resolution is better).
                Also, consider that a coarse mesh in the numerical representation might miss narrow peaks.
                (Exceptional points defined explicitly are unaffected by resolution.)
            extreme_domain: bounds the domain of the result in case the answer must be limited,
                e.g., if tempo must be on [30, 200] bpm, or temperature must be on [-273.15, 100]°C.
            crisper:  The :class:`Crisper`  object that performs the defuzzification.
                If none is indicated, :attr:`default_crisper` is used.

        Return:
            The crisp equivalent of this fuzzy number, according to ``crisper``.
        """
        # Obtain a numerical representation of the fuzzy number at the appropriate resolution:
        numerical = self.evaluate(resolution)
        # Impose an extreme domain, if required:
        if extreme_domain is not None:
            numerical.impose_domain(extreme_domain)
        # Defuzzify the fuzzy number to obtain its crisp value:
        if crisper is None:
            crisper = Value.default_crisper
        v = crisper.defuzzify(numerical)
        return v

    def __float__(self) -> float:
        """Returns the crisp float value, via :meth:`.crisp`, using only default defuzzification parameters."""
        value = self.crisp(Value.default_resolution)
        return value

    def map(self, range: Tuple[float, float], map: str = "lin",
            resolution: float = default_resolution, interp: Interpolator = None) -> Map:
        """Creates a callable object that maps the :math:`s(v)` of ``self`` to the real numbers.

        A :class:`.Value` is a function of suitability vs. value.  Sometimes that function is a useful result.
        It can be used in crisp mathematical expressions via the callable :class:`.Map` object returned by
        this method.

        The range of the internal function is restricted to [0,1].  To make it more convenient, the parameters
        allow you to translate this to ``range`` via a ``map`` (linear, logarithmic, or exponential).  This should
        make the result more easily adaptable.

    Args:
        resolution: The distance between sample values in the numerical representation.
            This controls how accurately the :class:`.Map` represents the original (a smaller resolution is better).
            (Exceptional points that the :class:`.Value` defines explicitly are unaffected by resolution.)
        range:  Translates the range of the internal function to the indicated range.  See :meth:`scale`.
        map:  And does so via linear, logarithmic, or exponential mapping.  See :meth:`scale`.
        interp:  An :class:`Interpolator` object for interpolating between the sample points.
            If none is indicated, :attr:`.default_interpolator` is used.

    Returns:
        A callable object that can be used as a mathematical function.

    Example:"
        | loudness = amplitude.map(range=(0,96), map = "log", resolution = .001, interp = "linear")
        | y = loudness(pitch)

        """
        numerical = self.evaluate(resolution)
        return Map(numerical, range, map, interp)

    @staticmethod
    def guard(v: float) -> float:
        """A helper function to be called by implementations of :meth:`suitability` to prevent crazy results."""
        if v == -inf:
            return 0
        elif v == inf:
            return 1
        elif v == nan:
            return Truth.default_threshold
        else:
            return min(max(v, 0), 1)

    @staticmethod
    def _scale(p: np.ndarray, dir: str, range: Tuple[float, float],
               map: str = "lin", clip: bool = False) -> np.ndarray:
        """A helper function to scale the suitabilities from a set of (v,y) pairs according to ``range``
        taken as (min,max) with optional log mapping."""
        v = p[:, 0]
        s = p[:, 1]
        t = Truth.scale(s, dir, range, map, clip)
        return np.dstack((v, t))

    @staticmethod
    def _compare(a: Value, b: Value, type: str, resolution: float = default_resolution,
                 extreme_domain: (float, float) = None, crisper: Crisper = None) -> bool:
        """An inelegant private function to crisp both sides of a comparison and return the result.
        Is there a better way to do it?  This, at least, allows one to use :meth:`crisp`'s options.
        Is there a better definition for the comparators than to crisp then compare the floats?"""
        if type == "gt":
            return a.crisp(resolution, extreme_domain, crisper) > b.crisp(resolution, extreme_domain, crisper)
        elif type == "ge":
            return a.crisp(resolution, extreme_domain, crisper) >= b.crisp(resolution, extreme_domain, crisper)
        elif type == "lt":
            return a.crisp(resolution, extreme_domain, crisper) < b.crisp(resolution, extreme_domain, crisper)
        elif type == "le":
            return a.crisp(resolution, extreme_domain, crisper) <= b.crisp(resolution, extreme_domain, crisper)
        elif type == "eq":
            return a.crisp(resolution, extreme_domain, crisper) == b.crisp(resolution, extreme_domain, crisper)
        else:  # type == "ne":
            return a.crisp(resolution, extreme_domain, crisper) != b.crisp(resolution, extreme_domain, crisper)

    # Each comparator needs to be overloaded:
    def __gt__(self, other):
        return Value._compare(self, other, type="gt")

    def __ge__(self, other):
        return Value._compare(self, other, type="ge")

    def __lt__(self, other):
        return Value._compare(self, other, type="lt")

    def __le__(self, other):
        return Value._compare(self, other, type="le")

    def __eq__(self, other):
        return Value._compare(self, other, type="eq")

    def __ne__(self, other):
        return Value._compare(self, other, type="ne")


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
            v = self.xp[exceptional[0]][1]
        else:
            if (self.d is None) or ((value < self.d[0]) or (value > self.d[1])):
                v = self.ds
            else:
                if interp is None:
                    interp = Value.default_interpolator
                v = interp.interpolate(value, self.v, self.s)
        return Value.guard(v)

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
                v = self.ds
            else:
                v = self._sample(np.array([v]))[0]
        return Value.guard(v)


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
                 range: Tuple[float, float] = None, map: str = "lin", clip: bool = False,
                 default_suitability: float = 0,
                 discrete: bool = False, step: float = 1, origin: float = None) -> None:
        """
        Args:
            knots:  A collection of (*value*, *suitability*) pairs---knots to be interpolated between, producing s(v).
                All *values* must be unique and all *suitabilities* must be on [range[0], range[1]].
            interp: The Interpolator used to construct the s(v) function.  The default is linear.
            default_suitability:  The suitability outside the defined domain, defaults to 0.
            discrete:  If ``True``, the continuous s(v) function describes the *suitabilities*
                of a set of singular points, the *values* of which are determined by ``step`` and ``origin``.
            step:  If ``discrete``, the singular points are spaced at these intervals about ``origin``.
            origin: If ``discrete``, the value about which the singular points are spaced.
                If undefined, the default is the midpoint value of ``points``.
                N.B.: it can be outside ``points``.

        Other Parameters:
            range, map, clip:  relate to mapping fuzzy units.  See :meth:`.Truth.scale`.
            """
        p = np.array(knots)
        p = p[p[:, 0].argsort()]
        if range is not None:
            p = Value._scale(p, "in", range, map, clip)
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
                 range: Tuple[float, float] = None, map: str = "lin", clip: bool = False,
                 default_suitability: float = 0) -> None:
        """
        Args:
            points:  A collection of (*value*, *suitability*) pairs.  All *values* must be unique
                and all *suitabilities* must be on [range[0], range[1]].
            default_suitability:  The suitability outside the defined domain, defaults to 0.

        Other Parameters:
            range, map, clip:  relate to mapping fuzzy units.  See :meth:`.Truth.scale`.
            """
        p = np.array(singular_points)
        if range is not None:
            p = Value._scale(p, "in", range, map, clip)
        super().__init__(resolution=Value.default_resolution, domain=None,
                         points=p, default_suitability=default_suitability)

# # Triangle, Trapezoid, Cauchy, Gauss, Bell, DPoints, CPoints --- "literals" defining a value;
# # ValueNot, ValueAnd, ValueOr --- logic on values;
# # Sum, Difference, Prod, Quotient, Focus, Abs, Inverse, Negative --- arithmetic on values.

# doc; Trapezoid, Cauchy, Gauss, Bell; interps; logic ops; arithmetic ops; overloads; crispers; review; test; parser
