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
    * The notion of weighting terms in an equation still exists, but in two forms: trinary (by :meth:`.Value.weight`)
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
to ``bool``, :class:`.Value` objects are analogous to ``float``.

*The Base Class: Value*
.......................

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


*The Working Class: Numerical*
..............................

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


*Literal Fuzzy Numbers*
.......................

What happens when :meth:`.Value.evaluate` calls reach the "leaves" of the expression "tree"?  What are they?
They are input variables holding well-defined fuzzy numbers.  There are several classes for creating them, as
"literals", by giving intuitive parameters.

* Methods for defining fuzzy numbers parametrically:

    * :class:`.Triangle`
    * :class:`.Trapezoid`
    * :class:`.Cauchy`
    * :class:`.Gauss`
    * :class:`.Bell`
    * :class:`.Sigmoid`
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
peak and width, a very natural way for talking vaguely about a number.  The class :class:`.Sigmoid` does the same for
talking vaguely about an inequality.

The classes :class:`.DPoints` and :class:`.CPoints` allow one ot describe a fuzzy number as a set of :math:`(v,s)`
pairs.  In the former, this is simply a discrete set.  In the latter, it implies a continuous function interpolated
between the points (the choice of interpolator is another parameter).  The class :class:`.Exactly` is for talking about
crisp number in the fuzzy world---it defines a fuzzy number with suitability 1 at the crisp value and 0 elsewhere.


*Making Your Own Literal Fuzzy Numbers*
.......................................

The above are fine for describing numbers in the ordinary sense---and literal numbers are often needed---but it is
most practical and interesting to use fuzzy numbers that represent real things---physical objects, mental experiences,
and relations among them.  This is done by using arbitrary, parameterized functions as fuzzy numbers.  The only
requirement is that their range be restricted to [0,1].  For example: Is the sun shining?  It depends on the time of
day, and, for a significant part of the day, a fuzzy answer is appropriate---and may bear on the thermostat argument.
Another:  sensory dissonance is a very predictable experience.  Usually, we are interested in its amount vs. the pitch
interval between two tones.  That curve, scaled to [0,1], becomes a very useful fuzzy number if you want to tune a
musical instrument.  Both examples may depend on many factors---custom-made fuzzy numbers may well depend on other
objects as parameters---but they may be boiled down to useful functions and used as fuzzy numbers in fuzzy reasoning.

No doubt you can think of many examples in whatever fields you have mastered.  To bring your knowledge into
the :mod:`fuzzy` package, you simply subclass of one of the existing classes.  There are four good candidates:

        * :class:`.Numerical`, for arbitrary discrete points.
        * :class:`.DPoints`, its subclass, which can map an arbitrary range to the required [0,1].
        * :class:`.Literal`, for functions defined by a method, on a continuous or discretized domain.
        * :class:`.CPoints`, its subclass, which defines the function as an interpolation between given knots and adds
          the above mapping functionality.

How do you choose?  If your fuzzy number could be easily described by a function, s(v), written in the form of a
Python method, use :class:`.Literal`.  If you know about certain important points, or curve features, or empirical
data, but not a detailed mathematical function, use :class:`.CPoints`.  If it's discrete, and you know the points that
matter or can determine them algorithmically, use :class:`.Numerical`; and, if that is the case but you also need
to map them from some other range, use :class:`.DPoints`.

First, you must implement your class's ``__init__`` method.  Add to it the parameters that shape your function.
If you're subclassing :class:`.Numerical`, :class:`.DPoints`, or :class:`.CPoints`, you'll set
your points in it as well.  Finally, remember to call the superclass's constructor in your
own: ``super().__init__(...)``.

If you're subclassing :class:`.Literal`, you'll need to do one more thing:  implement the :meth:`._sample` method.
This is where the action is.  It is a mathematical function, based on your initialized parameters, and
defined over a continuous domain---given a value, it returns the suitability of that value.  Its input and output,
though, are not single floats, but `Numpy <https://numpy.org/doc/stable/index.html>`_ arrays of floats, so the
mathematical expressions you use should be compatible with Numpy.  This will not change the usual syntax very much.
Of course, unlike an ordinary mathematical function, you have the full resources of Python and Numpy
to perform algorithms.

If you want to use the suitability mapping functionality mentioned above, it's available from :meth:`.Value._scale`.
Don't worry too much about singular points (like ``±inf`` or ``nan``), or otherwise defining suitabilities
outside [0,1]---this is guarded against internally.

There you go.  You've taken a concept from the real world---containing your human knowledge, experience, and insight,
I hope---and made it a computable object, available for reasoning in fuzzy algorithms.


*Qualifying Fuzzy Numbers*
..........................

With fuzzy numbers defined, you may wish to qualify them.......



* Methods, for emphasizing or deëmphasizing the contribution of a :class:`Value` to whatever expression
  it is in---equivalent to partially defuzzifying or refuzzifying it.

    * :meth:`.weight`
    * :meth:`.focus`


*Logical Operators*
...................

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


*Arithmetic Operators*
......................

* The N arithmetic operators:

    * :meth:`.add_` (:math:`+`, (associative) addition, ``+``),
    * :meth:`.sub_` (:math:`-`, (binary) subtraction, ``-``),
    * :meth:`.mul_` (:math:`\\times`, (associative) multiplication, ``-``),
    * :meth:`.div_` (:math:`\\div`, (binary) division, ``+``),
    * :meth:`.abs_` (:math:`|x|`, (unary) absolute value, ``+``),
    * :meth:`.neg_` (:math:`-`, (unary), negative (flipping the sign, not equivalent to logical negation) ``-x``),
    * :meth:`.inv_` (:math:`1/x`, (unary) inversion, ``-``),
    * :meth:`.pow_` (:math:`x^a`, (binary) exponentiation, ``**``), do I dare pow? exp? log?




*Obtaining Final Solutions*
...........................

Map!

* Two methods for making final decisions---"defuzzifying" the *fuzzy* :class:`Value` to a *crisp* ``float``:

    * :meth:`.Value.crisp`, which allows the crisper to be given in the call, or
    * :func:`.float`, which refers to a global default:  :attr:`.Value.default_crisper`.

* The six comparisons: ``<``, ``>`` , ``<=``, ``>=``, ``==``, ``!=``. ??? defined by crisping with ``float()``?

*Helpers*
.........

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
from math import floor, ceil, log, sqrt
from typing import Union, Tuple  # , ClassVar,

import numpy as np
from scipy.stats import norm as gauss

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
    default_interpolator: Interpolator = Interpolator(kind="linear")
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

        Args:
            v: any real number, a proposed value.

        Returns:
            The suitability of the proposed value, a measure of truth in the range [0,1].
        """

    @staticmethod
    def _xp_helper(v: float, xp: np.ndarray = None) -> float:
        """Implements the check for discrete points.

        This is called by :class:`.Literal` and :class:`.Numerical`.

        Args:
            v: A value to be checked.
            xp:  An array of (v, s) points.

            Returns:
                If ``v`` is in ``xp``, the corresponding ``s``, otherwise, ``None``.
            """
        s = None
        if xp is not None:
            exceptional = np.where(v == xp[:, 0])[0]  # returns the [row index] where condition is true...
            if exceptional.shape[0] != 0:  # ...as an array!  So, if it's not empty, that's the answer:
                s = xp[exceptional[0]][1]
        return s

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

        Caution:
            Probably, no one will ever call any class's :meth:`.evaluate` directly, nor reimplement it (since the
            implementations provided in :class:`.Numerical` and  :class:`.Literal` should do for all subclasses).
            Perhaps I should make it private (with an underscore), but users may subclass it, so I want the
            documentation to show up here.
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

        A :class:`.Value` is a function of suitability vs. value.  Sometimes that function is itself a useful result.
        It can be used in crisp mathematical expressions via the callable :class:`.Map` object returned by
        this method.

        The range of the internal function is restricted to [0,1].  To make it more convenient, the parameters
        allow you to translate this to ``range`` via a ``map`` (linear, logarithmic, or exponential).  This should
        make the result more easily adaptable.

    Args:
        resolution: The distance between sample values in the numerical representation.
            This controls how accurately the :class:`.Map` represents the original (a smaller resolution is better).
            Explicitly defined exceptional points are unaffected by resolution.
        range:  Translates the range of the internal function to the indicated range.  See :meth:`.Truth.scale`.
        map:  And does so via linear, logarithmic, or exponential mapping.  See :meth:`.Truth.scale`.
        interp:  An :class:`crisp.Interpolator` object for interpolating between the sample points.
            If none is indicated, :attr:`.default_interpolator` is used.

    Returns:
        A callable object that can be used as a mathematical function.

    Example:
        | ``loudness = amplitude_vs_pitch.map(range=(0,96), map = "log"")``
        | ``y = loudness(pitch)``

        """
        numerical = self.evaluate(resolution)
        return Map(numerical, range, map, interp)

    @staticmethod
    def _guard(s: Operand) -> Union[float, np.ndarray]:
        """A private helper function to deal with exceptional suitabilities {-inf, nan, +inf}
        as {0, :attr:`Truth.default_threshold`, 1}.

        This is used internally in :class:`.Numerical` and :class:`Literal`, from which all user-defined fuzzy numbers
        will probably descend, so it is unlikely that you will need to use it directly.

        Args:
            s: A presumed suitability (or array of them), which should be on [0,1], but might be exceptional.

        Returns:
            The best equivalent, restricted to [0,1]."""
        s = np.nan_to_num(s, nan=Truth.default_threshold, posinf=1, neginf=0)
        s = np.clip(s, 0, 1)
        if isinstance(s, np.ndarray):
            return s
        else:
            return float(s)

    @staticmethod
    def _scale(p: np.ndarray, expected_range: Tuple[float, float] = None, intended_range: Tuple[float, float] = (0, 1),
               map: str = "lin", clip: bool = False) -> np.ndarray:
        """A private helper function to scale a set of (v,y) pairs to a subrange of [0,1].

        This is called by :class:`.DPoints` for discrete points.  It uses :meth:`.Truth.scale`. but handles the
        separation and recombination of the 2D array.

        Args:
            p: A 2D array of (x, y) points.
            expected_range: The expected range of input data.  Default: the actual range given.
            intended_range: Any subrange of [0,1].  Default: (0,1)

        Other Parameters:
            [range], map, clip:  See :meth:`.Truth.scale`.

        Returns:
            The same exceptional points with the suitabilities rescaled."""
        v = p[:, 0]
        s = p[:, 1]
        s = Truth.scale(s, "in", expected_range, map, clip)
        s = Truth.scale(s, "out", intended_range, "lin", False)
        return np.dstack((v, s))[0]

    @staticmethod
    def _compare(a: Value, b: Value, type: str, resolution: float = default_resolution,
                 extreme_domain: (float, float) = None, crisper: Crisper = None) -> bool:
        """An inelegant private function to crisp both sides of a comparison compare them, and return the result.
        Is there a better way to do it?  This, at least, allows one to use :meth:`crisp`'s options.
        The overloaded operators just use the defaults.
        Is there a better definition for the comparators than to crisp then compare the floats?

        Args:
            a, b: Two fuzzy numbers to compare.
            type: The usual digraph indicating {>, >=, <, <=, ==, !=}.
            resolution, extreme_domain, crisper:  See :meth:`.Value.crisp`.
        Returns:
            The crisp truth of the comparison.
        """
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

    Caution:
        Probably, no one will ever instantiate a :class:`.Numerical` directly.  Perhaps I should make it private,
        but users may subclass it, so I want the documentation to show up here.

    Internally, within the :mod:`.value` module, most calculation in is carried out with :class:`.Numerical`\\ s.
    Externally, to users, :class:`.Numerical` is just a base class for fuzzy numbers defined by a set
    of discrete point that are not easily described by an analytical function.

    To represent suitability over all real numbers, it uses:

        :Discrete part:  A two-dimensional :class:`numpy.ndarray`, ``xp``, representing a set of exceptional
            points, :math:`\\{(v_i, s_i)\\}`.
        :Continuous domain:  A tuple ``d``, representing the domain, :math:`D = [d_0, d_1]`, over which
            the continuous part is defined.
        :Continuous part:  Two :class:`numpy.ndarray`\\ s,  ``v`` and ``s``, for (value, suitability) pairs,
            representing uniformly-spaced samples of a function, :math:`s(v)`.  (The sample points of ``v`` will extend
            beyond ``d`` with at least one guard point on either side.)
        :Elsewhere:  A default suitability, ``ds``, representing a result, :math:`d_s`, that is assumed for any value
            otherwise undefined (neither among the :math:`v_i` nor on :math:`D`).

    To subclass :class:`.Numerical`...
    """

    # xp: np.ndarray  # exceptional points as (value, suitability) pairs
    v: np.ndarray  # values        } v & s:  a numerical representation of a function, s(v),
    s: np.ndarray  # suitabilities } over ``d`` + guard points.
    ds: float  # suitability elsewhere

    @classmethod
    def _get_sample_limits(cls, domain, resolution):
        """A private helper function to determine the sample points needed for a numerical representation over a domain.

        This is used by :class`.Numerical`'s __init__ and impose_domain methods.

        Args:
            domain: The continuous domain (min, max) to find sample points over.
            resolution: How closely to space them.

        The sample points are integer multiples of the resolution so that, in any particular expression, they
        will all line up and no interpolation will be necessary.  So, every possible sample point has a unique
        integer associated with it:  sample point = integer index * resolution.

        Returns:
            v_0: The lowest sample point (a guard point, one or two points below min)
            v_n: The highest sample point (a guard point, one or two points above max)
            n = The total number of sample points needed."""
        if domain is None:
            return 0, 0, 0
        else:
            v_0 = floor(domain[0] / resolution) - 1  # lowest sample number (a guard point)
            v_n = ceil(domain[1] / resolution) + 1  # highest sample number (a guard point)
            # N.B.: The - 1 and + 1 above ensure that there is at least one guard point beyond the required domain.
            # They are needed for interpolations near the ends, called by :meth:`.suitability`.
            # They aren't actually needed for linear interpolations, but it can't hurt.
            n = v_n - v_0 + 1  # total number of samples
            return v_0, v_n, n

    def __init__(self, resolution: float = Value.default_resolution, domain: (float, float) = None,
                 points: np.ndarray = None, default_suitability: float = 0) -> None:
        """Initialization prepares the array of sample points, ``v``:

            * They are integer multiples of the resolution, so that all arrays in the calculation of a particular
              expression will line up---and so evaluations will have no interpolations to introduce error.
            * They cover the continuous domain, ``d``, with sample points, plus guard points on either end.
              The guard points provide for interpolation by the :meth:`.Value.suitability` method.

        Args:
            resolution: The separation between sample points.  A smaller resolution is better.  It must be
                positive and non-zero.
            domain: Values over which the continuous domain will be defined, as a tuple: (min, max).
            points: A two-dimensional array of exceptional points as (v, s) pairs.  The structure of the array is:

                | axis=0: columns:---  column 0: values;   column 1: suitabilities.
                | axis=1: rows

        """
        # The domain of the continuous function and the suitability elsewhere, outside the defined domains.
        super().__init__(domain, default_suitability)
        if resolution <= 0:  # Check for legal resolution.
            raise ValueError("resolution must be >= 0.")
        self.resolution = resolution
        if domain is None:
            self.v, self.s = np.empty(0), np.empty(0)
        else:  # Only bother with sampling the continuous domain if necessary.
            v_0, v_n, n = Numerical._get_sample_limits(domain, resolution)
            # v_0, v_n = range of sample numbers.;  n = total number of samples
            v_0, v_n = v_0 * resolution, v_n * resolution  # value bounds of sampled domain.
            # If I someday get a lot of problems from v[i]==0, I might offset the samples by resolution/2 ?
            # Create sample points on the continuous domain; s is to be calculated by subclasses:
            self.v = np.linspace(v_0, v_n, n)
            self.s = np.ones_like(self.v) * self.ds
        if points is None:  # The discrete domain:  exceptional points:
            self.xp = None  # directions:  axis=0 columns, axis=1 rows; col. 0=values, col. 1 = suits.
        else:
            points = np.array(points)
            if (np.max(points[:, 1], axis=0) > 1) or (np.min(points[:, 1], axis=0) < 0):
                raise ValueError("Suitabilities like ``xp[v,1]`` must be on [0,1].")
            sorted = points[points[:, 0].argsort()]
            s = sorted[:, 0]
            if s[:-1][s[1:] == s[:-1]]:
                raise ValueError("You cannot define two suitabilities for the same value in ``xp``.")
            self.xp = sorted  # Pleasant side effect: this leaves xp sorted by ascending values.

    def suitability(self, value: float, interp: Interpolator = None) -> float:
        """Returns the suitability of a given value, according to ``self``.

        In order of priority, it considers:

            * The exceptional points, ``xp``.
            * Any point defined on the continuous domain, ``d``.
            * The default suitability, ``ds``.

        Args:
            value:  Any real number.
            interp:  The :class:`.crisp.Interpolator` used for points on ``d`` between the sample points of ``v``.
                If none is given, the :attr:`.default_interpolator` is used.

        Returns:
            The suitability of ``value`` as defined by this :class:`.Numerical`."""
        s = Value._xp_helper(value, self.xp)
        if s is None:
            if (self.d is None) or ((value < self.d[0]) or (value > self.d[1])):
                s = self.ds
            else:
                if interp is None:
                    interp = Value.default_interpolator
                s = interp.interpolate(value, self.v, self.s)
        return Value._guard(s)

    def evaluate(self, resolution: float) -> Numerical:
        """It returns itself because it is the evaluation.

        In any other subclass of Value, this is where the work would be done.

        Args:
            resolution: The spacing between the values of the sample points.

        Return:
            ``self``.

        Caution:
            Probably, no one will ever call :meth:`.Numerical.evaluate` directly, nor reimplement it.
            Perhaps I should make it private (with an underscore), but I think it's better to make the operation
            of the module explicit.
        """
        return self

    def impose_domain(self, imposed_domain: (float, float)) -> Numerical:
        """Returns a copy of ``self`` restricted to the given domain.

        This is useful if values outside the given domain cannot be valid solutions to a problem.

        The resulting continuous domain is redefined to be the intersection of the old and the given.
        Exceptional points outside the given domain and unneeded sample points are simply discarded.

        Args:
            imposed_domain:  The extremes of the permitted result's domain, as a tuple (min, max).

        Returns:
            A copy of ``self`` undefined outside the given domain.

        Note:
            * The new domain is the intersection of the old and the given, i.e.,
              the result is ``self.d`` ∩ ``imposed_domain``.
            * Where the imposed domain exceeds the old domain, nothing is added.  Nothing new is defined.
            * The default suitability is unaffected.  It will still be returned for any undefined value, whether
              previously defined or not."""
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
        v_0_o, v_n_o, n_o = Numerical._get_sample_limits(self.d, self.resolution)  # The original sample numbers
        v_0_i, v_n_i, n_i = Numerical._get_sample_limits(d, self.resolution)  # The restricted sample numbers
        trim_left = v_0_i - v_0_o  # New v and s with the difference trimmed off:
        trim_right = trim_left + n_i
        cv = self.v[trim_left:trim_right]
        cs = self.s[trim_left:trim_right]
        trimmed.xp = xp
        trimmed.v = cv
        trimmed.s = cs
        trimmed.ds = self.ds
        return trimmed


# "literals" defining a value go here: Triangle, Trapezoid, Cauchy, Gauss, Bell, DPoints, CPoints, Exactly:
# Practical fuzzy numbers representing real things, e.g., Dissonance,
# are probably descended from Literal (if generally continuous) or Numerical (if discrete and non-uniform).


class Literal(Value):
    """An abstract base for fuzzy numbers defined by a mathematical function given as a Python method.

    The fuzzy number may be continuous (the default) or discretized.  This is useful if the suitabilities may be
    easily defined by a mathematical function, but only valid for discrete values.
    There are three ways of doing this:

        * For an arbitrary collection of values, given in ``discrete``.
        * For a set of uniformly-spaced values, by setting ``uniform=True`` and, optionally, ``step`` and ``origin``.
        * Both of the above together.

    Subclasses must implement :meth:`.Literal._sample` to be the s(v) function that defines the fuzzy number.
    It's input and output are :class:`numpy.ndarray`\\ s, so it should be done using Numpy mathematics.

    Subclasses may well choose not to expose either or both discretization behaviors in their interface.
    They may also set their ``self.origin`` to a default, if it is not given on initialization.
    This should probably be the value where the function reaches its "peak" (``range[1]``), or some other
    critical point.  For example:

        | ``if origin is None:``
        |   ``origin = value_at_peak``

    Note:
        A Literal (or Numerical) that represents a continuous function, can have exceptional points added by
        simple assignment, e.g.:

            | ``a_function = Triangle(1, 2, 3)``
            | ``an_exception = Exactly(8)``
            | ``a_function.xp = an_exception.xp``
            | ``print(a_function.suitability(7))``
            | ``print(a_function.suitability(8))``

        prints 0 then 1.  Note that the exceptional point needn't be in the domain of the continuous part.
    """

    def __init__(self, domain: Tuple[float, float],
                 range: Tuple[float, float] = (0, 1), default_suitability: float = 0.,
                 discrete: Iterable[float] = None,
                 uniform: bool = False, step: float = 1, origin: float = None):
        """
        Args:
            domain:  The domain, [min, max], over which the suitability varies.  The full, continuous domain is
                used, unless it is discretized by defining ``discrete`` or by setting ``uniform=True``.
            range:  The extremes of suitability that will be reached.  Default: [0,1].
            default_suitability:  Returned for values that are otherwise undefined.  Default: 0.
            discrete: If defined, the domain is restricted to these values, given explicitly
                (and to any defined by ``uniform``).
            uniform:  If ``True``, the domain is restricted to a set of uniformly-spaced values, according
                to ``step`` and ``origin`` (and to any defined by ``discrete``).
            step:  Uniform points are spaced at these intervals over ``domain``.
            origin: Uniform points will be measured from ``origin``.  It needn't be in the domain.
                Default: the middle of the domain.  Subclasses would do well to default to the value where the
                function "peaks" (i.e., reaches ``range[1]``)."""
        super().__init__(domain, default_suitability)
        if not (Truth.is_valid(range[0]) and Truth.is_valid(range[1])):
            raise ValueError("Suitabilities like those in ``range`` must be on [0,1].")
        self.range = range
        if discrete is not None:
            discrete = np.array(discrete)
            if (discrete < domain[0]).any() or (discrete > domain[1]).any():
                raise ValueError("discrete points outside the domain are redundant---they'd report the default anyway.")
        self.discrete = discrete
        self.uniform = uniform
        self.step = step
        if uniform:
            if step <= 0:
                raise ValueError("Step must be > 0.")
            if origin is None:
                self.origin = (domain[1] - domain[0]) / 2
            else:
                self.origin = origin
        else:
            self.origin = 0
        self.xp = None

    @abstractmethod
    def _sample(self, v: np.ndarray) -> np.ndarray:
        """This is where you implement the s(v) function that defines your fuzzy number.
        It should return a suitability on [0,1] for any real number in the domain, ``self.d``."""

    def evaluate(self, resolution: float) -> Numerical:
        """Returns a numerical representation of itself.

        This is where the discretization behavior of :class:`.Literal` is implemented.  N.B.: all results are
        run through :meth:`.Value._guard` to ensure that no undefined numbers or non-fits creep in.

        See :meth:`.Value.evaluate`.

        Caution:
            Probably, no one will ever call :meth:`.Literal.evaluate` directly, nor reimplement it.
            Perhaps I should make it private, but I think it's better to make the operation of the module explicit.
        """
        if self.uniform or (self.discrete is not None):
            v = np.empty(0)
            if self.uniform:  # find the sample points
                n0 = ceil((self.d[0] - self.origin) / self.step)  # extreme sample numbers relative to origin
                n1 = floor((self.d[1] - self.origin) / self.step)
                v0 = self.origin + n0 * self.step  # extreme values
                v1 = self.origin + n1 * self.step
                v = np.linspace(v0, v1, n1 - n0 + 1)  # calculate the pairs
            if self.discrete is not None:
                v = np.unique(np.concatenate((v, self.discrete)))
            # build the result
            s = Value._guard(self._sample(v))
            xp = np.dstack((v, s))[0]  # fold them together
            n = Numerical(resolution, domain=None, points=xp, default_suitability=self.ds)
        else:
            n = Numerical(resolution, domain=self.d, default_suitability=self.ds)
            s = self._sample(n.v)  # Sample the Literal.
            if s[-2] != s[-2]:  # This strange case is because Akima interpolation produces two NaNs at the end.
                s[-2] = 2 * s[-3] - s[-4]
            s = Value._guard(s)  # Handle singular points and clip anything not on [0,1]
            if len(s) > 2:  # Linear extrapolation to the guard points---excursions beyond [0,1] allowed here, I think.
                s[0] = 2 * s[1] - s[2]
                s[-1] = 2 * s[-2] - s[-3]
            n.s = s
        return n

    def suitability(self, v: float) -> float:
        """Returns the suitability of a given value.

        See :meth:`.Value.suitability`.
        """
        s = Value._xp_helper(v, self.xp)
        if s is None:
            s = self.ds
            if self.uniform or (self.discrete is not None):
                if self.uniform and ((v - self.origin) / self.step).is_integer():
                    s = self._sample(np.array([v]))[0]
                if (self.discrete is not None) and (v in self.discrete):
                    s = self._sample(np.array([v]))[0]
            elif (self.d is not None) and ((v >= self.d[0]) and (v <= self.d[1])):
                s = self._sample(np.array([v]))[0]
        return Value._guard(s)


class Triangle(Literal):
    """Describes a fuzzy number as a triangular function."""

    def __init__(self, a: float, b: float, c: float,
                 range: Tuple[float, float] = (0, 1), default_suitability: float = 0,
                 discrete: Iterable[float] = None,
                 uniform: bool = False, step: float = 1, origin: float = None):
        """
        Args:
            a, b, c:  The minimum, preferred, and maximum values.  The function is piecewise linear between
                these points.  The condition :math:`a \\le b \\le c` must hold
            range:  Default: (0,1).  The suitabilities are:

                | ``range[0]`` at ``a`` and ``c``,
                | ``range[1]`` at ``b``.

        Other Parameters:
            discrete, uniform, step, origin: relate to discretizing the domain.  See :class:`.Literal`.
            origin:  Default: ``b``.
            """
        if not (a <= b <= c):
            raise ValueError("a <= b <= c must hold.")
        self.b = b
        if origin is None:
            origin = b
        super().__init__((a, c), range, default_suitability, discrete, uniform, step, origin)

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
        s = s * (self.range[1] - self.range[0]) + self.range[0]
        return s


class Trapezoid(Literal):
    """Describes a fuzzy number as a trapezoidal function."""

    def __init__(self, a: float, b: float, c: float, d: float,
                 range: Tuple[float, float] = (0, 1), default_suitability: float = 0,
                 discrete: Iterable[float] = None,
                 uniform: bool = False, step: float = 1, origin: float = None):
        """
        Args:
            a, b, c, d:  The extreme domain, [a,d], and, within it, the preferred domain [b,c].
                The function is piecewise linear between
                these points.  The condition :math:`a \\le b \\le c \\le d` must hold.
            range:  Default: (0,1).  The suitabilities are:

                | ``range[0]`` at ``a`` and ``d``,
                | ``range[1]`` at ``b`` and ``c``.

        Other Parameters:
            discrete, uniform, step, origin: relate to discretizing the domain.  See :class:`.Literal`.
            origin:  Default: the center of the preferred region.
            """
        if not (a <= b <= c <= d):
            raise ValueError("a <= b <= c <= d must hold.")
        self.b = b
        self.c = c
        if origin is None:
            origin = (c - b) / 2
        super().__init__((a, d), range, default_suitability, discrete, uniform, step, origin)

    def _sample(self, v: np.ndarray) -> np.ndarray:
        """Returns the suitability for every value in ``v``."""
        if self.b == self.d[0]:
            left = np.ones_like(v)
        else:
            left = (v - self.d[0]) / (self.b - self.d[0])
        if self.d[1] == self.c:
            right = np.ones_like(v)
        else:
            right = (self.d[1] - v) / (self.d[1] - self.c)
        middle = np.ones_like(v)
        triangle = np.fmin(left, right)
        s = np.fmin(triangle, middle)
        s = s * (self.range[1] - self.range[0]) + self.range[0]
        return s


class Cauchy(Literal):
    """Describes a fuzzy number as the bell-shaped function due to Augustin-Louis Cauchy.

    This is a way of talking about a number as "``c`` ± ``hwhm``"."""

    def __init__(self, c: float, hwhm: float, domain: Tuple[float, float] = None,
                 range: Tuple[float, float] = (0, 1), default_suitability: float = 0,
                 discrete: Iterable[float] = None,
                 uniform: bool = False, step: float = 1, origin: float = None):
        """
        Args:
            c:  The most preferred value, at which the bell peaks.  It need not be in the ``domain``.
            hwhm: The half width at half maximum.
            domain: The extreme domain where the function is defined. Default:  the domain that covers
                the suitability down to 1/1000 of its peak.

        Other Parameters:
            discrete, uniform, step, origin: relate to discretizing the domain.  See :class:`.Literal`.
            origin:  Default: ``c``.
            """
        self.c = c
        if not (hwhm > 0):
            raise ValueError("hwhm must be greater than 0.")
        self.hwhm = hwhm
        if origin is None:
            origin = c
        if domain is None:
            domain = (c - 31.6069612585582 * hwhm, c + 31.6069612585582 * hwhm)
        super().__init__(domain, range, default_suitability, discrete, uniform, step, origin)

    def _sample(self, v: np.ndarray) -> np.ndarray:
        """Returns the suitability for every value in ``v``."""
        s = (self.hwhm ** 2) / (np.square(v - self.c) + self.hwhm ** 2)
        s = s * (self.range[1] - self.range[0]) + self.range[0]
        return s


class Gauss(Literal):
    """Describes a fuzzy number as the bell-shaped function due to Carl Friedrich Gauss.

    This is a way of talking about a number as a normal distribution about an expectation value."""

    def __init__(self, c: float, sd: float, domain: Union[Tuple[float, float], float] = None,
                 range: Tuple[float, float] = (0, 1), default_suitability: float = 0,
                 discrete: Iterable[float] = None,
                 uniform: bool = False, step: float = 1, origin: float = None):
        """
        Args:
            c:  The most preferred value, at which the bell peaks.  It need not be in the domain.
            sd: The size of one standard deviation---a larger value gives a wider bell.
            domain: The extreme domain where the function is defined. It can be given as a tuple, (min, max),
                or as ``float`` indicating the number of standard deviations about ``c``.
                Default:  the domain that covers the suitability down to 1/1000 of its peak.

        Other Parameters:
            discrete, uniform, step, origin: relate to discretizing the domain.  See :class:`.Literal`.
            origin:  Default: ``c``.
            """
        self.c = c
        if not (sd > 0):
            raise ValueError("sd must be greater than 0.")
        self.sd = sd
        if origin is None:
            origin = c
        if domain is None:  # if undefined:  .001 of peak
            domain = (c - 3.71692219 * sd, c + 3.71692219 * sd)
        if not isinstance(domain, tuple):
            d = domain * sd
            domain = (c - d, c + d)
        super().__init__(domain, range, default_suitability, discrete, uniform, step, origin)

    def _sample(self, v: np.ndarray) -> np.ndarray:
        """Returns the suitability for every value in ``v``."""
        s = self.sd * 2.50662827 * gauss.pdf(v, loc=self.c, scale=self.sd)
        s = s * (self.range[1] - self.range[0]) + self.range[0]
        return s


class Bell(Literal):
    """Describes a fuzzy number as a generalized bell membership function.

    This is a way of talking about a number as ``c`` ± ``hwhm`` with confidence ``slope``, or with
    vagueness ``transition_width``."""

    def __init__(self, c: float, hwhm: float, slope: float = None, domain: Union[Tuple[float, float], float] = None,
                 range: Tuple[float, float] = (0, 1), default_suitability: float = 0,
                 discrete: Iterable[float] = None,
                 uniform: bool = False, step: float = 1, origin: float = None,
                 transition_width: float = None):
        """
        Args:
            c: The center of the bell, the most preferred value.
            hwhm: The half width at half maximum of the bell.
            slope: The steepness of the sides---the absolute slope at half maximum.
                Default: see ``transition_width`` below.
            transition_width: Is an alternative to specifying the slope.  It is the width, given in multiples
                of ``hwhm``, of the region on either side of the bell where the suitability varies on [.1, .9].
                If defined, it overrides the definition of ``slope``.  Default: 1.
            domain: The extreme domain where the function is defined.  Default:  the domain that covers
                the suitability down to 1/1000 of its peak.

        Warning:
            It's fairly easy to set a moderately shallow ``slope`` and get a huge default ``domain``.
            It's wise to set ``domain`` manually if your ``slope < 1``.

        Other Parameters:
            discrete, uniform, step, origin: relate to discretizing the domain.  See :class:`.Literal`.
            origin:  Default: ``c``.
            """
        self.c = c
        if hwhm == 0:
            raise ValueError("``a`` cannot equal zero.")
        self.hwhm = hwhm
        if (slope is None) and (transition_width is None):
            transition_width = 1
        if transition_width is None:
            if not (slope > 0):
                raise ValueError("``slope`` must be greater than 0.")
            b = slope * 2 * hwhm
        else:
            if not (transition_width > 0):
                raise ValueError("``transition_width`` must be greater than 0.")
            b = 1.09861229 / log(.5 * (transition_width + sqrt(4 + (transition_width ** 2))))
        self.b = b
        if origin is None:
            origin = c
        if domain is None:  # if undefined:  .001 of peak
            w = hwhm * (3 ** (3 / (2 * b))) * (37 ** (1 / (2 * b)))
            domain = (c - w, c + w)
        super().__init__(domain, range, default_suitability, discrete, uniform, step, origin)

    def _sample(self, v: np.ndarray) -> np.ndarray:
        """Returns the suitability for every value in ``v``."""
        s = 1 / (1 + np.abs((v - self.c) / self.hwhm) ** (2 * self.b))
        s = s * (self.range[1] - self.range[0]) + self.range[0]
        return s


class Sigmoid(Literal):
    """Describes a fuzzy inequality as a sigmoid curve.

    The "``sense``" of an inequality can be either ``">"`` or ``"<"``.
    A Sigmoid, then, is a way of talking about an inequality as: "a value is ``sense c`` to within ``width``", or
    "with confidence ``slope``"."""

    def __init__(self, sense: str, c: float, width: float = None,
                 domain: Tuple[float, float] = None, range: Tuple[float, float] = (0, 1),
                 discrete: Iterable[float] = None,
                 uniform: bool = False, step: float = 1, origin: float = None,
                 slope: float = None):
        """
        Args:
            sense: Either ``">"`` (more suitable on the right---when greater than ``c``) or ``"<"``
                (more suitable on the left---when less than ``c``).
            c:  The center of the transition, at which the suitability is .5.  It need not be in the ``domain``.
            width: The full width of the transition region, where the suitability varies on [.1, .9].
                Default: see ``slope`` below.
            slope: Is an alternative to specifying the width.  It is simply the slope at ``c``.  If defined,
                it overrides the definition of ``width``.  Default: 1.
            domain: The extreme domain where the function is defined. Default:  the domain that covers
                the suitability on [.001, .999].

        Other Parameters:
            discrete, uniform, step, origin: relate to discretizing the domain.  See :class:`.Literal`.
            origin:  Default: ``c``.

        Warning:
            It's fairly easy to set a moderately shallow ``slope`` and get a huge default ``domain``.
            It's wise to set ``domain`` manually if your ``slope < 1``.

        Caution:
            * The :meth:`.Sigmoid.suitability` method has been overridden to give the extremes of ``range`` according
              to the ``sense`` of the inequality.  Consequently, there is no ``v`` for which it automatically returns
              ``default_suitability``.  Therefore, that parameter is not in the signature.
            * Making ``range[1] > range[0]`` will reverse the ``sense`` of the inequality.

            """

        self.c = c
        if (slope is None) and (width is None):
            slope = 1
        if slope is None:
            if not (width > 0):
                raise ValueError("``width`` must be greater than 0.")
            a = 4.39444915467244 / width
        else:
            if not (slope > 0):
                raise ValueError("``slope`` must be greater than 0.")
            a = 4 * slope
        self.sense = -1 if sense == "<" else 1
        self.a = a
        if origin is None:
            origin = c
        if domain is None:  # if undefined:  s = [.001, .999]
            w = 6.906754778648554 / self.a
            domain = (c - w, c + w)
        super().__init__(domain, range, 0, discrete, uniform, step, origin)

    def _sample(self, v: np.ndarray) -> np.ndarray:
        """Returns the suitability for every value in ``v``."""
        s = 1 / (1 + np.exp(-self.sense * self.a * (v - self.c)))
        s = s * (self.range[1] - self.range[0]) + self.range[0]
        return s

    def suitability(self, v: float) -> float:
        """Returns the suitability of any real number, ``v``.

        To follow the behavior expected of an inequality, it returns an extreme of ``range`` according to the ``sense``
        of the inequality (``">"`` or ``"<"``).    I.e., it's a shelf function.
        N.B.: it does not automatically return ``default_suitability`` for any ``v``. """
        if v < self.d[0]:
            return self.range[0] if self.sense > 0 else self.range[1]
        elif v > self.d[1]:
            return self.range[1] if self.sense > 0 else self.range[0]
        else:
            return super().suitability(v)


class CPoints(Literal):
    """A fuzzy number defined as knots for interpolation.

    Note the similarity of its interface to that of :class:`DPoints`---it is easy to convert the calls.
    """

    def __init__(self, knots: Iterable[Tuple[float, float]], interp: str = None,
                 expected_range: Tuple[float, float] = (0, 1), intended_range: Tuple[float, float] = (0, 1),
                 map: str = "lin", default_suitability: float = 0,
                 discrete: Iterable[float] = None,
                 uniform: bool = False, step: float = 1, origin: float = None) -> None:
        """
        Args:
            knots:  A collection of (value, suitability) pairs---knots to be interpolated between, producing s(v).
                All values must be unique and all suitabilities will be scaled to ``intended_range``, if defined.
            interp: Interpolator type used to construct the s(v) function.  See :class:`.Interpolator`.  Some
                interpolators may define curves that stray outside [0,1], but these will be clipped automatically.
                Default: ``linear``.
            expected_range: The expected range of input data.  Default: ``(0,1)``.

                * The default allows direct entry of data.
                * If ``expected_range==None``, the input will be normalized to fill ``intended_range``.

            intended_range: Any subrange of [0,1] (it cannot be exceeded).  Default: ``(0,1)``

        Other Parameters:
            map:  relates to mapping between ranges.  See :meth:`.Truth.scale`
            discrete, uniform, step, origin: relate to discretizing the domain.  See :class:`.Literal`.

            """
        p = np.array(knots)
        p = p[p[:, 0].argsort()]
        self.points_v = p[:, 0]
        self.points_s = p[:, 1]
        self.points_s = Truth.scale(self.points_s, "in", expected_range, map, False)
        self.points_s = Truth.scale(self.points_s, "out", intended_range, "lin", False)
        domain = (np.min(self.points_v), np.max(self.points_v))
        super().__init__(domain, intended_range, default_suitability, discrete, uniform, step, origin)
        if interp is None:
            self.interp = Value.default_interpolator
        else:
            self.interp = Interpolator(kind=interp)

    def _sample(self, v: np.ndarray) -> np.ndarray:
        """Returns the suitability for every value in ``v``."""
        return self.interp.interpolate(v, self.points_v, self.points_s)


class DPoints(Numerical):
    """A fuzzy number defined as discrete points.

    Note the similarity of its interface to that of :class:`CPoints`---it is easy to convert the calls."""

    def __init__(self, points: Iterable[Tuple[float, float]],
                 expected_range: Tuple[float, float] = (0, 1), intended_range: Tuple[float, float] = (0, 1),
                 map: str = "lin", clip: bool = False,
                 default_suitability: float = 0) -> None:
        """
        Args:
            points:  A collection of (value, suitability) pairs---discrete points.
                All values must be unique and all suitabilities will be scaled to ``intended_range``, if defined.
            expected_range: The expected range of input data.  Default: ``(0,1)``.

                * The default allows direct entry of data.
                * If ``expected_range==None``, the input will be normalized to fill ``intended_range``.

            intended_range: Any subrange of [0,1] (it cannot be exceeded).  Default: ``(0,1)``
            default_suitability:  The suitability elsewhere than the defined points, defaults to 0.

        Other Parameters:
            expected_range, intended_range, map, clip:  See :meth:`.Truth.scale`.
            """
        if isinstance(points[0], Tuple):  # If it's more than one point:
            p = np.array(points)
            if intended_range is not None:
                print(f"p: {p}")
                v = p[:, 0]  # This slicing doesn't work for a single point.  Is there a better way?
                s = p[:, 1]
                s = Truth.scale(s, "in", expected_range, map, clip)
                s = Truth.scale(s, "out", intended_range, "lin", False)
                p = np.dstack((v, s))[0]
        else:  # If it's only one point:
            v = points[0]
            s = points[1]
            p = np.dstack((v, s))[0]
        super().__init__(resolution=Value.default_resolution, domain=None,
                         points=p, default_suitability=default_suitability)


class Exactly(DPoints):
    """A fuzzy number exactly equivalent to a crisp number.

    This enables crisp numbers to be used in fuzzy calculations.
    """

    def __init__(self, value: float) -> None:
        """
        Args:
            value:  The value where suitability is 1---it is 0 at all other points.
            """
        super().__init__(points=(value, 1), default_suitability=0)


# Logic operators on fuzzy numbers:





# # Sum, Difference, Prod, Quotient, Focus, Abs, Inverse, Negative --- arithmetic on values.

# doc; Trapezoid, Cauchy, Gauss, Bell; interps; logic ops; arithmetic ops; overloads; crispers; review; test; parser
