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

The class also includes an :meth:`._impose_domain` method in case one needs to impose an extreme domain on a result,
e.g., if values outside a certain range are nonsensical.


*Literal Fuzzy Numbers*
.......................

What happens when :meth:`.Value.evaluate` calls reach the "leaves" of the expression "tree"?  What are they?
They are input variables holding well-defined fuzzy numbers.











There are several classes for creating them, as
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

The module :mod:`.operator`, contains the code for fuzzy logic and math operators on :class:`.Value`\\ s.
The code for crispers, interpolators and suitability mapping are all in :mod:`.crisp`.


"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import ceil, floor, pi, isfinite  # , log, sqrt
from typing import Union, Tuple  # , ClassVar,

import matplotlib.pyplot as plt
import numpy as np
from screeninfo import get_monitors

from fuzzy.crisp import Crisper, MedMax, Interpolator, Map, NumericalMap
from fuzzy.truth import Truth, TruthOperand, default_threshold

monitor_w, monitor_h = 1920, 1080  # When the module loads, find the monitor dimensions, for use by :meth:`.display`.
for m in get_monitors():
    if m.is_primary:
        if (monitor_w is not None) and (monitor_w > 0):
            monitor_w, monitor_h = m.width, m.height

default_resolution: float = .01  # Needed when calling float(FuzzyNumber).
"""The minimum difference in value that is considered by:

* The :meth:`.FuzzyNumer.crisp` method (as the default) and by ``float()``;
* All comparisons, e.g., :meth:`.__gt__` (as the default) and by their operators, e.g., ``>``.
* The constructor of :class:`._Numerical` (as the default), including 
  when :meth:`.FuzzyNumber._get_numerical` is called.
  
Who can say what is best for all units and situations?  No one!  But, in the above situations, 
there is no place for a parameter."""
default_interpolator: Interpolator = Interpolator(kind="linear")
"""The interpolator that is used when:

* Constructing a :class:`literal.CPoints` fuzzy number (to interpolate the sample points between the knots).
* Calculating the truth in the continuous domain of a :class:`._Numerical` 
  (to interpolate between the sample points)."""
default_crisper: Crisper = MedMax()
"""The :class:`.Crisper` (defuzzifier) that is used by the methods :meth:`.FuzzyNumber.crisp` (by default) 
and ``float()``."""
default_sampling_method: str = "Chebyshev"
"""How :class:`.Literal`\\ s are sampled.  
The options are Chebyshev (producing near-minimax approximations); or "uniform"."""


def _guard(t: TruthOperand) -> Union[float, np.ndarray]:
    """A private helper function to deal with exceptional truths {-inf, nan, +inf}
    as {0, :attr:`Truth.default_threshold`, 1}.

    This is used internally in :class:`._Numerical` and :class:`Literal`, from which all user-defined fuzzy numbers
    will probably descend, so it is unlikely that you will need to use it directly.

    Args:
        s: A presumed truth (or array of them), which should be on [0,1], but might be exceptional.

    Returns:
        The best equivalent, restricted to [0,1], returned as an array or float depending on the argument type."""
    r = np.nan_to_num(t, nan=default_threshold, posinf=1, neginf=0)
    r = np.clip(r, 0, 1)
    if isinstance(t, np.ndarray):
        return r
    else:
        return float(r)


def _t_for_v_in_xv(v: float, xv: np.ndarray, xt: np.ndarray) -> Union[float, None]:
    """Implements the check for discrete points.

    This is called by :meth:`._Numerical.t`.

    Args:
        v: A value to be checked.
        xp:  An array of (v, s) points.

        Returns:
            If ``v`` is in ``xp``, the corresponding ``t``, otherwise, ``None``.
        """
    if (xv is None) or (xt is None):
        return None
    i = np.where(xv == v)[0]
    if len(i) == 0:
        return None
    return xt[i][0]



class Domain(Tuple):
    def __new__(cls, d: Tuple[float, float]) -> Domain:
        if d is not None:
            if d[1] < d[0]:
                raise ValueError(f"Domains must have d[1] >= d[0]. ({d[0]}, {d[1]}) is invalid.")
        return super(Domain, cls).__new__(cls, d)

    def __str__(self):
        return str(f"({self[0]:,.4g}, {self[1]:,.4g})")

    def span(self) -> float:
        """Returns the extent of the domain."""
        return self[1] - self[0]

    def center(self) -> float:
        """Returns the central coördinate of the domain."""
        return self[0] + self.span() / 2

    def intersection(self, d: Domain) -> Union[Domain, None]:
        """Returns the domain that is ``self`` ∩ ``d``."""
        if d is None:
            return None
        new_min = max(self[0], d[0])
        new_max = min(self[1], d[1])
        return None if new_min > new_max else Domain((new_min, new_max))

    def union(self, d: Domain) -> Domain:
        """Returns the domain that is ``self`` ∪ ``d``."""
        if d is None:
            return self
        new_min = min(self[0], d[0])
        new_max = max(self[1], d[1])
        return Domain((new_min, new_max))

    def contains(self, v: float) -> bool:
        """True iff ``v`` is on ``self``."""
        if self is None:
            return False
        else:
            return not ((v < self[0]) or (v > self[1]))

    @staticmethod
    def sort(*d) -> Union[Domain, None]:
        if d is None:
            return None
        if d[1] < d[0]:
            return Domain((d[1], d[0]))
        else:
            return Domain((d[0], d[1]))


class FuzzyNumber(ABC):
    """A fuzzy real number.

    The representation is a function, :math:`t(v)`, of "suitability" or "truth" (on [0,1]) vs. value
    (on the real numbers).  Generally, subclasses may explicitly define:

        * Exceptional points, (v, t) pairs;
        * A continuous function, t(v), over a single contiguous domain, defined either

            * numerically (by a :class:`._Numerical` object), or
            * programmatically (by a Python/Numpy/SciPy method).

    But, all :class:`.FuzzyNumber`\\ s have an attribute, ``e`` for "elsewhere", describing the truth for every real
    number where it is not explicitly defined.  So, all :class:`.FuzzyNumber`\\ s are defined for all ``float`` inputs.

    It implements:

        * :meth:`.display`, which shows a picture of the function;
        * :meth:`.map`, which turns the function into a callable object;
        * :meth:`.FuzzyNumber.crisp`, which finds the best real number equivalent for it (by defuzzification); and
        * ``float()``, which does the same using only default parameters.

    Its subclasses implement:

        * :meth:`.t`, which returns the truth of a given value; and
        * :meth:`._get_numerical`, which returns a numerical representation
          of :math:`t(v)`, a :class:`._Numerical`; and
        * :meth:`._get_domain`, which returns the domain over which
          the continuous part of :math:`t(v)` is explicitly defined."""

    maximum_precision = 1e6
    """Limits the number of samples in numerical representations.
    
    Certain operations (e.g. the reciprocal) can explode the defined domain and lead to calls for huge numbers of 
    samples.  This constant prevents overflows in such cases.  If you have an expression with this problem, set 
    ``allowed_domain`` to the region you're interested it---then all the precision you use will be focused on 
    the domain that matters."""

    def __init__(self, elsewhere: float = 0):
        """Args:
            elsewhere:  The truth for values that are otherwise undefined (the default is 0)."""
        self.e = elsewhere  # Must be on [0,1].  no checks for now.

    # abstracts
    @abstractmethod
    def _get_domain(self) -> Union[Domain, None]:
        """The intersection of the expressed and allowed domains.

        No need for this to impose any allowed_domain!!!!!!!?
        going up the tree to find the total domain: higher d = d_op(operand d).
        going down the tree to say only sample restricted d:
        lower allowed d = d_op_inv(allowed d).
        eg, if the operator is "+5" and operand d= 3,5, send up 3+5,5+5 = 8,10.
        if only interested in result d = 0,1, only sample (send down) 0-5,1-5 = -5,-4
        "up" = _get_domain for precision calc
        "down" = _get_numerical to ask down to sample c narrowly
        (and its return can also trim xp)



        By "expressed" I mean the single contiguous domain over which the fuzzy number is defined explicitly as a
        continuous function.  This does not include definitions of exceptional points or of the default truth
        elsewhere.  Since a fuzzy number might be an expression involving an arbitrary number of logic and math
        operators, this isn't a simple question, but it can be handled by recursive calls.

        Args:
            allowed_domain: The extreme domain on which a result is acceptable.

        Return:
            The intersection of the defined and acceptable domains.  If the answer is ``None``, you might want to
            reconsider ``allowed_domain`` (but: some exceptional points might still be in play and "elsewhere" is
            always in play.)"""

    @abstractmethod
    def _get_numerical(self, precision: int, allowed_domain: Domain = None) -> _Numerical:
        """Obtains and returns a numerical representation of itself.

        This is where the work is done in each subclass.  In a :class:`Literal` number, the work is to sample its
        analytical function appropriately.  In an :class:`Operator`, the work is to call for evaluations
        of its operands, operate on them, and return the numerical result.

        Arguments:
            precision: The number of sample points used to cover the defined domain of the continuous function.
                This determines how accurately the result represents the original.  It will have been calculated
                from the caller's desired ``resolution``, the minimum significant difference in value.
                (In practice, two guard points will always be added outside the domain.)
            allowed_domain: The extreme domain on which a result is acceptable.

        Note:
            :class:`.Operator`\\ s hold their operands, which are fuzzy numbers---either :class:`.Literal`\\ s or more
            Operators---so calling this method on an expression built up in this way can result in many recursive
            calls.  By restricting the requested domain with ``allowed_domain`` at each step, calculations avoid
            considering inconsequential values, and concentrate all of their precision (sample points) only
            on stretches of domain that will have a bearing on the final result.

        Note:
            In cases where ``precision == 0``, the continuous part is excluded from calculations.

        Return:
            A numerical representation of ``self``, i.e., of the :math:`t(v)` that defines the fuzzy number.
        """

    @abstractmethod
    def t(self, v: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Given a value, return its truth.

        It should refer to, in order of priority:  the exceptional points, the continuous function,
        and the default truth "elsewhere".  The main case for this method's use is to check on how good
        the result of :meth:`.FuzzyNumber.crisp` is.

        Args:
            v: any real number, a proposed value.

        Returns:
            The suitability of the proposed value, a measure of truth in the range [0,1],
            the degree to which it is a member of the result."""

    # implementations: private helpers
    def _expression_as_numerical(self, resolution: float, allowed_domain: Tuple[float, float] = None) -> _Numerical:
        """A helper that gets the numerical representation of the entire expression, restricted to ``allowed_domain``,
        at the required ``resolution``.  It is used by :meth:`.FuzzyNumber.crisp`, :meth:`.map`. and :meth:`.display`.
        Its main task is to calculate the required precision from the given resolution and to make sure the resulting
        domain is properly restricted.  ``resolution`` must be > 0."""
        if isinstance(allowed_domain, Tuple):
            allowed_domain = Domain(allowed_domain)
        domain = self._get_domain()  # What the raw, unrestricted result would be.
        if allowed_domain is not None:
            domain = allowed_domain.intersection(domain)  # What the restricted result will be...
        span = None if domain is None else domain.span()  # ...for the precision calculation:
        if span is None:
            precision = 0  # continuous domain is excluded. exceptional points might exist
        elif span == 0:
            precision = 1  # continuous domain is a point.  Trouble?
        else:
            if span > FuzzyNumber.maximum_precision * resolution:
                precision = FuzzyNumber.maximum_precision
            else:
                precision = ceil(span / resolution)  # number of sample points in each continuous part
            precision = precision + 1 if (precision % 2) == 0 else precision  # Insist on odd precisions.
        numerical = self._get_numerical(precision, allowed_domain)  # only seeks allowed domain
        if allowed_domain is not None:  # Impose an extreme domain, if required:
            numerical = _Numerical._impose_domain(numerical, allowed_domain)  # noqa Discard some exceptional points
            # ---has to be done separately because continuous was restricted (above) to its intersection with allowed.
            # hmmm.  a parallel xp_domain could keep track of it and avoid unnecessary calculation.
            # It might be more efficient, but it wouldn't affect precision.
        return numerical

    # implementations: public
    def crisp(self, resolution: float, allowed_domain: Domain = None, crisper: Crisper = None) -> float:
        """Returns a crisp value that is equivalent to ``self``\\ s fuzzy value.

        Arguments:
            resolution: The smallest significant difference between values in the numerical representation.
                This controls the accuracy of the result (a smaller resolution is better).
                Also, consider that a coarse mesh in the numerical representation might miss narrow peaks.
                (Exceptional points defined explicitly are unaffected by resolution.)
            allowed_domain: bounds the domain of the result in case the answer must be limited,
                e.g., if tempo must be on [30, 200] bpm, or temperature must be on [-273.15, 100]°C.
            crisper:  The :class:`.Crisper`  object that performs the defuzzification.
                If none is indicated, :attr:`.default_crisper` is used.

        Return:
            The crisp equivalent of this fuzzy number, according to ``crisper``, considering only ``allowed_domain``.
        """
        numerical = self._expression_as_numerical(resolution, allowed_domain)
        # Defuzzify the fuzzy number to obtain its crisp value:
        if crisper is None:
            crisper = default_crisper
        v = crisper.defuzzify(numerical)
        return v

    def __float__(self) -> float:
        """Returns the crisp float value, via :meth:`.FuzzyNumber.crisp`, using only default parameters."""
        return self.crisp(default_resolution)

    def map(self, range: Tuple[float, float] = (0, 1), map: str = "lin") -> Map:
        """Creates a callable object that maps the :math:`t(v)` of ``self`` to the real numbers.

        A :class:`.FuzzyNumber` is a function of truth vs. value.  Sometimes that function is itself a useful result.
        It can be used in crisp mathematical expressions via the callable :class:`.Map` object returned by
        this method.

        The range of the internal function is restricted to [0,1].  To make it more convenient, the parameters
        allow you to translate this to ``range`` via a ``map`` (linear, logarithmic, or exponential).  This should
        make the result more easily adaptable.

        This method returns a callable object that stores the expression as a tree of operators and literals.
        The :meth:`.numerical_map` method is similar, but stores the expression as a :class:`._Numerical`.
        This one is more accurate, the other may be more efficient for some complex expressions.

    Args:
        range:  Translates the range of the internal function to the indicated range.  See :meth:`.Truth.scale`.
        map:  And does so via linear, logarithmic, or exponential mapping.  See :meth:`.Truth.scale`.

    Returns:
        A callable object that can be used as a mathematical function.

    Example:
        | ``loudness = amplitude_vs_pitch.map(range=(0,96), map = "log"")``
        | ``y = loudness(pitch)``
        """
        return Map(self, range, map)


    def numerical_map(self, resolution: float, allowed_domain: Tuple[float, float] = None,
            range: Tuple[float, float] = (0, 1), map: str = "lin",
            interp: Interpolator = None) -> Map:
        """Creates a callable object that maps the :math:`t(v)` of ``self`` to the real numbers.

        A :class:`.FuzzyNumber` is a function of truth vs. value.  Sometimes that function is itself a useful result.
        It can be used in crisp mathematical expressions via the callable :class:`.NumericalMap` object returned by
        this method.

        The range of the internal function is restricted to [0,1].  To make it more convenient, the parameters
        allow you to translate this to ``range`` via a ``map`` (linear, logarithmic, or exponential).  This should
        make the result more easily adaptable.

        This method returns a callable object that stores the expression as a :class:`._Numerical`.
        The :meth:`.map` method is similar, but stores the expression as a tree of operators and literals.
        The other is more accurate, but this may be more efficient for some complex expressions.

    Args:
        resolution: The distance between sample values in the numerical representation.
            This controls how accurately the :class:`.NumericalMap` represents the original
            (a smaller resolution is better).  Explicitly defined exceptional points are unaffected by resolution.
        allowed_domain:  Restricts the defined domain of the result to no more than this parameter, i.e., discarding
            any exceptional points or continuous domain outside it---these will return the default "elsewhere" truth.
        range:  Translates the range of the internal function to the indicated range.  See :meth:`.Truth.scale`.
        map:  And does so via linear, logarithmic, or exponential mapping.  See :meth:`.Truth.scale`.
        interp:  An :class:`.crisp.Interpolator` object for interpolating between the sample points.
            If none is indicated, :attr:`.default_interpolator` is used.

    Returns:
        A callable object that can be used as a mathematical function.

    Example:
        | ``loudness = amplitude_vs_pitch.map(range=(0,96), map = "log"")``
        | ``y = loudness(pitch)``
        """
        numerical = self._expression_as_numerical(resolution, allowed_domain)
        return NumericalMap(numerical, range, map, interp)

    def display(self, resolution: float = None, domain: Domain = None, display_percentage: float = 25) -> None:
        # TODO: Needs work
        """Displays the t(v) that is the :class:`.FuzzyNumber`.

        Args:
            resolution:  The minimum significant difference in value for the fuzzy number.
                Default: the equivalent of one pixel.
            domain:  The domain of values to display. Default: the defined continuous domain.
            display_percentage:  The percentage of your monitor area to be taken up by the plot itself.
        """
        display_factor = (display_percentage / 100) ** .5
        px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
        plt.subplots(figsize=(monitor_w * display_factor * px, monitor_h * display_factor * px))
        if resolution is None:
            defined_c_domain = self._get_domain()
            if defined_c_domain is not None and domain is not None:
                defined_c_domain = defined_c_domain.intersection(domain)
            if defined_c_domain is not None:
                resolution = defined_c_domain.span() / (monitor_w * display_factor)
        fn = self._expression_as_numerical(resolution, domain)
        if domain is None:  # set the domain to show all exceptional points.
            c_domain = None if fn.cv is None else fn.cd
            x_domain = None if fn.xv is None else Domain((np.amin(fn.xv), np.amax(fn.xv)))
            c_domain = x_domain if c_domain is None else c_domain
            x_domain = c_domain if x_domain is None else x_domain
            domain = Domain((0, 0)) if ((c_domain is None) and (x_domain is None)) else c_domain.union(x_domain)
        if domain[0] == domain[1]:
            domain = Domain((domain[0]-.5, domain[0]+.5))
        # plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        plt.xlim(*domain)  # domain to plot
        plt.ylim(-.01, 1.01)
        plt.grid()
        v = fn.cv
        s = fn.ct
        if v is not None:
            plt.plot(v, s, color="red")
        v = np.array([*domain])
        s = np.array([fn.e, fn.e])
        plt.plot(v, s, color="yellow")
        if fn.xv is not None:
            xpv = fn.xv
            xps = fn.xt
            plt.plot(xpv, xps, "o", markersize=3, color="blue")
        # plt.title("Line graph")   # There's no way to get the name of the variable.
        plt.show()


class _Numerical(FuzzyNumber):  # user doesn't see it!  dataclass?

    def __init__(self, cd: Domain = None, cn: int = 0, cv: np.ndarray = None, ct: np.ndarray = None,
                 xv: np.ndarray = None, xt: np.ndarray = None, e: float = 0):
        """Args:
            cd: The domain over which the continuous numerical function is defined.  Valid if d[1]>=d[0].
            cn: The number of samples on  ``cd`` (not counting guard points). If cn>=0, no numerical function
                will be constructed. (And if cn=={1,2}, Literal will make it 3.)
            cv, ct: The continuous function's sample points: values, truths.  (Their len() must be equal.)
            xv, xt: The exceptional points: values, truths.   (Their len() must be equal.)
            e: The truth for all values not in ``cd`` or ``xv``---elsewhere.
            cv, xv: For each:  all elements must be unique.
            ct, xt, e:  For each:  all elements For each:  on [0,1] except for ct[0] and ct[-1] which should be finite.

            I'd check all of these, but this is private, so I'm trusting myself.  """
        self.cd = cd  # domain of continuous part
        if cn <= 0:
            cn, self.cd = 0, None
        self.cn = cn  # number of samples in  continuous part
        self.cv = cv  # continuous part: value samples
        self.ct = ct  # continuous part: truth samples
        self.xv = xv  # exceptional points: value samples
        self.xt = xt  # exceptional points: truth samples
        super().__init__(e)  # elsewhere

    def __str__(self):
        if self.cd is None:
            c = "no continuous part; "
        else:
            c = str(f"continuous part: {self.cn} points on domain {self.cd}, elsewhere: {self.e:.3g}; ")
        if self.xv is None:
            x = "no exceptional points."
        else:
            x = str(f"exceptional points: \n {np.dstack((self.xv, self.xt))[0]}")
        return str(f"_Numerical: {c} {x}.")

    def _get_domain(self, allowed_domain: Domain = None) -> Union[Domain, None]:
        if allowed_domain is None:
            return self.cd
        else:
            return self.cd.intersection(allowed_domain)

    def _get_numerical(self, precision: int, allowed_domain: Domain = None) -> _Numerical:
        """A :class:`._Numerical` will not resample itself, because that would lower its quality; therefore,
        it ignores the given precision, unless a precision of 0 insists on eliminating the continuous function.

        This method is designed called only on Literals and operands within Operators.
        Some subclasses of :class:`._Numerical` may be used as literals that define sets of points without
        a continuous function---e.g., :class:`.DPoints` and :class:`.Exactly`.  In these cases, this method's only
        task is to impose the allowed_domain by possibly discarding some exceptional points.  I don't foresee any
        situation where the parent _Numerical would be asked to return itself---no harm if it does, though.  """
        numerical = self if allowed_domain is None else _Numerical._impose_domain(self, allowed_domain)
        if precision == 0:
            numerical.cd = None
            numerical.cn = 0
            numerical.cv = None
            numerical.ct = None
        return numerical

    @staticmethod
    def _impose_domain(num: _Numerical, allowed_domain: Domain) -> _Numerical:
        """Returns a version of itself without exceptional points outside ``allowed_domain``.

        The continuous domain should have been restricted already by :meth:`._get_numerical` calls
        on operands---ultimately, these calls reach literals that will sample only the domain necessary to describe
        the allowed_domain of the caller.  In other words, discarding (xv, xt) points makes sense, but resampling
        or discarding (cv, ct) points would lower the quality, and needn't be done anyway.  So, here we only
        touch (xv, xt)."""
        if allowed_domain is None:
            return num
        new_xv, new_xt = num.xv, num.xt
        if num.xv is not None:
            i = np.where((num.xv < allowed_domain[0]))
            new_xv = np.delete(num.xv, i)
            new_xt = np.delete(num.xt, i)
            if new_xv is not None:
                i = np.where((new_xv > allowed_domain[1]))
                new_xv = np.delete(new_xv, i)
                new_xt = np.delete(new_xt, i)
            if len(new_xv) == 0:
                new_xv = new_xt = None
        actual_domain = allowed_domain.intersection(num.cd)
        return _Numerical(actual_domain, num.cn, num.cv, num.ct, new_xv, new_xt, num.e)

    def _sample(self, v: Union[np.ndarray, float], interp: Interpolator = None) -> Union[np.ndarray, float]:
        """Returns the truth of given values, not considering exceptional points."""
        a = isinstance(v, np.ndarray)
        v = np.atleast_1d(v)
        if self.cd is None:
            t = np.full_like(v, self.e, dtype=float)
        else:
            if interp is None:
                interp = default_interpolator
            t = interp.interpolate(v, self.cv, self.ct)
            outside = np.where((v < self.cv[0]) | (v > self.cv[-1]))
            t[outside] = self.e
            t = _guard(t)
        return t if a else t[0]

    def t(self, v: Union[np.ndarray, float], interp: Interpolator = None) -> Union[np.ndarray, float]:
        """Returns the truth of given values."""
        a = isinstance(v, np.ndarray)
        v = np.atleast_1d(v)
        c_e = self._sample(v, interp)
        if self.xv is not None:
            x = np.where(np.in1d(v, self.xv, assume_unique=True))  # where v is in xv
            y = np.where(np.in1d(self.xv, v, assume_unique=True))  # where xv is in v
            c_e[x] = self.xt[y]
        return c_e if a else c_e[0]
