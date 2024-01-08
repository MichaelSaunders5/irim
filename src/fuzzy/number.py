"""applies the idea of fuzzy truth to the representation of numbers.

This module contains the public methods for obtaining useful results from fuzzy expressions---``crisp``, ``map``,
``t``, ``display``, and their variants.  This is the main interest for users, but privately, it contains
the basic apparatus for representing fuzzy numbers.  Just as :class:`Truth` objects are analogous
to ``bool``, :class:`.FuzzyNumber` objects are analogous to ``float``.  I'll introduce their representation,
give an example of their use, discuss the public methods mentioned above, and finally, for the curious,
sketch what goes on behind the scenes to make the module work.  Since fuzzy expressions are equivalent to their
resulting fuzzy number, I will speak of them interchangably.



Introduction
------------


A fuzzy number is a function of truth vs. value, :math:`t(v)`.  We may think of it as describing the
*suitability* of each possible value for some purpose.  In this way, we might state our opinions on a subject
as "preference curves".  Fuzzy numbers might also represent empirical knowledge about how one variable varies with
another in some fixed range.  And so, they can model the real world.  By combining these fuzzy numbers with arithmetic
and mathematical operators into expressions, we can represent more complex ideas and model the interaction of the
real world with our opinions.  Finally, we have one or more fuzzy numbers from which we obtain results in the form of
crisp numbers, control functions, or visualizations---results that we can act upon.
Ideally, the end of the process will make some aspect of the real world conform to our preferences.

Fuzzy solutions to complex problems can look much like the crisp solutions.  Consider a situation modeled by crisp
equations.  When the operators of those equations are made fuzzy, and the independent variables can be fuzzy, every
contingency can be planned for, and every subtle effect can be taken into account.

Suppose we state our opinions on preferred room temperature as fuzzy numbers.  Yours may differ from mine, but we
can combine them by ANDing them together and crsiping the result to find a compromise.  Enrich the model with
equations for energy prices, budget concerns, expected temperatures, time spent in the building, etc., and we can
automate our argument about the thermostat.

So, make all the natural-language statements you can about a system, its problems and heuristics.  They are already
very close to fuzzy statements, and are easily converted.  Your knowledge of a domain can easily become a very
sophisticated computer program automating your own judgment.




How to Use It
-------------

There are two worlds in play.  There is the familiar world of absolute truths and precise numbers that we will
call "crisp".  There is also the world of partial truths and indefinite numbers that we will call "fuzzy".
In the fuzzy world, the vagueness is measured with precision, so we can compute with it.  The intricacies happen
largely out of sight, so our calculations and reasoning retain their familiar form.  The main differences are:

    * We enter the fuzzy world by defining fuzzy numbers, either:

        * Literally---via intuitive parameters (as in the room temperature example, where a few simple parameters
          suffice to define a vague range of values); or,
        * Analytically---by adopting *as* fuzzy numbers informative functions that vary between established limits
          (as in the dissonance example, where a highly structured function depends upon complex parameters, like
          sound spectra).

    * Within the fuzzy world, we can use not only the familiar arithmetic operators in the usual way, but we can also
      combine numbers with the logical operators (*and*, *or*, *not*, and so on) to compute statements of reasoning.
    * The notion of weighting terms in an equation still exists, but in two forms: trinary
      (by :meth:`.Operator.weight`, ``//``) and binary (by :meth:`.Operator.focus`, ``^``).
    * We reënter the crisp world by the methods:

        * :meth:`.FuzzyNumber.crisp`, to render a :class:`.FuzzyNumber` as a single ``float``, indicating the
          most suitable value, ; or
        * :meth:`.map`, to make the suitability vs. value function of the fuzzy number a callable object,
          usable in crisp expressions, ideal for use as a control function; or
        * :meth:`.display`, to visualize the result.


A user will normally begin by creating some literal truth and number objects using :class:`.Truth` and, for the
numbers, subclasses of :class:`.Literal`.  These are then used as operands in expressions, and the expressions stored in
variables.  The operators are methods of class :class:`.Operator` and can be called either as normal Python
methods like ``c = Operator.mul(a,b)`` or ``c = a.mul(b)`` or simply with overloaded symbols, like ``c = a * b``.
Finally, a result is obtained from the expression by the public methods of :class:`.FuzzyNumber`.  So,
one might write something like this:

    | ``fuel_price = Truth(.8)`` # It's a bit high.
    | ``urgency = Truth(.3)``   # Arrival time isn't too critical.
    | ``fuel_efficiency = Bell(50, 20)``    # We get the best milage at 50 m.p.h.
    | ``promptness = Triangle(30, 70, 120)``    # ETA would be {too late, perfect, too soon}.
    | ``fuzzy_speed = (fuel_efficiency ^ fuel_price) & (promptness // urgency)``
    | ``crisp_speed = fuzzy_speed.crisp()``    # This is the ideal speed.
    | ``print(fuzzy_speed.t(crisp_speed))``    # Looks good.
    | ``accelerator(crisp_speed)``    # Step on it.

to decide the speed of a journey, weighing the desire to minimize cost with the need to arrive at an appointed time.
This is a toy example---expressions can be arbitrarily complex and these techniques may be embedded in conventional
algorithms---but you can see how sophisticated behavior can result from this technique.

Literals and operators will be detailed in the next two modules.  Three parts of the present module are public:

    * The class :class:`.Domain`,
    * The module attributes,
    * Public methods of :class:`.FuzzyNumber`.

A :class:`.Domain` object represents a single contiguous interval on the domain of real numbers.
Domains are used internally for the business of numerical representation; I don't expect most users will need them.
The module attributes act as rare parameters in the "environment" of the :mod:`.fuzzy` package.  All of them are
explained in :meth:`fuzzy.operator.fuzzy_ctrl`.   By far the most important part of the module to most users will be
the public methods of :class:`.FuzzyNumber`.  They obtain usable results from fuzzy expressions.

There are six of them:

    * :meth:`.FuzzyNumber.t` --- returns the truth of a given value.
    * :meth:`.FuzzyNumber.map` --- returns a callable object that functions as ``t`` above,
      with an optional mapping (linear, logarithmic, or exponential) of *truth* to an external variable.
    * :meth:`.FuzzyNumber.numerical_map` --- does the same as ``map`` above, but storing the
      :math:`t(v)` function numerically rather than as a tree of operators and literals.
    * :meth:`.FuzzyNumber.crisp` --- returns a "defuzzified" version of the fuzzy number as a ``float``, the value
      of the number you may regard as the truest.
    * ``float()`` --- returns the same as :meth:`.FuzzyNumber.crisp` above, using the defaults (a default resolution
      and no domain restriction.
    * :meth:`.FuzzyNumber.display` --- displays a plot of the fuzzy number---its :math:`t(v)`.

Probably, the most used will be :meth:`.FuzzyNumber.crisp`, which produces a single number that can be acted upon.
The :meth:`.FuzzyNumber.map` method turns the fuzzy number into a function that can control a process, preserving all
the nuances of the fuzzy expression.  The :meth:`.FuzzyNumber.t` method can be used to measure the desirability of a
result, or to choose between proposed alternatives.  The :meth:`.FuzzyNumber.display` method may be the most useful
of all, allowing the user to see all the implications of an expression of fuzzy reasoning, perhaps revealing insights
or alternatives not suggested by ``crisp``.  In any of these cases, one can use an ``allowed_domain`` parameter to
focus attention (and calculation) on the area of interest.  With some expressions having large domains, this might be
necessary.  In the case of ``crisp`` and ``numerical_map``, it is also possible and highly recommended to set the
``resolution``, as only the user will know what is appropriate.


How It Works
------------


A fuzzy number is a function of truth vs. value, :math:`t(v)`.  Representations of this in the :mod:`.fuzzy` package
conceive of the function in three parts:

    * A default truth level that is reported for otherwise undefined values.  Sitting on top of this,
    * A segment of :math:`t(v)` defined over a single contiguous subdomain of the real numbers.  In :class:`.Literal`
      classes, this is defined analytically, by a Python method; in the :class:`._Numerical` class used internally for
      calculations, it is an array of :math:`(v, t)` sample points.  Overlaid on all of this is a sprinkling of
    * Discrete, "exceptional" points, represented as a set of :math:`(v, t)` pairs.

Only the first element, the default truth, is not optional.


:class:`.FuzzyNumber` is an abstract class from which the subclasses :class:`.Operator`
and :class:`._Numerical` descend; :class:`.Literal`, in turn, descends from :class:`.Operator`.
:class:`._Numerical` objects represent a fuzzy number numerically and are used for internal calculations.
:class:`.Operator`\\ s contain one or more operands which are
also :class:`.FuzzyNumber`\\ s---either :class:`.Operator`\\ s or :class:`.Literal`\\ s.  So, an expression
(which can be held in an ordinary Python variable) is a tree of :class:`.Operator`\\ s the "leaves" of which
are :class:`.Literal` objects.  Some of the :class:`.FuzzyNumber` methods (``t``, ``map``) do nothing but make
recursive calls down the tree to obtain their result.  Others (``crisp``, ``float``, ``numerical_map``,
and ``display``), produce a :class:`._Numerical` fuzzy number representing the result of the expression as an
intermediate step to their final result (a crisp value ``float``, a callable object representing :math:`t(v)`, or a
graphical display of :math:`t(v)`).

All descendants of :class:`.FuzzyNumber` must implement the public methods already mentioned:

    * :meth:`.FuzzyNumber.t`,
    * :meth:`.FuzzyNumber.map`,
    * :meth:`.FuzzyNumber.numerical_map`,
    * :meth:`.FuzzyNumber.crisp` (``float()`` is overloaded to call this without parameters),
    * :meth:`.FuzzyNumber.display`,

:meth:`.FuzzyNumber.t` must return the truth of the number at a given value.   In :class:`.Literal`\\ s, the set of
exceptional pointsif first consulted.  If the given value is not among them, it looks for it on the domain of its
continuous function.  the :meth:`.Literal_sample` method returns the truth of the value, as it is
defined on the continuous domain.  In :class:`.Operator`\\ s, the operands are queried with their
:meth:`.FuzzyNumber.t` methods, and the results are operated upon with the operator's :meth:`.Operator._op`
method to obtain the result---this can result in arbitrarily complex recursive calls down the tree to the
:class:`.Literal`\\ s.  In :class:`._Numerical`\\ s, :math:`t(v)` on the defined domain is obtained by calling
an :class:`.Interpolator` with the number's sample points.  Failing all of this---if the given value is not an
explicitly defined point or on the defined continuous domain, the default, "elsewhere" truth is returned.

:meth:`.FuzzyNumber.map` simply packages fuzzy expressions (trees of :class:`.Operator`\\ s and :class:`.Literal`\\ s)
into a callable object along with a handy mapping of the result.  So ``expression.t(v)`` can become ``function(v)``
in your code, where ``function`` is an object that can be passed around and sent to methods as needed.  Since it's
unlikely the output variable that you need is on [0,1], callable scales it to your desired range, by a linear,
logarithmic, or exponential mapping.   The scaling and mapping are defined at the time the callable is created.

There remaining methods require the creation of a numerical representation of the fuzzy expression.  This is the heart
of the module, and will be described below, with the private methods.

:meth:`.FuzzyNumber.numerical_map` does the same as :meth:`.FuzzyNumber.map`, but instead of packaging a tree and
making recursive calls down to all the literals, it creates a numerical representation of the expression (a
:class:`._Numerical`) and packages that.  Although less accurate than :meth:`.FuzzyNumber.map`, it can be made as
accurate as it needs to be, and may be more efficient in some cases.

:meth:`.FuzzyNumber.crisp` creates a :class:`._Numerical` and delegates the business of crisping it to a
:class:`.Crisper` object, as there are many defuzzification algorithms to choose from.

:meth:`.FuzzyNumber.display`, creates a :class:`._Numerical` and graphs it via :mod:`.matplotlib`.  The default
settings choose a resolution of one sample point per pixel, so that the function is presented fairly with no
computation wasted.   The continuous part is graphed as a red line, the discrete points as blue dots, and the
default truth as a yellow horizontal line.  As usual, a domain parameter allows one to focus on areas of interest.

All descendants of :class:`.FuzzyNumber` also have the following private methods, used internally to provide
the functionality of the above:

    * :meth:`.FuzzyNumber._get_domain`, which reports the interval of real numbers on which the continuous part of
      the :math:`t(v)` function is defined (explicitly defined discrete points may lie outside this).
    * :meth:`.FuzzyNumber._get_numerical`, which returns a numerical representation of the fuzzy number, a
      :class:`._Numerical` object, given an ``allowed_domain`` and a ``precision`` (an integer number of sample
      points used to represent function over its continuous domain).
    * :meth:`.FuzzyNumber._expression_as_numerical`, does the same as :meth:`.FuzzyNumber._get_numerical`, but for
      a given ``resolution`` (smallest significant difference in value) instead of precision.

The first two are abstract, and must be implemented by descendants of :class:`.FuzzyNumber`.  The third is implemented
in  :class:`.FuzzyNumber` itself.

:meth:`.FuzzyNumber._get_domain` is easy for :class:`.Literal`\\ s and :class:`._Numerical`\\ s, since it is simply
part of the object's definition.   In :class:`.Operator` objects, the operator calls the method on its own operands,
obtaining one or more :class:`.Domain` objects, and then putting them through its own :meth:`.Operator.d_op` method
to predict the domain of all its possible results.

When :meth:`.FuzzyNumber._get_numerical` is called on a :class:`.Literal` object, it samples its own continuous domain
(ignoring exceptional points) by calling its :meth:`.Literal_sample` method on a domain given it by the caller
(an important point, discussed below).
(N.B., the default sampling of literals is by Chebyshev collocation points, but uniformly-spaced samples are an
option; in any case, guard points outside the defined domain are taken to ensure good interpolation near the edges.)
When :meth:`.FuzzyNumber._get_numerical` is called on :class:`.Operator`\\ s, they first call
:meth:`.FuzzyNumber._get_numerical` on their operands to obtain :class:`._Numerical` versions of them.  They then
do what they have to do with these to obtain a :class:`._Numerical` result.  This can be quite involved, and is
described in :mod:`fuzzy.operator`.

:meth:`.FuzzyNumber._expression_as_numerical` is the method actually called by the public methods that require a
numerical version of their fuzzy number.  The user calling one of those methods knows something the :mod:`fuzzy`
package cannot predict:  the resolution of the result required for the user's application.  The user, on the other
hand, cannot easily predict the precision (the number of sample points) that will be needed to achieve this.
:meth:`.FuzzyNumber._expression_as_numerical` translates the user's required ``resolution`` into the ``precision``
parameter for the internal :meth:`.FuzzyNumber._get_numerical` call.
The required precision is the span of the "natural domain" that the expression defines divided by the given resolution.
This domain is found by querying the expression with :meth:`.FuzzyNumber._get_domain`, a call which propagates all the
way down to every literal, returning results back to the root of the tree through the transforms of each operator's
:meth:`.Operator.d_op` method.  This is the domain over which the expression is defined as a continuous function, if
there are no other constraints.

There may well be other constraints.   The user may only be interested in solutions over a subdomain of the natural
domain, some values for which the function is defined mathematically may be physically nonsensical, and some natural
domains may be impractically large, e.g., if division by small numbers is involved.  The user therefore, may restrict
the ``allowed_domain`` of the result by a parameter.  It's the intersection of this and the natural domain that is
used to determine the precision.  It also determines the  domain given in every downward call
of :meth:`.FuzzyNumber._get_numerical`.  When operators make this call on their operands, they must transform the
domain of the result to that of the operands---the inverse of the domain transformation of :meth:`.Operator.d_op`.
In this way, all the sample points are applied only to regions of literals that will make a difference in the
domain of interest to the user---no precision is wasted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import ceil  # , log, sqrt
from typing import Union, Tuple  # , ClassVar,

import matplotlib.pyplot as plt
import numpy as np
from screeninfo import get_monitors

from fuzzy.crisp import Crisper, MedMax, Interpolator, Map, NumericalMap
from fuzzy.truth import TruthOperand, default_threshold

monitor_w, monitor_h = 1920, 1080  # When the module loads, find the monitor dimensions, for use by :meth:`.display`.
for m in get_monitors():
    if m.is_primary:
        if (monitor_w is not None) and (monitor_w > 0):
            monitor_w, monitor_h = m.width, m.height

default_resolution: float = .01  # Needed when calling float(FuzzyNumber).
"""The minimum difference in value to be considered. 

Its choice depends on the units you are using and the system you are modeling.  It is impossible for me to guess at 
what is best for your purpose, but, in some programming situations, there is no place for an explicit parameter to be 
given.  You should state the resolution you require where possible, or set the default to it for your convenience.
The default is used in these cases:

* The :meth:`.FuzzyNumer.crisp` method (as the default) and by ``float()``;
* All comparisons, e.g., :meth:`.__gt__` and their operators, e.g., ``>``.
* The constructor of :class:`._Numerical` (as the default), including 
  when :meth:`.FuzzyNumber._get_numerical` is called.
  
"""

default_interpolator: Interpolator = Interpolator(kind="linear")
"""An :class:`.crisp.Interpolator` (an interpolation algorithm, together with any parameters it may need).

Its uses include:

* Constructing a :class:`literal.CPoints` fuzzy number (to interpolate the sample points between the knots).
* Calculating the truth in the continuous domain of a :class:`._Numerical` 
  (to interpolate between the sample points)."""

default_crisper: Crisper = MedMax()
"""A :class:`.crisp.Crisper` (an interpolation algorithm, together with any parameters it may need).

It is used by the methods :meth:`.FuzzyNumber.crisp` (by default) and ``float()``."""

default_sampling_method: str = "Chebyshev"
"""How :class:`.Literal`\\ s are sampled to create :class:`._Numerical`\\ s.

The options are ``Chebyshev`` (producing near-minimax approximations) or ``uniform``."""


def _guard(t: TruthOperand) -> Union[float, np.ndarray]:
    """A private helper function to deal with exceptional truths {-inf, nan, +inf},
    interpreting them as {0, :attr:`Truth.default_threshold`, 1}, and ensuring validity by clipping.

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
    """Finds the truth for a given value, if it is present in a set of discrete points.
    This is a helper called by :meth:`._Numerical.t`.

    Args:
        v: A value to be checked.
        xp:  An array of (value, truth) points.

        Returns:
            If ``v`` is in ``xv``, the corresponding element of ``xt``, otherwise, ``None``.
        """
    if (xv is None) or (xt is None):
        return None
    i = np.where(xv == v)[0]
    if len(i) == 0:
        return None
    return xt[i][0]


class Domain(Tuple):
    """Indicates a contiguous set of real numbers.

    This is used in many places when working with fuzzy numbers, e.g., to indicate the values for which a number is
    defined as a continuous function of truth.  A domain of ``None`` indicates the empty domain. The class provides
    several convenient methods for working with them: :meth:`.span`, :meth:`.center`, :meth:`.intersection`,
    :meth:`.union`, and :meth:`.contains`.

    """

    def __new__(cls, d: Tuple[float, float]) -> Domain:
        """Args:
            d: A tuple of floats indicating the extremes of the domain.  They can be equal, but the second must not be
            greater than the first, or an exception will be raised.  (If you need to create a :class:`.Domain` out of
            numbers that may have gotten twisted around, use :meth:`.Domain.sort`.)

        Returns:
            A :class:`.Domain` of the extent indicated."""
        if d is not None:
            if d[1] < d[0]:
                raise ValueError(f"Domains must have d[1] >= d[0]. ({d[0]}, {d[1]}) is invalid.")
        return super(Domain, cls).__new__(cls, d)

    def __str__(self):
        return str(f"({self[0]:,.4g}, {self[1]:,.4g})")

    def span(self) -> float:
        """Returns the length of the domain.

        Returns:
            The length of the real numbers occupied by ``self``."""
        if self is None:
            return 0
        return self[1] - self[0]

    def center(self) -> float:
        """Returns the central coördinate of the domain.

        Returns:
            The midpoint of ``self``."""
        return self[0] + self.span() / 2

    def intersection(self, d: Domain) -> Union[Domain, None]:
        """Returns the domain that is ``self`` ∩ ``d``.

        Args:
            A domain to be combined with ``self``, by cutting away all of ``self`` not in ``d``.

        Returns:
            The domain of all points in both ``self`` and ``d``.
            This may be the empty domain, indicated by ``None``."""
        if d is None:
            return None
        new_min = max(self[0], d[0])
        new_max = min(self[1], d[1])
        return None if new_min > new_max else Domain((new_min, new_max))

    def union(self, d: Domain) -> Domain:
        """Returns the domain that is ``self`` ∪ ``d``.

        Args:
            A domain to be combined with ``self``, by adding all of ``d`` not in ``self``, and all points in-between.

        Returns:
            The domain of all points in either ``self``, or ``d``, or in-between; i.e., all points bounded by the
            extreme bounds of ``self`` and ``d``.  ``None`` is returned only of both are the empty domain."""
        if d is None:
            return self
        if self is None:
            return d
        new_min = min(self[0], d[0])
        new_max = max(self[1], d[1])
        return Domain((new_min, new_max))

    def contains(self, v: float) -> bool:
        """True if and only if ``v`` is on ``self``.

        Args:
            A real number to be tested for membership in the domain ``self``.

        Returns:
            Whether ``v`` is in ``self``.
        """
        if self is None:
            return False
        else:
            return not ((v < self[0]) or (v > self[1]))

    @staticmethod
    def sort(*d: float) -> Union[Domain, None]:
        """Constructs a :class:`Domain` without regard to the ordering of the two bounds.

        Args:
            Two real numbers (only the first two are considered), to act as the bounds of the domain.

        Returns:
            A :class:`Domain` having the indicated bounds."""
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
        * :meth:`.map`, which turns the function into a callable object that, given a value, returns its truth
          (optionally mapped).
        * :meth:`.numerical_map`, which does the same as :meth:`.map`, but the callable object is
          internally numerical (which might sometimes be more efficient);
        * :meth:`.FuzzyNumber.crisp`, which finds the best real number equivalent for it (by defuzzification); and
        * ``float()``, which does the same using only default parameters.
        * :meth:`._expression_as_numerical`, which returns the fuzzy number (as a :class:`._Numerical` object) that
          is the result of a given an expression (of fuzzy numbers combined with logical and arithmetic operators),
          restricted to a given domain at a given resolution (minimum significant difference in value).

    Its subclasses implement:

        * :meth:`.t`, which returns the truth of a given value.
        * :meth:`._get_domain`, which returns the domain over which
          the continuous part of :math:`t(v)` is explicitly defined.
        * :meth:`._get_numerical`, which returns a :class:`._Numerical` object that is version of the fuzzy number
          with its continuous part, :math:`t(v)`, represented numerically for a given domain and precision
          (number of samples).

          """

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
        """Returns the "natural" domain of a fuzzy number---the extreme domain of the continuous part of the result
        of an expression, without any restriction to an allowed domain.

        For a :class:`.Literal` or :class:`._Numerical`, this is a straightforward report.  For an expression (that is,
        if the :class:`.FuzzyNumber` is an :class:`.Operator`, which may contain a whole tree of operators and
        literals), this is achieved by asking the operands for their natural domains and putting them through the
        operator's :meth:`.d_op` method to find how the operator transforms them.  In this way, calls propagate down
        the tree to all its member literals, and returned domains propagate up to the top.  Later, similar calls
        will be made downward using the inverses of the domain transforms, :meth:`.d_op_inv` methods, to tell the
        literals where to sample themselves.  The natural domain of an expression is needed to convert a desired
        resolution (minimum acceptable error in value) to precision (number of sample points).  Users almost certainly
        know the former but not the latter.

        Return:
            The domain of ``self`` that is defined as a continuous function.
        """

    @abstractmethod
    def _get_numerical(self, precision: int, allowed_domain: Domain = None) -> _Numerical:
        """Obtains and returns a numerical representation of ``self``.

        This is where the work is done in each subclass.  In a :class:`Literal` number, the work is to sample its
        analytical function appropriately.  In an :class:`Operator`, the work is to call for evaluations
        of its operands, operate on them, and return the numerical result.  This method is called by
        :meth:`._expression_as_numerical`, which calculates the precision and begins the cascade of recursive calls.

        Arguments:
            precision: The number of sample points used to cover the domain of the continuous function part of
                the result.  This determines how accurately the result represents the original.  It will have been
                calculated in :meth:`._expression_as_numerical` from the caller's desired ``resolution``, the minimum
                significant difference in value.  (In practice, two guard points will always be added outside the
                domain to ensure good interpolation near its edges.)
            allowed_domain: The extreme domain on which a result is acceptable.  Ultimately, recursive calls of this
                method will reach literals, which will only sample themselves on domains that will affect the
                result, so that no precision is wasted.

        Note:
            :class:`.Operator`\\ s hold their operands, which are fuzzy numbers---either :class:`.Literal`\\ s or more
            Operators---so calling this method on an expression built up in this way can result in many recursive
            calls.  By restricting the requested domain with ``allowed_domain`` at each step, calculations avoid
            considering inconsequential values, and concentrate all of their precision (sample points) only
            on stretches of domain that will have a bearing on the final result.

        Note:
            In cases where ``precision == 0``, the continuous part is excluded from calculations.

        Return:
            A numerical representation of ``self``, i.e., with a continuous function part, :math:`t(v)`,
            defined by NumPy arrays.
        """

    @abstractmethod
    def t(self, v: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Given a value, return its truth.

        Implementations should refer to, in order of priority:  the exceptional points, the continuous function,
        and the default truth "elsewhere".  The main case for this method's use is to check on how good
        the result of :meth:`.FuzzyNumber.crisp` is.  It also has some internal uses.

        Args:
            v: Any real number, a proposed value; a single one or an array of them.

        Returns:
            The suitability of the proposed values as a measure of truth in the range [0,1],
            the degree to which each is a member of the result."""

    # implementations: private helpers
    def _expression_as_numerical(self, resolution: float, allowed_domain: Tuple[float, float] = None) -> _Numerical:
        """A helper that gets the numerical representation of ``self``, restricted to ``allowed_domain``,
        at the required ``resolution``.

        This works even if ``self`` is a complex tree of :class:`.Operator`\\ s and :class:`.Literal`\\ s.
        It is used by :meth:`.FuzzyNumber.crisp`, :meth:`.numerical_map`, and :meth:`.display`.

        Its main task is to calculate the required precision from the given resolution and to make sure the domain
        of the result is properly restricted.  It does this by (assuming the most complex case, an expression
        represented as a tree of operators terminating in literals):

        1. Finding the natural domain of the result by the downward calls described in :meth:`._get_domain`.
        2. Using this and the given ``resolution`` to calculate a required precision.
        3. Using the calculated precision and the domain of the result (the intersection of the natural and allowed),
           to initiate a cascade of downward calls of :meth:`._get_numerical`, modifying the requested domain
           appropriately each time with the operator's :meth:`.d_op_inv` method.  Eventually the calls will reach
           terminal :class:`.Literal`\\ s that will sample themselves only over the domain that will affect the
           result.
        4. When the resulting :class:`._Numerical` has propagated to the top of the tree, any extraneous exceptional
           points (outside ``allowed_domain``) are removed, and it is returned.

        Args:
            resolution: The smallest difference in value the caller considers significant.  It must be > 0.
            allowed_domain: The acceptable domain for the result.  There are three reasons for restricting it:

                * If some mathematically valid solutions are physically nonsensical or inconvenient, or the user is
                  simply not interested in them.
                * If valid solutions extend across a vast domain that it would be impractical to sample, e.g., as
                  sometimes happens with division by small numbers.
                * If one is iteratively seeking a more precise solution and "zooming in" on an area of interest.

        Returns:
            A :class:`._Numerical` object that represents ``self`` according to the constraints of arguments."""
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
            resolution: This controls the accuracy of the result (a smaller resolution is better).  It should be
                at least as small as the maximum error you consider acceptable.  Internally, it is the sampling
                interval of the numerical representation.  Consider also that a coarse mesh in the numerical
                representation might miss narrow peaks in the continuous function.
                (Exceptional points, because they are defined explicitly, are unaffected by resolution.)
            allowed_domain: Bounds the domain of the result in case the answer must be limited.
                There are three reasons for restricting it:

                * If some mathematically valid solutions are physically nonsensical or inconvenient, or if the user is
                  simply not interested in them; e.g., if tempo must be on [30, 200] bpm, or temperature
                  must be on [-273.15, 100]°C.
                * If valid solutions extend across a vast domain that it would be impractical, undesirable
                  or uninformative to sample, e.g., as sometimes happens with division by small numbers.
                * If one is iteratively seeking a more precise solution and "zooming in" on an area of interest.

            crisper:  The :class:`.Crisper`  object that performs the defuzzification.
                If none is indicated, :attr:`.default_crisper` is used.  For a discussion of defuzzification
                algorithms, see :mod:`fuzzy.crisp`.

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
        """Returns the crisp float value, via :meth:`.FuzzyNumber.crisp`, using only default parameters.

        That is, an unrestricted domain with :attr:`.default_resolution`.  It could be convenient to set the default
        and use this if you have to do many similar calculations, but beware:  numerical methods without error
        bounds are meaningless, and failing to consider what resolution you require risks garbage output."""
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
                      interp: Interpolator = None) -> NumericalMap:
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
            (a smaller resolution is better).  Discrete points defined by ``self`` are unaffected by resolution.
        allowed_domain:  Restricts the defined domain of the result to no more than this parameter, i.e., discarding
            any exceptional points or continuous domain outside it---these will return the default "elsewhere" truth.
        range:  Translates the range of the internal function to the indicated range.  See :meth:`.Truth.scale`.
        map:  And does so via linear, logarithmic, or exponential mapping.  See :meth:`.Truth.scale`.
        interp:  An :class:`.crisp.Interpolator` object for interpolating between the sample points.
            If none is indicated, :attr:`.default_interpolator` is used.

    Returns:
        A callable object that can be used as a mathematical function.

    Example:
        | ``voltage = furnace_input_vs_thermostat_reading.map(range=(0,240)")``
        | ``y = voltage(temperature)``
        """
        numerical = self._expression_as_numerical(resolution, allowed_domain)
        return NumericalMap(numerical, range, map, interp)

    def display(self, resolution: float = None, domain: Domain = None, display_percentage: float = 25) -> None:
        """Displays the truth vs. value function that is ``self``.

        The parameter ``display_percentage`` is automatically related to the size of your monitor, as measured by
        this module when it loads.  Leaving the ``resolution`` undefined allows the method to set it to the equivalent
        of one pixel, for maximum efficiency.

        Args:
            resolution:  The minimum significant difference in value for the fuzzy number.
                Default: the equivalent of one pixel.
            domain:  The domain of values to display. Default: the union of the defined continuous
                domain with that of the most extreme discrete points.
            display_percentage:  The percentage of your monitor area to be taken up by the plot itself.
                Default: 25%.
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
            domain = Domain((domain[0] - .5, domain[0] + .5))
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
        # plt.title("Line graph")   # There's no way to get the name of the variable being plotted.
        plt.show()


class _Numerical(FuzzyNumber):  # user doesn't see it!  dataclass?
    """A fuzzy number represented numerically.

    These exist so that operators have something to operate upon.  This is where the real work is done.
    Literals make _Numerical versions of themselves by having an analytical function of t(v) and sampling it as needed.
    Operators construct their results by getting _Numerical versions of their operands and doing math on them. """

    def __init__(self, cd: Domain = None, cn: int = 0, cv: np.ndarray = None, ct: np.ndarray = None,
                 xv: np.ndarray = None, xt: np.ndarray = None, e: float = 0):
        """Args:
            cd: The domain over which the continuous numerical function is defined.  Valid if d[1]>=d[0].
            cn: The number of samples on  ``cd`` (not counting guard points). If cn<=0, no numerical function
                will be constructed. (And if cn=={1,2}, Literal will make it 3.)
            cv, ct: The continuous function's sample points: values, truths.  (Their len() must be equal.)
            xv, xt: The exceptional points: values, truths.   (Their len() must be equal.)
            e: The truth for all values not in ``cd`` or ``xv``---elsewhere.
            cv, xv: Within each:  all elements must be unique.
            ct, xt, e:  For each:  all elements must be on [0,1] except maybe for ct[0] and ct[-1]
                (guard points) which should be finite.

            I'd check all of these, but this is private, so I'm trusting myself (i.e., the Literals and Operators)."""
        self.cd = cd  # domain of continuous part
        if cn <= 0:
            cn, self.cd = 0, None
        self.cn = cn  # number of samples in  continuous part
        self.cv = cv  # continuous part: value samples
        self.ct = ct  # continuous part: truth samples
        self.xv = xv  # exceptional points: value samples
        self.xt = xt  # exceptional points: truth samples
        super().__init__(e)  # elsewhere
        self.clean()

    def clean(self):
        """Discard exceptional points that lie exactly on the continuous function of t(v)---because they would
        cause spurious results when entering into some arithmetic operations."""
        if self.xv is not None:
            i = np.where((self._sample(self.xv) == self.xt))
            self.xv, self.xt = np.delete(self.xv, i), np.delete(self.xt, i)
            if len(self.xv) == 0:
                self.xv, self.xt = None, None
        return self

    def __str__(self):
        if self.cd is None:
            c = "no continuous part; "
        else:
            c = str(f"continuous part: {self.cn} points on domain {self.cd}; ")
        if self.xv is None:
            x = "no exceptional points."
        else:
            x = str(f"exceptional points: \n {np.dstack((self.xv, self.xt))[0]}")
        return str(f"_Numerical: {c} {x}, elsewhere: {self.e:.3g}.")

    def _get_domain(self, allowed_domain: Domain = None) -> Union[Domain, None]:
        if allowed_domain is None:
            return self.cd
        else:
            return self.cd.intersection(allowed_domain)

    def _get_numerical(self, precision: int, allowed_domain: Domain = None) -> _Numerical:
        """When asked for a numerical copy of itself, a _Numerical returns itself, but there are some fine points.

        A :class:`._Numerical` will not resample itself, because that would lower its quality; therefore,
        it ignores the given precision, unless a precision of 0 insists on eliminating the continuous function.

        This method is designed to be called only on Literals and operands within Operators.  Why would it be called
        on a _Numerical (which is already, of course, numerical).   Someone might subclasses
        :class:`._Numerical` someday (but probably ought to subclass :class:`.Literal` instead).  In such a case,
        this method's only task is to impose the allowed_domain by possibly discarding some exceptional points.
        I don't foresee any situation where the parent _Numerical would be asked to return itself
        ---no harm if it does, though.  """
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
        """Returns the truth of given values, not considering exceptional points.  I.e., it's the truth of elsewhere
        and the continuous function.  See :meth:`._Numerical.t`."""
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
        """Returns the truth of given values.  See :meth:`._Numerical._sample`."""
        a = isinstance(v, np.ndarray)
        v = np.atleast_1d(v)
        c_e = self._sample(v, interp)
        if self.xv is not None:
            x = np.where(np.in1d(v, self.xv, assume_unique=True))  # where v is in xv
            y = np.where(np.in1d(self.xv, v, assume_unique=True))  # where xv is in v
            c_e[x] = self.xt[y]
        return c_e if a else c_e[0]
