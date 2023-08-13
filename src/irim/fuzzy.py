"""
All the facilities for working with fuzzy logic and math.

Introduction
------------

There are three main families of classes here: :class:`Norm`, :class:`Truth`, and :class:`Value`:

Norm
    Norms define the fundamental fuzzy operators (t-norms and co-norms) and perform their calculations.
    Don't worry if you aren't familiar with these terms;  the default is probably all you will ever need.

    The abstract :class:`Norm` provides the operations:

        * :meth:`.Norm.not_` (¬, *negation*), and
        * :meth:`.Norm.clip`, which clips floats onto [0,1];

    and guarantees the provision of:

        * :meth:`.Norm.and_` (∧, *t-norm*),
        * :meth:`.Norm.or_` (∨, *co-norm*), and their integral forms:
        * :meth:`.Norm._and_integral`
        * :meth:`.Norm._or_integral`,

    though these will probably only ever be used privately, within the fuzzy arithmetic operators.
    The logic operators can operate on `float` or :class:`numpy.ndarray` [and maybe Numerical?];  and,
    in the case of the associative operators, on a `tuple` containing any number or combination of these.

    :class:`Norm` objects are created by a factory class method, :meth:`.Norm.define`, and there are a great variety
    of them to choose from:  :class:`Lax` (the opposite of :class:`Drastic`), :class:`MinMax` (Gödel-Zadeh),
    (parameterized) :class:`Hamacher`, :class:`Prod` (Goguen), :class:`Einstein`, :class:`Nilpotent`,
    :class:`Lukasiewicz` (Łukasiewicz), and :class:`Drastic`.  It's even possible to create one that is a linear
    combination of two; or one based entirely on how "strict" you want it to be---that is, on how likely it is to
    produce extreme results (falser ands and truer ors).

    Unless you want to switch between Norms often, however, it will be easiest to set the :mod:`fuzzy` module's
    :attr:`global_norm` attribute to the one you want.  The logical and mathematical operators of the :class:`Truth`
    and :class:`Value` classes refer to it for all their calculations.  The default is :class:`Prod`.

Truth
    A :class:`Truth` object is essentially a ``float`` restricted to [0,1].  It represents a *degree of truth*,
    from perfect falsehood (0), to absolute truth (1) or anything in-between.  It may be used to describe
    the truth of a proposition, the strength of an opinion, the certainty of a fact, the preference for a choice,
    or any other measured quantity of physical or mental reality that might vary between established limits.
    The class provides many methods and overloaded operators for working with logic:

        * The 3 basic logical operators:

            * :meth:`.Truth.and_` (∧, ``&``),
            * :meth:`.Truth.or_` (∨, ``|``), and
            * :meth:`.Truth.not_` (¬, ``~``).

        * The other 8 non-trivial logical connectives familiar from propositional calculus and electronic logic
          gates:

            * :meth:`.imp` (→, "implies", ``>>``),
            * :meth:`.con` (←, *converse*, ``<<``),
            * :meth:`.iff` (↔, *equivalence*, "if and only if", "xnor"), its inverse
            * :meth:`.xor` (⨁, *exclusive or*); and the other inverses:
            * :meth:`.nand` (↑, not conjunction),
            * :meth:`.nor` (↓, not disjunction),
            * :meth:`.nimp` (⇸, not implication), and
            * :meth:`.ncon` (⇷, not converse implication).

        * The 6 comparisons: ``<``, ``>`` , ``<=``, ``>=``, ``==``, ``!=``.
        * Methods:

            * :meth:`.isValid` to check validity (presence on the range [0,1]), and
            * :meth:`.clip` to ensure it by clipping.

        * Methods:  :meth:`.validate` and :meth:`.scale` to translate to and from a range of real numbers,
          linearly or logarithmically.
        * A :meth:`.crisp` "defuzzification" method for making final decisions, i.e., for converting from a
          *fuzzy* :class:`Truth` to a *crisp* ``bool``, by comparison to a threshold (given or by global default).
          (Consider, though, the utility of simply using a :class:`Truth` in its nuanced, ineffably beautiful,
          fuzzy form.)


Value
    The :class:`.Value` class applies the idea of fuzzy truth to the representation of numbers.  A :class:`Value`
    object is a function of truth vs. value.  We may think of the function as describing the suitability of each
    possible value for some purpose.  E.g., we might describe room temperature as a symmetrical triangular function
    on (68, 76)°F or (20, 24)°C.  In this way, we might state our opinions as "preference curves".  We might also
    represent empirical knowledge as fuzzy numbers.  E.g., sensory dissonance vs. interval is a highly structured
    function with many peaks and valleys across a wide domain, but all of this information can be encapsulated
    in a single :class:`Value` object.

and...


How to Use the Module
---------------------

There are two ways to use these:

* The hard way: create a Norm object and use its functions as logic and arithmetic operators.
* The easy way: set the global Norm and Defuzzifier  (if you aren't happy with the defaults)
  and use overloaded operators on :class:`Truth` and :class:`Value` objects.

The hard way
............

To use the operators you must first create a Norm object (see the factory method, :meth:`Norm.define`).
This defines the t-norm/co-norm pair that defines logic and arithmetic operations.

Example:
    ``n = Norm.define()``

This object has the fuzzy operators as methods.  For logic operators, you call it to operate on *fits*
(fuzzy units: floats on [0,1]), Numpy arrays of fits, or :class:`Value` objects (which can represent fuzzy numbers).

Example:
    | ``a, b = .2, .8``
    | ``print(n.not_(a), n.and_(a, b), n.or_(a, b)``
    | yields: .8, .16, .84.

For arithmetic operators, you call it to operate on :class:`Value` objects.


The easy way
............


How the Module Works
--------------------

very well?

overviews of the three families ...

"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union, Tuple, Callable, ClassVar

import numpy as np

# is the following the right way to define types for the type hints?:
# Truth = float # This causes documentation problems, I've changed it to the 'Truth' form here:
Operand = Union[float, np.ndarray, 'Truth']  #      maybe Numeric as well?
Operator = Callable[[Operand, Operand], Operand]
# Operand = float | np.ndarray  # maybe write it this way after Python 3.8?


# It seems easier to make these definitions class attributes rather than module attributes:
# _default_truth_threshold: Truth = .5
# _default_norm = Norm.define()
# _default_crisper = Crisper()


class Norm(ABC):
    """Norm objects define and provide the fuzzy logic operators.

    The most important Norm function is the factory :meth:`Norm.define`.
    You create a Norm object with it, e.g., ``n = Norm.define()`` and then its functions are fuzzy logic
    and arithmetic operators, e.g., ``n.and_(a, b)``.  Many norms are available.  When in doubt, use the default.

    The operators generally accept:

    * :class:`fuzzy.Unit` a *fit* (floats on [0,1])
    * :class:`np.ndarray` of fits
    * any combination of the above in a tuple (The ndarrays must all be of the same length.)
    * :class:`fuzzy.Number` (two or more for binary operators)

    The logic operators are:

    * Unary:
        * :meth:`clip` Ensures that the input is on [0,1].
        * :meth:`not_` Negation (flips suitability).

    * Binary:
        * :meth:`and_`
        * :meth:`or_`

    * Continuous (Used internally for arithmetic.  They operate over the domain of a fit-valued function.):
        * :meth:`and_integral`
        * :meth:`or_integral`

    The arithmetic operators are:

    * Unary:
        * :meth:`foc_` "Focus" narrows or widens the peaks of the preference curve.
        * :meth:`neg_`  Negative (flips sign of value).
        * :meth:`abs_`  Absolute value.
        * :meth:`inv_`  Inversion.

    * Binary:
        * :meth:`add_`  Addition.
        * :meth:`sub_`  Subtraction.
        * :meth:`mul_`  Multiplication.
        * :meth:`div_`  Division.

    """

    @classmethod
    def define(cls, **kwargs: str | float) -> Norm:
        """A factory method to create Norms: It parses kwargs and returns a subclass of Norm.

        Because of DeMorgan's law, the definition of AND (a t-norm) implies the definition of its dual,
        OR (a co-norm), i.e., they come in pairs.

        Kwargs:  (dict{str|float}) in one of these combinations:
            * ``None``:  The default:  product / probabilistic Sum (``'pp'``).
            * ``norm=`` a norm key (table below) indicating a simple norm.
            * ``norm=`` a norm key for a parameterized norm;
              ``p=`` a (list of) parameter(s)  on [0,100] or [-100,100].
            * ``norm1, norm2=`` keys for two simple norms forming a compound by linear combination;
              ``weight=`` [0,100], so that 0: ``norm1``, 100: ``norm2``.
            * ``strictness=`` [-100,100]: a norm defined by the tendency to have
              an extreme value (for greater strictness, :meth:`and_` is more likely to be false
              and :meth:`or_` is more likely to be true).

        The norm keys are, from least to most strict:

        +------------+------------+-------------------------------------------------+
        |  strictness|    key     |name                                             |
        +============+============+=================================================+
        |        -100| ``'lx'``   |lax                                              |
        +------------+------------+-------------------------------------------------+
        |         -75| ``'mm'``   |minimum / maximum                                |
        +------------+------------+-------------------------------------------------+
        |         -35| ``'hh'``   |Hamacher product / sum                           |
        +------------+------------+-------------------------------------------------+
        |          -5| ``'pp'``   |product / probabilistic sum                      |
        +------------+------------+-------------------------------------------------+
        |          10| ``'ee'``   |Einstein product / sum                           |
        +------------+------------+-------------------------------------------------+
        |          50| ``'nn'``   |nilpotent product / sum                          |
        +------------+------------+-------------------------------------------------+
        |          75| ``'lb'``   |Lukasiewicz / bounded sum                        |
        +------------+------------+-------------------------------------------------+
        |         100| ``'dd'``   |drastic product / sum                            |
        +------------+------------+-------------------------------------------------+
        |[-35,-5,100]| ``'hhp'``  |parameterized Hamacher;  ``p=`` 0: Hamacher;     |
        |            |            |50: product; 100: drastic                        |
        +------------+------------+-------------------------------------------------+

        Returns:
            A Norm object which has methods for fuzzy logic and math operators,
            :meth:`clip_`, :meth:`not_`, :meth:`and_`, :meth:`or_`;
            :meth:`add_`, :meth:`sub_`, :meth:`mul_`, :meth:`div_`,
            :meth:`foc_`, :meth:`abs_`, :meth:`neg_`, :meth:`inv_`.

        Example:
            | ``n = Norm(norm="pp")``
            | ``n.and_(.5,.5)``
            | returns .25.
        """
        if kwargs is None:
            n = Prod()
        elif "norm" in kwargs:
            n = cls._simple_factory(kwargs.get("norm"), kwargs.get("p"))
        elif "strictness" in kwargs:
            n = StrictnessNorm(kwargs.get("strictness"))
        elif "norm1" in kwargs and "norm2" in kwargs and "weight" in kwargs:
            n1 = cls._simple_factory(kwargs.get("n1"))
            n2 = cls._simple_factory(kwargs.get("n2"))
            w = kwargs.get("weight")
            n = CompoundNorm(n1, n2, w)
        else:
            n = Prod()
        return n

    @classmethod
    def _simple_factory(cls, norm_key: str, *args: float) -> SimpleNorm:
        """A factory for creating :class:`SimpleNorm` objects,
        used by :class:`CompoundNorm` and :class:`StrictnessNorm`.

        Args:
            norm_key: indicates which type to return (see :class:`Norm`).
            args: a tuple of parameters if a :class:`ParameterizedNorm` is indicated.

        Returns:
            A :class:`SimpleNorm` of the desired type.
            """
        if norm_key == "lx":
            return Lax()
        elif norm_key == "mm":
            return MinMax()
        elif norm_key == "hh":
            return Hamacher()
        elif norm_key == "pp":
            return Prod()
        elif norm_key == "ee":
            return Einstein()
        elif norm_key == "nn":
            return Nilpotent()
        elif norm_key == "lb":
            return Lukasiewicz()
        elif norm_key == "dd":
            return Drastic()
        elif norm_key == "hhp":
            return ParameterizedHamacher(float(args[0]))

    @classmethod
    def _operate(cls, operator: Operator, *operands: Tuple[Operand]) -> Operand:
        """Parses the args of commutative binary operators (which may be any combination of fits or numpy arrays)
        and performs the operation on them together, returning an array (if there were any) or a fit.
        The fits and arrays operate with each other separately, then the final fit and array together.
        All arrays must be of the same length.

        Args:
            operator: any commutative binary operator (a method defined by a subclass of :class:`Norm`).
            operands: a list of :class:`Truth` or :class:`np.ndarray` of fits (which must all be of the same length).

        Returns:
            a single :class:`np.ndarray`, if any were input, or else a single fit."""
        item = list(operands)
        for i in range(1, len(item)):
            item[0] = operator(item[0], item[i])
        r = item[0]
        return r    # Truth(r) if isinstance(r, float) else r

    @staticmethod
    def clip_(s: Operand) -> Operand:
        """Clips a number to the range [0,1], ensuring that it's a fuzzy unit.

        Args:
            s: the float or :class:`np.ndarray` (or maybe :class:`Numerical`) to be clipped

        Returns:
            an object of the same type with all elements clipped to [0,1]---fuzzy units.
        """
        # equivalent:  max(min(s, 1), 0)
        # I need something here for :class:`Numerical`, right?
        # Should actual clipping raise an exception or print a warning?
        r = np.clip(s, 0, 1)
        return r    # Truth(r) if isinstance(r, float) else r

    @staticmethod
    def not_(s: Operand) -> Operand:
        """The standard fuzzy negation, :math:`r = 1 - s`.

        Args:
            s: the float or :class:`np.ndarray` to be negated.

        Returns:
            an object of the same type with all elements negated.
        """
        # I don't anticipate using other negations:
        # this implies a one-to-one relation between t-norms and co-norms through DeMorgan's law.
        return 1 - s    # Truth(r) if isinstance(r, float) else r

    def and_(self, *args: Operand) -> Operand:
        """Fuzzy logic AND on any combination of floats and equal-length numpy.ndarrays valued on [0,1]."""
        return self._operate(self._and, *args)

    def or_(self, *args: Operand) -> Operand:
        """Fuzzy logic OR on any combination of floats and equal-length numpy.ndarrays valued on [0,1]."""
        return self._operate(self._or, *args)

    @abstractmethod
    def _and(self, a: Operand, b: Operand) -> Operand:
        """A private definition of fuzzy logic *binary* AND, array version."""

    @abstractmethod
    def _or(self, a: Operand, b: Operand) -> Operand:
        """A private definition of fuzzy logic *binary* OR, array version."""

    # I'll also define And- and Or-integrals (like and-ing or or-ing every point of a function together).
    # I only need the or_integral for fuzzy arithmetic, but I'm defining the and_integral for completeness.
    # many of these implementations will require:
    #     s = np.trapz(z) / line_length                       # definite line integral
    #     p = math.exp(np.trapz(np.log(z))) / line_length     # definite geometric (product) line integral
    # ---the definite (Riemann, geometric) integrals over some line on a function
    # (fuzzy _and_ (t-norm), in practice) of the Cartesian product of two fuzzy values.
    # They must always be divided by their line_length so that they have the same metric (this is not true
    # for min/max operators, because extrema aren't diminished by rarity).  For the same reason,
    # the units of line_length should always be the sample interval on either edge of the Cartesian product.

    @abstractmethod
    def _and_integral(self, z: np.ndarray, line_length: float) -> float:
        """A private definition of fuzzy logic AND-integral.

        Args:
            z: an array of suitabilites (on [0,1]) vs. uniformly-spaced values
            line_length: arclength of the line over which the definite integral is to be taken,
                in units of sample intervals of the Cartesian product."""

    @abstractmethod
    def _or_integral(self, z: np.ndarray, line_length: float) -> float:
        """A private definition of fuzzy logic OR-integral.

        Args:
            z: an array of suitabilites (on [0,1]) vs. uniformly-spaced values
            line_length: arclength of the line over which the definite integral is to be taken,
                in units of sample intervals of the Cartesian product."""


# noinspection PyAbstractClass
class SimpleNorm(Norm):
    """These are created without arguments."""


# Here I'll implement the SimpleNorms from least to most strict (strong and, weak or to the reverse):
#   I would provide more t-norms (Schweizer, Frank, Yager, Aczél–Alsina, Dombi, Sugeno–Weber, etc.),
#   but I don't know how to interpret them continuously (which is necessary for the fuzzy arithmetic)---
#   I only know how to do Riemann and geometric (product) integrals.  That should be plenty!


class Lax(SimpleNorm):
    """Defines the lax (``lx``) t-norm/co-norm pair
       (my own invention, the opposite extreme from :class:`Drastic` (``dd``), fodder for :class:`CompoundNorm`)."""

    def _and(self, a: Operand, b: Operand) -> np.ndarray:
        # equivalent:  if a == 0: r = b;  elif b == 0: r = a;  else: r = 1
        c = b * a.logical_not + a * b.logical_not
        r = c.logical_not - (a.logical_not * b.logical_not)
        return r + c

    def _or(self, a: Operand, b: Operand) -> np.ndarray:
        # equivalent:  if a == 1: r = b;  elif b == 1: r = a;  else: r = 0
        c = b * numpy.equal(a, 1)
        d = a * numpy.equal(b, 1)
        return np.clip(c + d, 0, 1)

    def _and_integral(self, z: np.ndarray, line_length: float) -> float:
        return np.amax(z) if np.amin(z) == 0 else 1

    def _or_integral(self, z: np.ndarray, line_length: float) -> float:
        return np.amin(z) if np.amax(z) == 1 else 0


class MinMax(SimpleNorm):
    """Defines the Gödel-Zadeh (minimum / maximum) (``mm``) t-norm/co-norm pair."""

    def _and(self, a: Operand, b: Operand) -> np.ndarray:
        return np.fmin(a, b)

    def _or(self, a: Operand, b: Operand) -> np.ndarray:
        return np.fmax(a, b)

    def _and_integral(self, z: np.ndarray, line_length: float) -> float:
        return np.amin(z)

    def _or_integral(self, z: np.ndarray, line_length: float) -> float:
        return np.amax(z)


class Hamacher(SimpleNorm):
    """Defines the Hamacher product / sum (``hh``) t-norm/co-norm pair."""

    def _and(self, a: Operand, b: Operand) -> np.ndarray:
        # equivalent:  return 0 if a == b == 0 else a * b / (a + b - a * b)  # Could get +inf near a==b==0?
        c = a * b / (a + b - a * b)
        return np.nan_to_num(c, nan=0.0, posinf=1.0, neginf=0.0)

    def _or(self, a: Operand, b: Operand) -> np.ndarray:
        # equivalent:  return 1 if a == b == 1 else (a + b - 2 * a * b) / (1 - a * b)
        c = (a + b - 2 * a * b) / (1 - a * b)
        return np.nan_to_num(c, nan=1.0, posinf=1.0, neginf=0.0)

    def _and_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z) / line_length
        p = math.exp(np.trapz(np.log(z))) / line_length
        return 0 if s == p else (p / (s - p))

    def _or_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z) / line_length
        p = math.exp(np.trapz(np.log(z))) / line_length
        return s / (1 + p)


class Prod(SimpleNorm):
    """Defines the Goguen (product / probabilistic sum) (``pp``) t-norm/co-norm pair---often called simply "Prod"."""

    def _and(self, a: Operand, b: Operand) -> np.ndarray:
        return a * b

    def _or(self, a: Operand, b: Operand) -> np.ndarray:
        return a + b - a * b

    def _and_integral(self, z: np.ndarray, line_length: float) -> float:
        p = math.exp(np.trapz(np.log(z)))  # definite geometric (product) integral
        return p / line_length

    def _or_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z)
        p = math.exp(np.trapz(np.log(z)))
        return (s - p) / line_length


class Einstein(SimpleNorm):
    """Defines the Einstein product / sum (``ee``) t-norm/co-norm pair."""

    def _and(self, a: Operand, b: Operand) -> np.ndarray:
        return a * b / (a * b - a - b + 2)

    def _or(self, a: Operand, b: Operand) -> np.ndarray:
        return (a + b) / (1 + a * b)

    def _and_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z) / line_length
        p = math.exp(np.trapz(np.log(z))) / line_length
        return p / (p - s + 2)

    def _or_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z) / line_length
        p = math.exp(np.trapz(np.log(z))) / line_length
        return s / (1 + p)


class Nilpotent(SimpleNorm):
    """Defines the Kleene-Dienes (nilpotent minimum / maximum) (``nn``) t-norm/co-norm pair."""

    def _and(self, a: Operand, b: Operand) -> np.ndarray:
        # equivalent:  min(a, b) if a + b > 1 else 0
        return np.fmin(a, b) * ((a + b) > 1)

    def _or(self, a: Operand, b: Operand) -> np.ndarray:
        # equivalent:  max(a, b) if a + b < 1 else 1
        mask = ((a + b) < 1)
        return np.fmax(a, b) * mask + np.logical_not(mask)

    def _and_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z) / line_length
        return np.amin(z) if s > 1 else 0

    def _or_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z) / line_length
        return np.amax(z) if s < 1 else 1


class Lukasiewicz(SimpleNorm):
    """Defines the Łukasiewicz / bounded sum (``lb``) t-norm/co-norm pair."""

    def _and(self, a: Operand, b: Operand) -> np.ndarray:
        # equivalent:  max(0.0, a + b - 1)
        c = a + b - 1
        return np.clip(c, 0, 1)

    def _or(self, a: Operand, b: Operand) -> np.ndarray:
        # equivalent:  min(a + b, 1)
        c = a + b
        return np.clip(c, 0, 1)

    def _and_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z) / line_length
        return max(0, s - 1)

    def _or_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z) / line_length
        return min(s, 1)


class Drastic(SimpleNorm):
    """Defines the drastic (``dd``) t-norm/co-norm pair."""

    def _and(self, a: Operand, b: Operand) -> np.ndarray:
        # equivalent:  if a == 1: r = b;  elif b == 1: r = a;  else: r = 0
        c = b * numpy.equal(a, 1)
        d = a * numpy.equal(b, 1)
        return np.clip(c + d, 0, 1)

    def _or(self, a: Operand, b: Operand) -> np.ndarray:
        # equivalent:  if a == 0: r = b;  elif b == 0: r = a;  else: r = 1
        c = b * a.logical_not + a * b.logical_not
        return c + c.logical_not

    def _and_integral(self, z: np.ndarray, line_length: float) -> float:
        return np.amin(z) if np.amax(z) == 1 else 0

    def _or_integral(self, z: np.ndarray, line_length: float) -> float:
        return np.amax(z) if np.amin(z) == 0 else 1


# noinspection PyAbstractClass
class ParameterizedNorm(SimpleNorm):
    """The __init__ method of its subclasses should take user parameter(s) on [0,100] or [-100,100]
    and map them onto whatever their t-norms/co-norms require."""


class ParameterizedHamacher(ParameterizedNorm):
    """Defines the parameterized version of the Hamacher product / sum (`hhp`) t-norm/co-norm pair.

    The user parameter, ``p``, is expected to be on [0,100] and must be >=0 (it will be clipped if it is not).

        +--------+----------------------+
        | ``p=`` |    equivalent norm   |
        +========+======================+
        | 0      |    Hamacher (``he``) |
        +--------+----------------------+
        | 50     |    product (``pp``)  |
        +--------+----------------------+
        | 100    |    drastic (``dd``)  |
        +--------+----------------------+

    """

    def __init__(self, user_parameter=0.0):
        """Maps the user parameter on [0,100] to the behavior described in :class:`ParameterizedHamacher`:
        50 = Prod, 0 and 100 are very close to Hamacher and drastic."""
        user_parameter = max(user_parameter, 0)  # (user_parameter, self._p) =  (0, .001), (50, 1), (100, 1000)
        self._p = 0 if user_parameter < 0 else 10 ** (.06 * user_parameter - 3)

    def _and(self, a: Operand, b: Operand) -> np.ndarray:
        # equivalent:   0 if self._p == a == b == 0 else a * b / (self._p + (1 - self._p) * (a + b - a * b))
        c = a * b / (self._p + (1 - self._p) * (a + b - a * b))
        return np.nan_to_num(c, nan=0.0, posinf=1.0, neginf=0.0)

    def _or(self, a: Operand, b: Operand) -> np.ndarray:
        c = (a + b + (self._p - 2) * a * b) / (1 + (self._p - 1) * a * b)
        return np.nan_to_num(c, nan=1.0, posinf=1.0, neginf=0.0)

    def _and_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z) / line_length
        p = math.exp(np.trapz(np.log(z))) / line_length
        return 0 if self._p == s == 0 else p / (self._p + (1 - self._p) * (s - p))

    def _or_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z) / line_length
        p = math.exp(np.trapz(np.log(z))) / line_length
        return (s + (self._p - 2) * p) / (1 + (self._p - 1) * p)


class CompoundNorm(Norm):
    """Instances return a linear combination of the results from two other norms according to a weight on [0,100].

    Args:
        n1, n2: two :class:`SimpleNorms` or their names (a 2--3 letter code).
        w: a parameter on [0,100].  If ``n2`` is stricter than ``n1``,
           then ``w`` may be thought of as a strictness parameter."""

    def __init__(self, n1: SimpleNorm | str, n2: SimpleNorm | str, w: float):
        self._n1 = Norm._simple_factory(n1) if isinstance(n1, str) else n1
        self._n2 = Norm._simple_factory(n2) if isinstance(n2, str) else n2
        self._w = w

    def _combination(self, r1: Operand, r2: Operand) -> Operand:
        return ((100 - self._w) * r1 + self._w * r2) / 100  # (w, result) = (0, n1), (50, avg(n1,n2)), (100, n2)

    def _and(self, a: Operand, b: Operand) -> np.ndarray:
        return self._combination(self._n1.and_(a, b), self._n2.and_(a, b))

    def _or(self, a: Operand, b: Operand) -> np.ndarray:
        return self._combination(self._n1.or_(a, b), self._n2.or_(a, b))

    def _and_integral(self, z: np.ndarray, line_length: float) -> float:
        return self._combination(self._n1._and_integral(z, line_length), self._n2._and_integral(z, line_length))

    def _or_integral(self, z: np.ndarray, line_length: float) -> float:
        return self._combination(self._n1._or_integral(z, line_length), self._n2._or_integral(z, line_length))


# noinspection PyAbstractClass
class StrictnessNorm(Norm):
    """Provides a norm of a given strictness, on a scale from [-100,100] (hard limit).
    By "strictness", I mean the tendency to extreme values---**and** more false and **or** more true.

    The provided norm pairs are, in increasing strictness,
    proportional to volume of the unit cube under the curve:

    +------------+------------+---------------------------------+
    |  strictness|    key     |name                             |
    +============+============+=================================+
    |     -100.00| ``'lx'``   |lax                              |
    +------------+------------+---------------------------------+
    |      -75.69| ``'mm'``   |minimum / maximum (Gödel, Zadeh) |
    +------------+------------+---------------------------------+
    |      -49.78| ``'hh'``   |Hamacher                         |
    +------------+------------+---------------------------------+
    |      -22.42| ``'pp'``   |product (Goguen)                 |
    +------------+------------+---------------------------------+
    |        5.00| ``'ee'``   |Einstein                         |
    +------------+------------+---------------------------------+
    |       33.20| ``'nn'``   |nilpotent (Kleene-Dienes)        |
    +------------+------------+---------------------------------+
    |       63.63| ``'lb'``   |Łukasiewicz                      |
    +------------+------------+---------------------------------+
    |      100.00| ``'dd'``   |drastic                          |
    +------------+------------+---------------------------------+

    """

    def __new__(cls, strictness: float = 0):
        """Creates a :class:`CompoundNorm` to achieve the desired strictness as described in :class:`StrictnessNorm`,
        by linear interpolation between the "landmark" :class`SimpleNorm`s.  The mapping is proportional to volume
        of the unit cube under the t-norm curve."""
        strictness = max(min(strictness, 100), -100)
        name = ["lx", "mm", "hh", "pp", "ee", "nn", "lb", "dd"]
        x = [-100.0, -75.69, -49.78, -22.42, 5.00, 33.20, 63.63, 100.0]
        y = np.arange(0, 8)
        w = np.interp(strictness, x, y)
        n = CompoundNorm(name[math.floor(w)], name[math.ceil(w)], 100 * (w % 1))
        return n


# It would be nice to put this module variable at the top of the file:

global_norm = Norm.define(norm="pp")  # The default Prod t-norm is probably all you need.
"""The Norm used by operators as a default, and by all overloaded operators."""


# Here is the class Truth, so that we can have overloaded logic operators



@dataclass
class Truth(float):
    """A "fuzzy unit" or "fit".

    As binary units are *bits* with a domain of ``bool`` ({0,1}), fuzzy units are *fits* with a domain of [0,1]---which,
    by analogy, we might call "fool".  Its numerical value measures a degree of truth on the continuum between perfect
    falsehood (0) and perfect truth (1).  This might be taken as the suitability of a choice, the appropriateness
    of a value, the likelihood of a circumstance, or the strength of an opinion.  Such measures can be used to turn
    judgements into calculations.

    The operators and methods are:

    * The basic ``and_`` (``&``), ``or_`` (``|``), and ``not_`` (``~``), as defined by a :class:`Norm`.
      (The underscores are necessary difference Python keywords.)
    * The other 8 significant binary truth tables built from them, named as in logic gates or propositional logic:

        * ``imp`` (``>>``), ``con`` (``<<``), ``iff``, ``xor``
        * and the negations: ``nand``, ``nor``, ``nimp``, ``ncon``

    * Comparisons: ``<``, ``>``, ``<=``, ``>=``, ``==``, ``!=``.
    * Conversions:

        * to ``float``, via :meth:`float()`, which is straightforward, or
        * to ``bool``, via :meth:`bool()` or :meth:`crisp(threshold)`:  defuzzification by comparison to a threshold
          ---either given or defined as a default for the class.

    * Ensuring the value is on the range [0,1] with :meth:`clip`.

    Arithmetic operators don't make sense within the class, but consider:  the solution to your problem mightn't be
    a crisp version of the ``Truth`` result, but a variable mapped from such a nuanced result.

    Parameters:
            global_threshold (float): Probably on [0,1].  It is used to convert a fuzzy ``Truth`` to
            a crisp ``bool``:  A :class:`fuzzy.Truth` object will :meth:`crisp()` (defuzzify) to ``True``
            at or above this number. It is also the default value for uninitialized ``Truth`` objects.
    """

    global_threshold: ClassVar[float] = .5
    s: float = field(default=global_threshold)


    def __str__(self):
        """Just the truth value."""
        return str(self.s)

    # The methods that deal with getting and setting:

    def clip(self) -> Truth:
        """Restricts the Truth's value to the domain of fuzzy truth, [0,1], by clipping."""
        self.s = 0 if self.s < 0 else 1 if self.s > 1 else self.s  # A little faster than the min/max, sorted or numpy.
        return self

    def crisp(self, threshold: float = None) -> bool:
        """Decides the crisp value of a fuzzy truth on the basis of ``threshold``.

            Note:
                You might consider simply using the ``Truth`` in its fuzzy form, as a ``float``, in order to take
                advantage of its nuance, i.e., by mapping it to some practical unit.

            Args:
                threshold:  Presumably on [0,1].  If the ``Truth`` is as true as this, we can call it "true".

            Returns:
                The crisp version of a fuzzy truth, a defuzzification of its logical proposition,
                a final decision about its suitability."""
        if not threshold:
            threshold = Truth.global_threshold
        if self.s < threshold:
            return False
        else:
            return True

    def __bool__(self) -> bool:
        """Crisps the ``Truth`` on the basis of the class's global threshold."""
        return Truth.crisp(self)

    # Here I implement logical connectives from propositional calculus / electronic logic gates:---
    # References to truth tables assume (p,q) = [0,1] (unary) and [00,01,10,11] (binary):

    # Unary operators (ignoring insistence (11), denial (00), and proposition (01))
    # ---we only need "¬", negation (10), accessed via: "s.not_()" or "~s".

    def not_(self, norm = None) -> Truth:
        """The negation ("not", ¬) unary operator (10), accessed by  ``s.not_()`` or ``~s``.

        Returns:
            The opposite of itself."""
        if not norm:
            norm = global_norm
        return norm.not_(self)  # Probably always the standard fuzzy not: 1 - self.s

    # binary operators
    # ignoring:  tautology (1111), contradiction (0000);
    # and the propositions and negations of single arguments:  (0101), (0011), (1010), (1100).  {0,3,5,10,12,15}
    # That leaves:
    # ∧ conjunction                           and_     &   0001    p&q             1   __and__(self, other)
    # ∨ disjunction                           or_      |   0111    p|q             7   __or__(self, other)
    # → material implication                  imp      >>  1101    ~p|q           13   __rshift__(self, other)
    # ← converse implication                  con      <<  1011    p|~q           11   __lshift__(self, other)
    # ↔ equivalence/biconditional (xnor)      iff          1001    (p&q)|(~p&~q)   9   __add__(self, other)
    # ⨁ nonequivalence/exclusive disjunction  xor          0110    (p|q)&~(p&q)    6   __sub__(self, other)
    # ↑ alternative denial                    nand         1110    ~(p&q)         14
    # ↓ joint denial                          nor          1000    ~(p|q)          8
    # ↛ material nonimplication               nimp         0010    p&~q            2
    # ↚ converse nonimplication               ncon         0100    ~p&q            4
    # I need a basic function for each.  Since "not, and, or" must be differenced with an underscore, all have one.
    # Operator precedence (inherited from Python):  ~  <<  >>  &  |
    # The traditional order is ~  &  |  >>, so beware and be free with parentheses.
    # I tried providing a |xxx| form for all the operators via https://pypi.org/project/infix/ .
    # It worked but produced spurious warnings, cautioned against its own use,
    # and looked inelegant on the page (operators had to be between "%%"), so I dropped it.
    # Unfortunately the 3 Boolean operators (not, and, or) cannot be overloaded---they're stuck in the crisp world.
    # In Python ^ is bitwise xor, but I'm not using it here because I'll need it for the sharpening (focus) operator.

    # First, the basic functions that provide the calculation via the `global_norm`:
    # and_, or_, imp_, con_, iff_, xor_;  nand_, nor_, nimp_, ncon_:

    def and_(self, other: Truth | float | int | bool, norm = None) -> Truth:
        """The conjunction ("and", ∧) binary operator (0001), accessed by  ``a.and_(b)`` or ``a & b``.

        Returns:
            The conjunction of ``self`` and ``other``, the extent to which they are both true.

        Warning:
            When calling ``a.and_(b)``, ``b`` may be ``Truth | float | int | bool``,
            but ``a`` must be a ``Truth`` object.
            Also, the boolean ``and`` operator is unaffected---it returns a ``bool``:
            the result of anding the crisped (defuzzified ``bool``) versions of the operands."""
        if not norm:
            norm = global_norm
        return norm.and_(self, other)

    def or_(self, other: Truth | float | int | bool, norm = None) -> Truth:
        """The disjunction ("inclusive or", ∨) binary operator (0111), accessed by ``a.or_(b)`` or ``a | b``.

        Returns:
            The disjunction of ``self`` and ``other``, the extent to which either is true.

        Warning:
            When calling ``a.or_(b)``, ``b`` may be ``Truth | float | int | bool``,
            but ``a`` must be a ``Truth`` object.
            Also, the boolean ``or`` operator is unaffected---it returns a ``bool``:
            the result of oring the crisped (defuzzified ``bool``) versions of the operands."""
        if not norm:
            norm = global_norm
        return norm.or_(self, other)

    def imp(self, other: Truth | float | int | bool, norm = None) -> Truth:
        """The material implication ("imply", →) binary operator (1101); the truth of such statements as
        "``self`` implies ``other``", or "if ``self`` then ``other``";  accessed by ``a.imp(b)`` or ``a >> b``.

        Returns:
            The implication of ``self`` to ``other``, the extent to which ``self`` must result in ``other``.

        Warning:
            When calling ``a.imp(b)``, ``b`` may be ``Truth | float | int | bool``,
            but ``a`` must be a ``Truth`` object."""
        if not norm:
            norm = global_norm
        return norm.or_(self.not_(norm), other)


    def con(self, other: Truth | float | int | bool, norm = None) -> Truth:
        """The converse implication ("con", ←) binary operator (1011); the truth of such statements as
        "``other`` implies ``self``", or "if ``other`` then ``self``";  accessed by ``a.con_(b)`` or ``a << b``.

        Returns:
            The implication of ``other`` to ``self`` (i.e., of ``self`` from ``other``),
            the extent to which ``other`` must indicate that ``self`` was true.

        Warning:
            When calling ``a.con(b)``, ``b`` may be ``Truth | float | int | bool``,
            but ``a`` must be a ``Truth`` object."""
        if not norm:
            norm = global_norm
        return self.or_(norm.not_(other), norm)

    def iff(self, other: Truth | float | int | bool, norm = None) -> Truth:
        """The equivalence or "biconditional" ("iff", ↔) binary operator (1001);
        familiar in Mathematics as "if and only if" and in Electronics as "xnor";
        the truth of such statements as, "``self`` and ``other`` imply each other", or
        "``self`` and ``other`` are true only to the same extent"; accessed by ``a.iff(b)``.

        Returns:
            The equivalence between ``self`` and ``other``, the extent to which they have the same degree of truth.

        Warning:
            When calling ``a.iff(b)``, ``b`` may be ``Truth | float | int | bool``,
            but ``a`` must be a ``Truth`` object."""
        if not norm:
            norm = global_norm
        return norm.or_(norm.and_(self, other),self.nor(other, norm))

    def xor(self, other: Truth | float | int | bool, norm = None) -> Truth:
        """The non-equivalence or "exclusive disjunction" ("exclusive or", ⨁) binary operator (0110);
        familiar in Electronics as "xor"; the truth of such statements as,
        "either ``self`` or ``other``, but not both together"; accessed by ``a.xor(b)``.

        Returns:
            The exclusive disjunction between ``self`` and ``other``, the extent to which they cannot occur together,
            the extent to which their degrees of truth differ.

        Warning:
            When calling ``a.xor(b)``, ``b`` may be ``Truth | float | int | bool``,
            but ``a`` must be a ``Truth`` object."""
        if not norm:
            norm = global_norm
        return norm.or_(self.nimp(other, norm), self.ncon(other, norm))

    def nand(self, other: Truth | float | int | bool, norm = None) -> Truth:
        """The alternative denial ("nand", ↑) binary operator (1110), the inverse of :meth:`and_`,
        accessed by ``a.nand(b)``.

        Returns:
            The inverse conjunction of `self` and ``other``, the extent to which they are not both true.

        Warning:
            When calling ``a.nand(b)``, ``b`` may be ``Truth | float | int | bool``,
            but ``a`` must be a ``Truth`` object."""
        if not norm:
            norm = global_norm
        return global_norm.not_(global_norm.and_(self, other))

    def nor(self, other: Truth | float | int | bool, norm = None) -> Truth:
        """The joint denial ("nor", ↓) binary operator (1000), the inverse of :meth:`or_`,
        accessed by ``a.nor(b)``.

        Returns:
            The inverse disjunction of ``self`` and ``other``, the extent to which both are false.

        Warning:
            When calling ``a.nor(b)``, ``b`` may be ``Truth | float | int | bool``,
            but ``a`` must be a ``Truth`` object."""
        if not norm:
            norm = global_norm
        return global_norm.not_(global_norm.or_(self, other))  # self.or_(other).not_()

    def nimp(self, other: Truth | float | int | bool, norm = None) -> Truth:
        """The material nonimplication ("nimply", ↛) binary operator (0010);  the truth of such statements as
        "``self`` does not imply ``other``", or "if ``self`` then not ``other``";  the inverse of :meth:`imp`;
        accessed by ``a.nimp(b)``.

        Returns:
            The nonimplication of ``self`` to ``other``; the extent to which ``self`` suppresses or inhibits ``other``;
            the extent to which the presence of ``self`` indicates that ``other`` will not occur.

        Warning:
            When calling ``a.nimp(b)``, ``b`` may be ``Truth | float | int | bool``,
            but ``a`` must be a ``Truth`` object."""
        if not norm:
            norm = global_norm
        return global_norm.and_(self, other.not_())

    def ncon(self, other: Truth | float | int | bool, norm = None) -> Truth:
        """The converse non-implication ("ncon", ↚) binary operator (0100); the truth of such statements as
        "``other`` does not imply ``self``", or "if ``other`` then not ``self``";  accessed by ``a.ncon(b)``.

        Returns:
            The non-implication of ``other`` to ``self``,
            the extent to which ``other`` indicates that ``self`` was false.

        Warning:
            When calling ``a.ncon(b)``, ``b`` may be ``Truth | float | int | bool``,
            but ``a`` must be a ``Truth`` object."""
        if not norm:
            norm = global_norm
        return global_norm.and_(global_norm.not_(self), other)

    # Operator symbol overrides:  these are for the most common operators that have a suitable overridable symbol:
    # `~  &  |  >>  <<` (not, and, or, implies, converse).

    def __invert__(self) -> Truth:
        """Overloads the unary ``~`` (bitwise not) operator.

        Returns:
            The opposite of itself.

        Warning:
            Applying ``~`` to anything but ``Truth`` objects will not produce a fuzzy result
            ---``float`` fails, ``int`` and ``bool`` do bitwise not, returning an ``int``.
            Also, the Boolean ``not`` operator is unaffected---it returns a ``bool``:
            the result of ``not fuzzy.crisp()``:  defuzzifying the ``Truth`` object
            according to ``Truth.global_threshold`` and inverting its ``bool`` result."""
        # There's no way to make this operate on ``float``, ``int``, or ``bool`` objects because there is no __rinvert__
        # method in which to put the Truth(float(self)).clip().not_(), because it's unary.
        return self.not_()

    def __and__(self, other: Truth) -> Truth:
        """Overloads the binary ``&`` (bitwise and) operator.

        Returns:
            The fuzzy-logical conjunction (∧) of the operands.

        Warning:
            One of the operands may be ``float | int | bool``, but at least one must be a ``Truth`` object,
            or the expected fuzzy result will not be produced.
            Also, the Boolean ``and`` operator is unaffected---it returns a ``bool``:
            the result of crisping, then anding the operands."""
        return Truth.and_(self, other)

    def __rand__(self, other: Truth) -> Truth:
        """Ensures that the overloading of ``&`` works as long as one operand is a ``Truth`` object."""
        return Truth.and_(Truth(float(other)).clip(), self)

    def __or__(self, other: Truth) -> Truth:
        """Overloads the binary ``|`` (bitwise or) operator.

        Returns:
            The fuzzy-logical disjunction (∨) of the operands.

        Warning:
            One of the operands may be ``float | int | bool``, but at least one must be a ``Truth`` object,
            or the expected fuzzy result will not be produced.
            Also, the Boolean ``or`` operator is unaffected---it returns a ``bool``:
            the result of crisping, then oring the operands."""
        return Truth.or_(self, other)

    def __ror__(self, other: Truth) -> Truth:
        """Ensures that the overloading of ``|`` works as long as one operand is a ``Truth`` object."""
        return Truth.or_(Truth(float(other)).clip(), self)

    def __rshift__(self, other: Truth) -> Truth:
        """Overloads the binary ``>>`` (bitwise right shift) operator.

        Returns:
            The fuzzy-logical implication (→) of the operands.

        Warning:
            One of the operands may be ``float | int | bool``, but at least one must be a ``Truth`` object,
            or the expected fuzzy result will not be produced."""
        return Truth.imp(self, other)

    def __rrshift__(self, other: Truth) -> Truth:
        """Ensures that the overloading of ``>>`` works as long as one operand is a ``Truth`` object."""
        return Truth.imp(Truth(float(other)).clip(), self)

    def __lshift__(self, other: Truth) -> Truth:
        """Overloads the binary ``<<`` (bitwise left shift) operator.

        Returns:
            The fuzzy-logical converse implication (←) of the operands.

        Warning:
            One of the operands may be ``float | int | bool``, but at least one must be a ``Truth`` object,
            or the expected fuzzy result will not be produced."""
        return Truth.con(self, other)

    def __rlshift__(self, other: Truth) -> Truth:
        """Ensures that the overloading of ``<<`` works as long as one operand is a ``Truth`` object."""
        return Truth.con(Truth(float(other)).clip(), self)


# Here are the classes for fuzzy value and arithmetic:
# Value --- base class;  Numerical --- "working" class of evaluation and defuzzification;
# Triangle, Trapezoid, Bell, Cauchy, Gauss, Points --- "atoms" defining a value;
# ValueNot, ValueAnd, ValueOr --- logic on values;
# Sum, Difference, Prod, Quotient, Focus, Abs, Inverse --- arithmetic on values.

# class Value(ABC):
#     """Represents a generally fuzzy real number (as a function of suitability (on [0,1]) vs. value).
#     It may be obtained (defuzzified) as a crisp value, along with that value's suitability.
#     """
#     _default_norm = Norm.define()
#     continuous_domain = None
#
#     @property
#     def continuous_domain(self):
#         return self._continuous_domain
#
#     @continuous_domain.setter
#     def continuous_domain(self, x):
#         self._continuous_domain = x
#
#     #
#     # def __init__(continuous_domain):
#     #     self.continuous_domain = continuous_domain
#
#     @abstractmethod
#     def evaluate(self, resolution: float):  # -> Numerical
#         """Obtains and returns a numerical representation of itself.
#         This is where the work is done in each subclass."""
#         return Numerical(self.continuous_domain, resolution)
#
#     def get(self, extreme_domain: (float, float), resolution: float, defuzzification_parameters) -> (float, float):
#         """Returns a crisp value that is equivalent to its (generally) fuzzy value,
#         along with that crisp value's suitability (a measure, on [0,1], of how good its answer is).
#
#         Returns: (v, s)
#             v: The crisp equivalent of this fuzzy number, a float.
#             s: The suitability (appropriateness, quality, truth, certainty) of v as an equivalent, a float on [0,1].
#
#         Arguments:
#             extreme_domain: bounds the result in case the answer must be limited,
#                 e.g., if tempo must be on [30, 200] bpm, or a parameter must be on [0,100].
#             resolution: the maximum distance between values that will be considered in the numerical representation.
#                 This controls the accuracy of the result (a smaller resolution is better).
#                 Also, consider that a coarse mesh in the numerical representation might miss narrow peaks.
#             defuzzification_parameters:  I don't know what they are yet.
#         """
#         # Evaluate, impose an extreme domain, defuzzify.  This is where it's implemented for every subclass.
#         numerical = self.evaluate(resolution)
#         # impose an extreme domain
#         numerical.impose_domian(extreme_domain)
#         # defuzzify.
#         v = s = 0  # dummy
#         return v, s
#
#
# class Numerical(Value):
#     """The numerical representation of a (generally) fuzzy value.
#
#     To represent the suitabilities of all real numbers, it uses:

#         * One Numpy array s(v) for one continuous domain [a, b].
#         * A set of exceptional points {(v_i, s_i)} that always override the array.
#         * An out-of-bounds suitability, s_0, that is assumed for any value otherwise undefined.
#
#         Is this where the defuzzify method is implemented?  Probably."""
#
#     def __init__(domain: (float, float), resolution: float):
#         """Initialization prepares the continuous domain with sample points that are
#         integer multiples of the resolution (so that all arrays in the calculation will line up),
#         covering the stated domain plus guard points on either end (for future interpolation).
#         So,  conveniently, subclasses perform a function on this array to sample themselves.
#
#         The set of exceptional points (the discrete domain) is an empty 2D array of value, suitability pairs.
#         Otherwise, undefined points in the domain default to a suitability of 0.
#
#         Args:
#             domain: values over which the continuous domain will be defined.
#             resolution: the separation between sample points (a smaller resolution is better).
#         """
#         self.continuous_domain = domain
#         v_0 = math.floor(domain[0] / resolution) - 1
#         v_n = math.ceil(domain[1] / resolution) + 1
#         number_of_samples = v_n - v_0 + 1
#         v_0, v_n = v_0 * resolution, v_n * resolution
#         # sample points on the continuous domain, to be filled with s(v) by subclasses:
#         self.continuous_v = np.linspace(v_0, v_n, number_of_samples)
#         self.continuous_s = np.linspace(v_0, v_n, number_of_samples)
#         # the discrete domain, to be filled as v,s by subclasses:
#         self.exceptional_points = np.empty((2, 0))
#         # the suitability elsewhere, outside the defined domains
#         self.out_of_bounds = 0
#
#     def suitability(self, value: float):
#         """Returns the suitability of a given value, as defined by this fuzzy value.
#
#         The exceptional points of the discrete domain override the definition of the continuous domain,
#         which is generally found by interpolation.  Points outside these domains return a default value."""
#         discrete = np.where(value == self.exceptional_points[0], self.exceptional_points[1])
#         if discrete is not None:
#             return discrete
#         else:
#             if value < self.continuous_domain[0] or value > self.continuous_domain[1]:
#                 return self.out_of_bounds
#             else:
#                 return np.interp(value, self.continuous_v, self.continuous_s)
#
#     def evaluate(self, resolution: float):  # -> Numerical
#         """It returns itself because it is the evaluation.
#
#         In any other subclass of Value, this is where the work would be done."""
#         return self
#
#     def impose_domian(self, imposed_domain: (float, float)):
#         """Discard any defined suitabilites <a | >b."""
#         self.exceptional_points = np.where(self.exceptional_points[0] > imposed_domain[0] and \
#                                            self.exceptional_points[0] < imposed_domain[1], self.exceptional_points)
#         self.continuous_s = np.where(self.continuous_v > imposed_domain[0] and \
#                                      self.continuous_v < imposed_domain[1], self.continuous_s)
#         self.continuous_v = np.where(self.continuous_v > imposed_domain[0] and \
#                                      self.continuous_v < imposed_domain[1], self.continuous_v)
#
#     def defuzzify(self) -> (float, float):
#         v = s = 0  # dummy  I don't know all the methods yet, but I prefer median of global maxima.
#         return v, s
#
#
# # Triangle, Trapezoid, Bell, Cauchy, Gauss, Points --- "atoms" defining a value;
# # ValueNot, ValueAnd, ValueOr --- logic on values;
# # Sum, Difference, Prod, Quotient, Focus, Abs, Inverse --- arithmetic on values.
#
# class Triangle(Value):
#     """Describes a fuzzy number as a trianglular function with a peak (maximum s) and extreme limits (s==0)"""
#
#     def __init__(peak, domain):
#         """Parameters:
#             peak:
#                 float: (most suitable) value , assumes s=1
#                 (float, float): (value, suitability)
#             domain:
#                 float:  HWHM
#                 (float, float): extreme domain where s>0"""
#
#         if isinstance(peak, (float, int)):  # assume its suitability = 1
#             self.peak = (peak, 1.0)
#         else:
#             self.peak = peak
#         if isinstance(domain, (float, int)):  # assume it is the HWFM about the peak
#             self.continuous_domain = (self.peak[0] - domain, self.peak[0] + domain)
#         else:
#             self.continuous_domain = domain  # shouldn't I check these things?
#
#     def evaluate(self, resolution: float):  # rethink this
#         n = Numerical(self.continuous_domain, resolution)
#         a_left = self.peak[1] / (self.peak[0] - self.continuous_domain[0])
#         a_right = self.peak[1] / (self.peak[0] - self.continuous_domain[1])
#         d = n.continuous_v  # - self.peak[0]
#
#         s = n.continuous_s
#         s = np.piecewise(d, [d < self.continuous_domain[0],
#                              (d > self.continuous_domain[0]) and (d < self.peak[0]),
#                              (d >= self.peak[0]) and (d < self.continuous_domain[1]),
#                              d > self.continuous_domain[1]],
#                          [lambda d: 0, lambda d: 1, lambda d: 2, lambda d: 3])
#         print(d)
#         # n.continuous_s =[lambda d: 0, lambda d: a_left*d, lambda d: 1-a_right*d, lambda d: 0]
#         print(s)
#         return n


# Here is where I am testing or playing around or something.


# #  This animates the norms (by strictness) for inspection.
#
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
#
# # Set the figure size
# plt.rcParams["figure.figsize"] = [5.00, 5.00]
# plt.rcParams["figure.autolayout"] = True
#
# # data for the contour plot
# data = np.empty((64, 64, 200))
#
#
# for t in range(0, 100):
#     n = Norm.define(strictness=t*2-100)
#     for x in range(0, 64):
#         for y in range(0, 64):
#             data[x, y, t ] = n.and_(x / 63, y / 63)
#
# # Create a figure and a set of subplots
# fig, ax = plt.subplots()
#
# # Method to change the contour data points
# def animate(i):
#     ax.clear()
#     ax.contourf(data[:, :, i], 64, cmap='plasma')
#
#
# ani = animation.FuncAnimation(fig, animate, 100, interval=10, blit=False)
#
# plt.show()

# #  This plots the norms for examination.
#
# import matplotlib.pyplot as plt
# nlist = ["lx","mm","hh","pp","ee","nn","lb","dd"]
# norm = []*8
# data = []*8
# ax = [plt.Figure()]*8
#
#
# fig, ((ax[0], ax[1], ax[2], ax[3]), (ax[4], ax[5], ax[6], ax[7])) = plt.subplots(nrows=2, ncols=4, sharey="all")
#
# for i in range(0,8):
#     ax[i].set_aspect('equal', adjustable='box')
#
# sa, sb = 10,75
#
# for i in range(0, 8):
#     st = sa + i*(sb-sa)/7
#     # norm.append(Norm.define(norm=nlist[i]))
#     n = Norm.define(strictness=st)
#     data.append(np.empty((32, 32)))
#     plt.axis('square')
#     for x in range(0, 32):
#         a = x / 31
#         for y in range(0, 32):
#             b = y / 31
#             data[i][x, y] = n.or_(a, b)
#     ax[i].contourf(data[i], 16, cmap='plasma')
#
# plt.show()

# x = Truth(.2)
# print(f"x={x}, 100*x={100*x}, x.not_()={x.not_()}, not x={not x}, ~x={~x}")
# y = Truth(.7)
# print(f"y={y}, 100*y={100*y}, y.not_()={y.not_()}, not y={not y}, ~y={~y}")
f = .499
# print(f"f={f}, 100*f={100*f}") # , f.not_()={f.not_()}, ~f={~f}
i = 0
# print(f"i={i}, 100*i={100*i}, ~i={~i}, not i={not i}")
b = True
# print(f"b={b}, 100*b={100*b}, ~b={~b}, not b={not b}")

c = Truth(.4)
d = Truth(.5)
e = Truth(.6)
# (.2).or_(.3)
# print(f"c and d={(c and d)}, c.and(d)={c.and_(d)}, d.and(e)={d.and_(e)}, c.and(e)={c.and_(e)}")
# print(f"c and d={d and c}, c&d={c & d}, d&e={d & e}, c&e={c & e}")
# print(f"")
#
# print(f"c.and(f)={c.and_(f)}, c.and(i)={c.and_(i)}, c.and(b)={c.and_(b)}")
# print(f"f.and(c)=err, i.and(c)=err, b.and(c)=err")
# print(f"c&f={c & f}, d&i={d & i}, c&b={c & b}")
# print(f"f&c={f & c}, i&d={i & d}, b&c={b & c}")
tr = Truth(1)
fa = Truth(0)
# TEST THE TRUTH TABLES
# print(f"not_=10: {fa.not_()}, {tr.not_()}")
# print(f"and_=0001: {fa.and_(fa)}, {fa.and_(tr)}, {tr.and_(fa)}, {tr.and_(tr)}")
# print(f"nand=1110: {fa.nand(fa)}, {fa.nand(tr)}, {tr.nand(fa)}, {tr.nand(tr)}")
# print(f"or_=0111: {fa.or_(fa)}, {fa.or_(tr)}, {tr.or_(fa)}, {tr.or_(tr)}")
# print(f"nor=1000: {fa.nor(fa)}, {fa.nor(tr)}, {tr.nor(fa)}, {tr.nor(tr)}")
# print(f"imp=1101: {fa.imp(fa)}, {fa.imp(tr)}, {tr.imp(fa)}, {tr.imp(tr)}")
# print(f"nimp=0010: {fa.nimp(fa)}, {fa.nimp(tr)}, {tr.nimp(fa)}, {tr.nimp(tr)}")
# print(f"con=1011: {fa.con(fa)}, {fa.con(tr)}, {tr.con(fa)}, {tr.con(tr)}")
# print(f"ncon=0100: {fa.ncon(fa)}, {fa.ncon(tr)}, {tr.ncon(fa)}, {tr.ncon(tr)}")
# print(f"iff=1001: {fa.iff(fa)}, {fa.iff(tr)}, {tr.iff(fa)}, {tr.iff(tr)}")
# print(f"xor=0110: {fa.xor(fa)}, {fa.xor(tr)}, {tr.xor(fa)}, {tr.xor(tr)}")

# TEST THE OVERLOADED OPERATORS ~ & | >> <<;  commutativity or not with float, int, bool:
# print(f"c={c}, d={d}, e={e}, f={f}, i={i}, b={b}")
# print(f"~c={~c}, ~d={~d}, ~e={~e}, ~f=ERR, ~i={~i}=UNX, ~b={~b}=UNX")
# print("")
# print(f"c&d={c&d}, c&f={c&f}, c&i={c&i}, c&b={c&b}")
# print(f"d&c={d&c}, f&c={f&c}, i&c={i&c}, b&c={b&c}")
# print(f"c|d={c|d}, c|f={c|f}, c|i={c|i}, c|b={c|b}")
# print(f"d|c={d|c}, f|c={f|c}, i|c={i|c}, b|c={b|c}")
# print(f"c>>d={c>>d}, c>>f={c>>f}, c>>i={c>>i}, c>>b={c>>b}")
# print(f"d>>c={d>>c}, f>>c={f>>c}, i>>c={i>>c}, b>>c={b>>c}")
# print(f"c<<d={c<<d}, c<<f={c<<f}, c<<i={c<<i}, c<<b={c<<b}")
# print(f"d<<c={d<<c}, f<<c={f<<c}, i<<c={i<<c}, b<<c={b<<c}")
# print("")   # truth tables:
# print(f"~=10: {~fa}, {~tr}")
# print(f"&=0001: {fa&(fa)}, {fa&(tr)}, {tr&(fa)}, {tr&(tr)}")
# print(f"|_=0111: {fa|(fa)}, {fa|(tr)}, {tr|(fa)}, {tr|(tr)}")
# print(f">>=1101: {fa>>(fa)}, {fa>>(tr)}, {tr>>(fa)}, {tr>>(tr)}")
# print(f"<<=1011: {fa<<(fa)}, {fa<<(tr)}, {tr<<(fa)}, {tr<<(tr)}")

# Truth.global_threshold = .8
