"""Norms define the fundamental fuzzy operators (t-norms and co-norms) and perform their calculations.

Don't worry if you aren't familiar with these terms;  the default is probably all you will ever need.

The abstract class :class:`Norm` provides the method:

    * :meth:`.Norm.not_` (¬, *negation*),

and guarantees the provision of:

    * :meth:`.Norm.and_` (∧, *t-norm*),
    * :meth:`.Norm.or_` (∨, *co-norm*);  and their integral forms:
    * :meth:`.Norm._and_integral`
    * :meth:`.Norm._or_integral`.

For :meth:`.Norm.not_`, only the standard fuzzy negation, ¬s=1-s, is used.  It is the only choice
that is continuous, invertible (¬¬s=s), and compatible with crisp negation (¬0=1, ¬1=0).
Because of De Morgan's laws (¬(p∧q)=(¬p)∨(¬q); ¬(p∨q)=(¬p)∧(¬q)), the definition of any two of {¬, ∧, ∨}
implies the definition of the third.  Therefore, every t-norm unambiguously implies a dual co-norm,
so concrete subclasses of :class:`.Norm` should implement a :meth:`.Norm.and_` and :meth:`.Norm.or_`
that are properly related.

The integral forms will probably only ever be used privately, within the fuzzy arithmetic operators that require them.
I call them "integrals" by analogy with Riemann and product integrals, which respectively sum and multiply an
infinitude of infinitesimal numbers---these logical integrals do the same with an infinitude of fuzzy truths.
They should be directly analogous to their discrete counterparts, :meth:`.Norm.and_` and :meth:`.Norm.or_`.

The logic operators can operate on ``float`` or :class:`numpy.ndarray` objects with the usual presumption of range
on [0,1].  The purpose of the :class:`numpy.ndarray` functionality is to support operations in the :class:`Value`
class---it is not expected to be used publicly.  Because :meth:`.Norm.not_` is unary,
it takes only one argument at a time.  The other methods, :meth:`.Norm.and_` and :meth:`.Norm.or_`, are
associative, so they can take one or more arguments in any combination of types (but if more than one array
is present, they must all have the same length).  Subclasses of :class:`Norm`, however, need only define them
as binary operations.  In any case, if the arguments include an array, the result is an array, otherwise,
it is a ``float``.

:class:`Norm` objects are created by a factory class method, :meth:`.Norm.define`, which instantiates one of the
concrete subclasses of :class:`Norm`.  There are a variety of them to choose from:  :class:`Lax`
(the opposite of :class:`Drastic`), :class:`MinMax` (Gödel-Zadeh), (parameterized) :class:`Hamacher`,
:class:`Prod` (Goguen), :class:`Einstein`, :class:`Nilpotent`, :class:`Lukasiewicz` (Łukasiewicz),
and :class:`Drastic`.  It's even possible to create one that is a linear combination of two;
or one based entirely on how "strict" you want it to be---that is, on how likely it is to
produce extreme results (falser ands and truer ors).

Unless you want to switch between Norms often, however, it will be easiest to set the :mod:`fuzzy` module's
:attr:`global_norm` attribute to the one you want (the default is :class:`Prod`).  The logical and mathematical
operators of the :class:`.Truth` and :class:`.Value` classes refer to it for all their calculations.

Most users will need only :class:`Truth` and  :class:`Value` objects, used as if they were ``bool`` and ``float``
objects with some added functionality.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import exp, floor, ceil
from typing import Union, Callable

import numpy as np


# It seems easier to make these definitions class attributes rather than module attributes:
# _default_norm = Norm.define()

class Norm(ABC):
    """Norm objects define and provide the fundamental fuzzy logic operators (¬, ∧, ∨).

    They are created by the factory method :meth:`.Norm.define`, e.g., ``n = Norm.define()``.  Its functions are then
    the fundamental operators, e.g., ``n.and_(a, b)``.  Many norms are available.  When in doubt, use the default.
    """

    Operand = Union[float, np.ndarray]  # Operand = float | np.ndarray  # maybe write it this way after Python 3.8?
    Operator = Callable[[Operand, Operand], Operand]

    @classmethod
    def define(cls, **kwargs: str | float) -> Norm:
        """A factory method to create Norms: It parses ``kwargs`` and returns a subclass of Norm.

        Kwargs:
            ``key=value`` pairs in one of these combinations:

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

        +-------------+------------+-------------------------------------------------+
        |  strictness |    key     |name                                             |
        +=============+============+=================================================+
        |        -100 | ``'lx'``   |lax                                              |
        +-------------+------------+-------------------------------------------------+
        |      -75.69 | ``'mm'``   |minimum / maximum (Gödel, Zadeh)                 |
        +-------------+------------+-------------------------------------------------+
        |      -49.78 | ``'hh'``   |Hamacher                                         |
        +-------------+------------+-------------------------------------------------+
        |      -22.42 | ``'pp'``   |product / probabilistic sum (Goguen)             |
        +-------------+------------+-------------------------------------------------+
        |        5.00 | ``'ee'``   |Einstein                                         |
        +-------------+------------+-------------------------------------------------+
        |       33.20 | ``'nn'``   |nilpotent (Kleene-Dienes)                        |
        +-------------+------------+-------------------------------------------------+
        |       63.63 | ``'lb'``   |Łukasiewicz / bounded sum                        |
        +-------------+------------+-------------------------------------------------+
        |         100 | ``'dd'``   |drastic                                          |
        +-------------+------------+-------------------------------------------------+
        |[-50,-22,100]| ``'hhp'``  |parameterized Hamacher;  ``p=`` 0: Hamacher;     |
        |             |            |50: product; 100: drastic                        |
        +-------------+------------+-------------------------------------------------+


        Returns:
            A :class:`Norm` object which has methods for the three fundamental fuzzy logic
            operators:  :meth:`not_`, :meth:`and_`, :meth:`or_` (and for the two private
            operators:  :meth:`_and_integral` and :meth:`_or_integral`).

        Example:
            | ``n = Norm(norm="pp")``
            | ``n.and_(.5,.5,.5)``
            | returns .125.
        """
        # This can't be the __new__ method, since the instance returned is a subclass--infinte recursion would result.
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
    def _operate(cls, operator: Operator, *operands: Iterable[Operand]) -> Operand:
        """Parses the args of associative binary operators (which may be any combination of fits or Numpy arrays)
        and performs the operation on them together, returning an array (if there were any) or a fit.
        All arrays must be of the same length.

        Args:
            operator: any commutative binary operator defined by a subclass of :class:`Norm` (i.e., ∧ or ∨).
            operands: a list of :class:`Truth` or :class:`numpy.ndarray` of fits (which must all be of the same length).

        Returns:
            a single :class:`numpy.ndarray`, if any were input, or else a single fit."""
        item = list(operands)
        for i in range(1, len(item)):
            item[0] = operator(item[0], item[i])
        return item[0]

    @staticmethod
    def not_(s: Operand) -> Operand:
        """The standard fuzzy negation, :math:`r = 1 - s`.

        Args:
            s: the ``float`` or :class:`numpy.ndarray` to be negated.

        Returns:
            an object of the same type with all elements negated.
        """
        return 1 - s

    def and_(self, *args: Operand) -> Operand:
        """Fuzzy logic *and* of all ``args``.

            Args:
                args: one or more ``float`` or :class:`numpy.ndarray` objects valued on [0,1].

            Returns:
                The "intersection" or "conjunction" of all its inputs.
                """
        return self._operate(self._and, *args)

    def or_(self, *args: Operand) -> Operand:
        """Fuzzy logic *or* of all ``args``.

            Args:
                args: one or more ``float`` or :class:`numpy.ndarray` objects valued on [0,1].

            Returns:
                The "union" or "disjunction" of all its inputs.
                """
        return self._operate(self._or, *args)

    @abstractmethod
    def _and(self, a: Operand, b: Operand) -> Operand:
        """A private definition of fuzzy logic *binary* AND, to be implemented by :mod:`Numpy`."""

    @abstractmethod
    def _or(self, a: Operand, b: Operand) -> Operand:
        """A private definition of fuzzy logic *binary* OR, to be implemented by :mod:`Numpy`."""

    # I'll also define And- and Or-integrals (like and-ing or or-ing every point of a function together).
    # I only need the _or_integral for fuzzy arithmetic, but I'm defining the _and_integral for completeness.
    # many of these implementations will require:
    #     s = np.trapz(z) / line_length                       # the definite (Riemann) line integral
    #     p = exp(np.trapz(np.log(z))) / line_length          # the definite geometric (product) line integral
    # ---the definite (summation, product) integrals over some line on a function.
    # They must always be divided by their line_length so that they have the same metric (this is not true
    # for min/max operators, because extrema aren't diminished by rarity).  For the same reason,
    # the units of line_length should always be the sample interval on either edge of the Cartesian product.
    # (I'm doing this because later, for fuzzy arithmetic operators, I'll need to take fuzzy Or-integrals over lines
    # (describing an operator) on the t-norm (fuzzy *and*) of a Cartesian product of two fuzzy values (the operands).)

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
    """Norms created without arguments are descended from this practically abstract class.

    Why can't it be explicitly abstract???."""


# Here I'll implement the SimpleNorms from least to most strict (strong and, weak or to the reverse):
# I would provide more t-norms (Schweizer, Frank, Yager, Aczél–Alsina, Dombi, Sugeno–Weber, etc.),
# but I don't know how to interpret them continuously, in integral forms (which is necessary for the
# fuzzy arithmetic)---I only know how to do Riemann and geometric (product) integrals.  But that should be plenty!


class Lax(SimpleNorm):
    """Defines the lax (``lx``) norm pair.

       Lax is my own invention---the opposite extreme from :class:`Drastic` (``dd``),
       and fodder for :class:`CompoundNorm`, allowing the full range of strictness."""

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
    """Defines the Gödel-Zadeh (``mm``) (minimum / maximum) norm pair."""

    def _and(self, a: Operand, b: Operand) -> np.ndarray:
        return np.fmin(a, b)

    def _or(self, a: Operand, b: Operand) -> np.ndarray:
        return np.fmax(a, b)

    def _and_integral(self, z: np.ndarray, line_length: float) -> float:
        return np.amin(z)

    def _or_integral(self, z: np.ndarray, line_length: float) -> float:
        return np.amax(z)


class Hamacher(SimpleNorm):
    """Defines the Hamacher (``hh``) norm pair."""

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
        p = exp(np.trapz(np.log(z))) / line_length
        return 0 if s == p else (p / (s - p))

    def _or_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z) / line_length
        p = exp(np.trapz(np.log(z))) / line_length
        return s / (1 + p)


class Prod(SimpleNorm):
    """Defines the Goguen (``pp``) (product / probabilistic sum) norm pair."""

    def _and(self, a: Operand, b: Operand) -> np.ndarray:
        return a * b

    def _or(self, a: Operand, b: Operand) -> np.ndarray:
        return a + b - a * b

    def _and_integral(self, z: np.ndarray, line_length: float) -> float:
        p = exp(np.trapz(np.log(z)))  # definite geometric (product) integral
        return p / line_length

    def _or_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z)
        p = exp(np.trapz(np.log(z)))
        return (s - p) / line_length


class Einstein(SimpleNorm):
    """Defines the Einstein (``ee``) norm pair."""

    def _and(self, a: Operand, b: Operand) -> np.ndarray:
        return a * b / (a * b - a - b + 2)

    def _or(self, a: Operand, b: Operand) -> np.ndarray:
        return (a + b) / (1 + a * b)

    def _and_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z) / line_length
        p = exp(np.trapz(np.log(z))) / line_length
        return p / (p - s + 2)

    def _or_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z) / line_length
        p = exp(np.trapz(np.log(z))) / line_length
        return s / (1 + p)


class Nilpotent(SimpleNorm):
    """Defines the Kleene-Dienes (``nn``) (nilpotent) norm pair."""

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
    """Defines the Łukasiewicz / bounded sum (``lb``) norm pair."""

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
    """Defines the drastic (``dd``) norm pair."""

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
    """Norms created with arguments are descended from this practically abstract class.

    The __init__ method of its subclasses should take user parameter(s) on [0,100] or [-100,100]
    and map them onto whatever their t-norms/co-norms require.

    Why can't it be explicitly abstract???"""


class ParameterizedHamacher(ParameterizedNorm):
    """Defines the parameterized version of the Hamacher (`hhp`) norm pair.

    Arg:
        user_parameter: is expected to be on [0,100] and must be >=0 (it will be clipped if it is not).

        +---------------------+----------------------+
        | ``user_parameter=`` |    equivalent norm   |
        +=====================+======================+
        | 0                   |    Hamacher (``hh``) |
        +---------------------+----------------------+
        | 50                  |    product (``pp``)  |
        +---------------------+----------------------+
        | 100                 |    drastic (``dd``)  |
        +---------------------+----------------------+

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
        p = exp(np.trapz(np.log(z))) / line_length
        return 0 if self._p == s == 0 else p / (self._p + (1 - self._p) * (s - p))

    def _or_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z) / line_length
        p = exp(np.trapz(np.log(z))) / line_length
        return (s + (self._p - 2) * p) / (1 + (self._p - 1) * p)


class CompoundNorm(Norm):
    """Defines a linear combination of two other norms according to a weight on [0,100].

    Args:
        n1, n2: a :class:`SimpleNorm` or its name (a 2--3 letter code).
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
    """Defines a norm of given strictness, on a scale from [-100,100] (hard limit).

    Arg:
        strictness: the tendency to extreme values---**and** more false and **or** more true.

            The provided norm pairs are, in increasing strictness,
            proportional to volume of the unit cube under the co-norm curve (r = p∨q):

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
        of the unit cube under the co-norm curve."""
        strictness = max(min(strictness, 100), -100)
        name = ["lx", "mm", "hh", "pp", "ee", "nn", "lb", "dd"]
        x = [-100.0, -75.69, -49.78, -22.42, 5.00, 33.20, 63.63, 100.0]
        y = np.arange(0, 8)
        w = np.interp(strictness, x, y)
        n = CompoundNorm(name[floor(w)], name[ceil(w)], 100 * (w % 1))
        return n


# It would be nice to put this module variable at the top of the file:

global_norm = Norm.define(norm="pp")  # The default Prod t-norm is probably all you need.
"""The Norm used by operators as a default, and by all overloaded operators."""
