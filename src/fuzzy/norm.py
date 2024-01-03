"""defines the fundamental fuzzy operators.

The fundamental fuzzy operators are *t-norms* and *co-norms* (equivalents to logical *and* and *or*).
Don't worry if you aren't familiar with these terms.  You don't need to choose a particular norm---the default
is probably all you will ever need.  The :class:`Norm` classes perform all the logic calculations used by the other
classes.  Users do not need to use norm objects or their methods directly.


The abstract class :class:`Norm` provides the method:

    * :meth:`.Norm.not_` (¬, *negation*),

and guarantees the provision of:

    * :meth:`.Norm.and_` (∧, *t-norm*),
    * :meth:`.Norm.or_` (∨, *co-norm*);  and its integral form:
    * :meth:`.Norm.or_integral`.

For :meth:`.Norm.not_`, only the standard fuzzy negation, ¬s=1-s, is used.  It is the only choice
that is continuous, invertible (¬¬s=s), and compatible with crisp negation (¬0=1, ¬1=0).
Because of De Morgan's laws (¬(p∧q)=(¬p)∨(¬q); ¬(p∨q)=(¬p)∧(¬q)), the definition of any two of {¬, ∧, ∨}
implies the definition of the third.  Therefore, every t-norm unambiguously implies a dual co-norm,
so concrete subclasses of :class:`.Norm` should implement a :meth:`.Norm.and_` and :meth:`.Norm.or_`
that are properly related.

The integral form will probably only ever be used privately, within the fuzzy arithmetic operators that require it.
I call it an "integral" by analogy with Riemann and product integrals, which respectively sum and multiply an
infinitude of infinitesimal numbers---the logical or-integral does the same with an infinitude of fuzzy truths.
It should be directly analogous to its discrete counterpart, :meth:`.Norm.or_`.  (It would be easy, for completeness,
to create analogous "integrals" for the other logical connectives, but this is the only one I need at present.)

The logic operators can operate on ``float`` or :class:`numpy.ndarray` objects with the usual presumption of range
on [0,1].  The purpose of the :class:`numpy.ndarray` functionality is to support operations in the :mod:`fuzzy`
package---it is not expected to be used publicly.  Because :meth:`.Norm.not_` is unary,
it takes only one argument at a time.  The other methods, :meth:`.Norm.and_` and :meth:`.Norm.or_`, are
associative, so they can take one or more arguments in any combination of types (but if more than one array
is present, they must all have the same length).  Subclasses of :class:`Norm`, however, need only define them
as binary operations.  In any case, if the arguments include an array, the result is an array, otherwise,
it is a ``float``.  All output values will be on the range [0,1].

:class:`Norm` objects are created by a factory class method, :meth:`.Norm.define`, which instantiates one of the
concrete subclasses of :class:`Norm`.  There are a variety of them to choose from:  :class:`Lax`
(the opposite of :class:`Drastic`), :class:`MinMax` (Gödel-Zadeh), (parameterized) :class:`Hamacher`,
:class:`Prod` (Goguen), :class:`Einstein`, :class:`Nilpotent`, :class:`Lukasiewicz` (Łukasiewicz),
and :class:`Drastic`.  It's even possible to create one that is a linear combination of two;
or one based entirely on how "strict" you want it to be---that is, on how likely it is to
produce extreme results (falser ands and truer ors).

Unless you want to switch between Norms often, however, it will be easiest to set the :mod:`fuzzy` module's
:attr:`default_norm` attribute to the one you want (the default is :class:`Prod`).  The logical and mathematical
operators of the other classes refer to it as a default for their fuzzy calculations.

Users will need only :class:`Truth` and  :class:`Literal` objects, used as if they were ``bool`` and ``float``
objects, operated upon with the methods described in :mod:`truth` and :mod:`operator`.  The following documentation,
however, will explain some of the internal functioning and may be useful to those wishing to extend it, e.g., by
adding new norms.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from math import floor, ceil
from typing import Union, Callable

import numpy as np


# It might be nice to define the module attribute _default_norm here, but it seems to need to be at the bottom.

class Norm(ABC):
    """Norm objects define and provide the fundamental fuzzy logic operators (¬, ∧, ∨, ∨-integral).

    They are created by the factory method :meth:`.Norm.define`, e.g., ``n = Norm.define()``.  Its functions are then
    the fundamental operators, e.g., ``n.and_(a, b)``.  Many norms are available.  When in doubt, use the default.
    """

    Operand = Union[float, np.ndarray]  # TruthOperand = float | np.ndarray  # maybe write it this way after Python 3.8?
    Operator = Callable[[Operand, Operand], Operand]
    strictness: float  # Used to report the strictness of the norm.

    @classmethod
    def define(cls, **norm_args) -> Norm:
        """A factory method to create Norms: It parses ``norm_args`` and returns a subclass of Norm.

        Keyword Parameters:

            There are five possible keys:

                'n1'
                    The code for norm type.
                'n1p'
                    A list of parameters defining the norm, if the norm requires it.  (Only "hhp" and "str" do.)
                'n2', 'n2p', 'cnp'
                    To indicate a compound norm, describe the second with ``'n2', 'n2p'`` and set the "compound norm
                    percentage" with ``'cnp'``---this weights the two together so that, e.g., ``'cnp':0`` is all n1,
                    and ``'cnp':100`` is all n2.

            E.g., ``fuzzy_ctrl(norm={'n1': "str", 'n1p': 25})``.
            In order of increasing strictness, the norm codes are:

            +-------------+------------+-------------------------------------------------+
            | strictness  |   code     |name                                             |
            +=============+============+=================================================+
            |    -100.00  | ``'lx'``   |lax                                              |
            +-------------+------------+-------------------------------------------------+
            |     -75.69  | ``'mm'``   |minimum / maximum (Gödel, Zadeh)                 |
            +-------------+------------+-------------------------------------------------+
            |     -49.78  | ``'hh'``   |Hamacher                                         |
            +-------------+------------+-------------------------------------------------+
            |     -22.42  | ``'pp'``   |product (Goguen)                                 |
            +-------------+------------+-------------------------------------------------+
            |       5.00  | ``'ee'``   |Einstein                                         |
            +-------------+------------+-------------------------------------------------+
            |      33.20  | ``'nn'``   |nilpotent (Kleene-Dienes)                        |
            +-------------+------------+-------------------------------------------------+
            |      63.63  | ``'lb'``   |Łukasiewicz                                      |
            +-------------+------------+-------------------------------------------------+
            |     100.00  | ``'dd'``   |drastic                                          |
            +-------------+------------+-------------------------------------------------+
            |[-50,-22,100]| ``'hhp'``  |parameterized Hamacher;  ``p=`` 0: Hamacher;     |
            |             |            |50: product; 100: drastic                        |
            +-------------+------------+-------------------------------------------------+
            |[-100, 100]  | ``'str'``  |strictness norm                                  |
            +-------------+------------+-------------------------------------------------+


        Returns:
            A :class:`Norm` object which has methods for the fundamental fuzzy logic
            operators:  :meth:`not_`, :meth:`and_`, :meth:`or_`, and :meth:`or_integral`,
            defined by the optional ``norm_args`` keyword parameters..

        Example:
            | ``n = Norm(norm="pp")``
            | ``n.and_(.5,.5,.5)``
            | returns .125.
        """
        # This must be factory method rather than an implementation of __new__, since the instance returned
        # is a subclass---a call to __new__ would result in infinite recursion.
        if norm_args is not None:
            n1, n1p = norm_args.get('n1', "pp"), norm_args.get('n1p', [0])
            n2, n2p = norm_args.get('n2', None), norm_args.get('n2p', [0])
            cnp = norm_args.get('cnp', "50")
            norm_1 = cls._simple_factory(n1, *n1p)
            if n2 is not None:
                norm_2 = cls._simple_factory(n2, *n2p)
                norm = CompoundNorm(norm_1, norm_2, cnp)
            else:
                norm = norm_1
        else:
            norm = cls._simple_factory("pp")
        return norm

    @classmethod
    def _simple_factory(cls, norm_key: str, *args: float) -> Norm:
        """A factory for creating :class:`SimpleNorm` objects,
        used by :class:`CompoundNorm` and :class:`StrictnessNorm`.

        Args:
            norm_key: indicates which type to return (see :class:`Norm`).
            args: a tuple of parameters used if a :class:`ParameterizedNorm` is indicated.

        Returns:
            A :class:`SimpleNorm` of the desired type.
        """
        if norm_key == "lx":
            n = Lax()
            n.strictness = -100
        elif norm_key == "mm":
            n = MinMax()
            n.strictness = -75.69
        elif norm_key == "hh":
            n = Hamacher()
            n.strictness = -49.78
        elif norm_key == "ee":
            n = Einstein()
            n.strictness = 5.00
        elif norm_key == "nn":
            n = Nilpotent()
            n.strictness = 33.20
        elif norm_key == "lb":
            n = Lukasiewicz()
            n.strictness = 63.63
        elif norm_key == "dd":
            n = Drastic()
            n.strictness = 100
        elif norm_key == "hhp":
            n = ParameterizedHamacher(float(args[0]))
        elif norm_key == "str":
            n = StrictnessNorm(float(args[0]))
        else:  # norm_key == "pp":
            n = Prod()
            n.strictness = -22.42
        return n

    @classmethod
    def _operate(cls, operator: Operator, *operands: Iterable[Operand]) -> Operand:
        """Parses the args of associative binary operators (which may be any combination of fits or Numpy arrays)
        and performs the operation on them together, returning an array (if there were any) or a fit.
        All arrays must be of the same length.

        Args:
            operator: any commutative binary operator defined by a subclass of :class:`Norm` (i.e., ∧ or ∨).
            operands: There are two possible combinations (N.B.: a "fit" is a float on [0,1]):

                * An iterable of objects which may be any mixture of:

                    * :class:`Truth` or other fit-valued numbers,
                    * :class:`numpy.ndarray`\\ s of identical length, containing fits.

                * A single :class:`numpy.ndarray` containing fits.

        Returns:

            * In the first case: a single :class:`numpy.ndarray`, if any were input, or else a single fit---the
              result of all the iterable's elements operated together.
            * In the second case: a single fit---the result of all the array's elements operated together."""
        if isinstance(operands[0], Iterable) and len(operands) == 1:
            item = operands[0]
        else:
            item = list(operands)
        for i in range(1, len(item)):
            item[0] = operator(item[0], item[i])
        return item[0]

    @staticmethod
    def not_(s: Operand) -> Operand:
        """The standard fuzzy negation, :math:`r = 1 - s`.

        Args:
            s: the ``float`` or :class:`numpy.ndarray` (valued on [0,1]) to be negated.

        Returns:
            an object of the same type as ``s`` with all its elements negated.
        """
        return 1 - s

    def and_(self, *args: Operand) -> Operand:
        """Fuzzy logic *and* of all ``args``.

            Args:
                args: one or more ``float`` or :class:`numpy.ndarray` objects valued on [0,1].

            Returns:
                The "intersection" or "conjunction" of all ``args``'s elements.  If ``args`` is a single array, or an
                ``Iterable`` of ``float``\\s, the result is a ``float``.  If it contains one or more arrays (which
                must all be of the same length), the result is a similar array.
                """
        return self._operate(self._and, *args)

    def or_(self, *args: Operand) -> Operand:
        """Fuzzy logic *or* of all ``args``.

            Args:
                args: one or more ``float`` or :class:`numpy.ndarray` objects valued on [0,1].

            Returns:
                The "union" or "disjunction" of all ``args``'s elements.  If ``args`` is a single array, or an
                ``Iterable`` of ``float``\\s, the result is a ``float``.  If it contains one or more arrays (which
                must all be of the same length), the result is a similar array.
                """
        return self._operate(self._or, *args)

    @abstractmethod
    def _and(self, a: Operand, b: Operand) -> Operand:
        """A private definition of fuzzy logic *binary* AND, to be implemented by each subclass of :class:`Norm`."""

    @abstractmethod
    def _or(self, a: Operand, b: Operand) -> Operand:
        """A private definition of fuzzy logic *binary* OR, to be implemented by each subclass of :class:`Norm`."""

    # I used to think the following:
    # ...I'll define And- and Or-integrals (like and-ing or or-ing every point of a function together).
    # I only need the or_integral for fuzzy arithmetic, but I could define the _and_integral for completeness.
    # many of these implementations will require:
    #     s = np.trapz(z) / line_length                       # the definite (Riemann) line integral
    #     p = exp(np.trapz(np.log(z))) / line_length          # the definite geometric (product) line integral
    # i.e., the definite (summation, product) integrals over some line on a function.
    # They must always be divided by their line_length so that they have the same metric (this is not true
    # for min/max operators, because extrema aren't diminished by rarity).  For the same reason,
    # the units of line_length should always be the sample interval on either edge of the Cartesian product.
    # ...however, I found a better, simpler definition of logic integrals that doesn't require all that.

    def or_integral(self, z: np.ndarray) -> float:
        """A general fuzzy logic OR-integral that will work for any norm.

        I need something like an integral that, instead of summing an infinite number of infinitesimals,
        **ors** together an infinite number of truths of infinitesimal significance.  The result cannot be less true
        than the truest point on the integrand.  If **or** is defined as :math:`m=\\max(a, b)`, it seems clear that
        the result is simply :math:`m`, the maximum of the integrand.  For most norms, however, the truth increases
        any time two non-zero truths are **ored** together.  So, I expect the result of **oring** of an infinite number
        of non-zero norms to approach a number on :math:`[m,1]`.  E.g.: if the integrand is zero except for one
        exceptional point, :math:`m`, then :math:`m` is the result; but, if the entire integrand is a
        constant :math:`m`, the result will be on :math:`[m,1)`, and larger for stricter norms.  If :math:`a` is
        defined as the average truth of the integrand, then :math:`r = m \\lor a` exhibits all the behavior I expect
        from the or-integral without requiring much computation.  Rigorous Real Analysis might provide a more correct
        definition, but I do not expect it to be as elegant or efficient.

        Args:
            z: an array of truth values (``floats`` on [0,1]) presumed to be samples from a truth-valued function.
                For accuracy, the samples should be as uniformly-spaced as possible."""
        m = np.max(z)  # The maximum truth of the integrand.
        if m == 0:
            return 0
        else:
            a = np.sum(z) / len(z)  # The average truth of the integrand.
        return self.or_(m, a)


# Here I'll implement the simple Norms from least to most strict.
# I could provide more t-norms (Schweizer, Frank, Yager, Aczél–Alsina, Dombi, Sugeno–Weber, etc.),
# but this should be enough for now.


class Lax(Norm):
    """Defines the lax (``lx``) norm pair.

       Lax is my own invention---the opposite extreme from :class:`Drastic` (``dd``),
       and fodder for :class:`CompoundNorm`, allowing the full range of strictness to be exploited."""

    def __str__(self):
        return str(f"The lax norm (inverse of drastic)(strictness = {self.strictness}).")

    def _and(self, a: Operand, b: Operand) -> np.ndarray:
        # equivalent:  if a == 0: r = b;  elif b == 0: r = a;  else: r = 1
        c = b * np.logical_not(a) + a * np.logical_not(b)
        r = np.where((a != 0) & (b != 0), np.ones_like(a), c)
        return r

    def _or(self, a: Operand, b: Operand) -> np.ndarray:
        # equivalent:  if a == 1: r = b;  elif b == 1: r = a;  else: r = 0
        c = b * np.equal(a, 1)
        d = a * np.equal(b, 1)
        return np.clip(c + d, 0, 1)

    # The or-integral would be non-one only for horizontal or vertical lines along the far edges---(1, b) or (a, 1),
    # which none of the four operators do (?).  Maybe I should overload or_integral to always return 0.
    # Or maybe the bit of residue the general or_integral catches will matter.


class MinMax(Norm):
    """Defines the Gödel-Zadeh (``mm``) (minimum / maximum) norm pair."""

    def __str__(self):
        return str(f"The Gödel-Zadeh (min/max) norm (strictness = {self.strictness}).")

    def _and(self, a: Operand, b: Operand) -> np.ndarray:
        return np.fmin(a, b)

    def _or(self, a: Operand, b: Operand) -> np.ndarray:
        return np.fmax(a, b)

    def or_integral(self, z: np.ndarray) -> float:
        """Equivalent to the general definition provided by the parent, but a bit faster."""
        return np.amax(z)


class Hamacher(Norm):
    """Defines the Hamacher (``hh``) norm pair."""

    def __str__(self):
        return str(f"The Hamacher norm (strictness = {self.strictness}).")

    def _and(self, a: Operand, b: Operand) -> np.ndarray:
        # equivalent:  return 0 if a == b == 0 else a * b / (a + b - a * b)  # Could get +inf near a==b==0?
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.divide(a * b, a + b - a * b)
        return np.nan_to_num(c, nan=0.0, posinf=1.0, neginf=0.0)

    def _or(self, a: Operand, b: Operand) -> np.ndarray:
        # equivalent:  return 1 if a == b == 1 else (a + b - 2 * a * b) / (1 - a * b)
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.divide(a + b - 2 * a * b, 1 - a * b)
        return np.nan_to_num(c, nan=1.0, posinf=1.0, neginf=0.0)


class Prod(Norm):
    """Defines the Goguen (``pp``) (product / probabilistic sum) norm pair."""

    def __str__(self):
        return str(f"The Goguen (product / probabilistic sum) norm (strictness = {self.strictness}).")

    def _and(self, a: Operand, b: Operand) -> np.ndarray:
        return a * b

    def _or(self, a: Operand, b: Operand) -> np.ndarray:
        return a + b - a * b


class Einstein(Norm):
    """Defines the Einstein (``ee``) norm pair."""

    def __str__(self):
        return str(f"The Einstein norm (strictness = {self.strictness}).")

    def _and(self, a: Operand, b: Operand) -> np.ndarray:
        return a * b / (a * b - a - b + 2)

    def _or(self, a: Operand, b: Operand) -> np.ndarray:
        return (a + b) / (1 + a * b)


class Nilpotent(Norm):
    """Defines the Kleene-Dienes (``nn``) (nilpotent) norm pair."""

    def __str__(self):
        return str(f'The Kleene-Dienes ("nilpotent") norm (strictness = {self.strictness}).')

    def _and(self, a: Operand, b: Operand) -> np.ndarray:
        # equivalent:  min(a, b) if a + b > 1 else 0
        return np.fmin(a, b) * ((a + b) > 1)

    def _or(self, a: Operand, b: Operand) -> np.ndarray:
        # equivalent:  max(a, b) if a + b < 1 else 1
        mask = ((a + b) < 1)
        return np.fmax(a, b) * mask + np.logical_not(mask)

    # This or-integral may be different from the standard, but I'm not very sure.


class Lukasiewicz(Norm):
    """Defines the Łukasiewicz / bounded sum (``lb``) norm pair."""

    def __str__(self):
        return str(f"The Łukasiewicz / bounded sum norm (strictness = {self.strictness}).")

    def _and(self, a: Operand, b: Operand) -> np.ndarray:
        # equivalent:  max(0, a + b - 1)
        c = a + b - 1
        return np.clip(c, 0, 1)

    def _or(self, a: Operand, b: Operand) -> np.ndarray:
        # equivalent:  min(a + b, 1)
        c = a + b
        return np.clip(c, 0, 1)

    # This or-integral may be different from the standard, but I'm not very sure.


class Drastic(Norm):
    """Defines the drastic (``dd``) norm pair."""

    def __str__(self):
        return str(f"The drastic norm (inverse of lax)(strictness = {self.strictness}).")

    def _and(self, a: Operand, b: Operand) -> np.ndarray:
        # equivalent:  if a == 1: r = b;  elif b == 1: r = a;  else: r = 0
        c = b * np.equal(a, 1)
        d = a * np.equal(b, 1)
        return np.clip(c + d, 0, 1)

    def _or(self, a: Operand, b: Operand) -> np.ndarray:
        # equivalent:  if a == 0: r = b;  elif b == 0: r = a;  else: r = 1
        c = b * np.logical_not(a) + a * np.logical_not(b)
        r = np.where((a != 0) & (b != 0), np.ones_like(a), c)
        return r

    # The or-integral would be non-zero only for horizontal or vertical lines along the far edges---(1, b) or (a, 1),
    # which only division does for r=0 or inf (and maybe multiplication for r=0?).
    # Maybe I should overload or_integral to always return 1.
    # But I think maybe the bit of residue the parent method catches will matter.


# So far I have two parameterized types: Hamacher and a general strictness norm.

class ParameterizedHamacher(Norm):
    """Defines the parameterized version of the Hamacher (``hhp``) norm pair.

    Arg:
        p: is expected to be on [0,100] and must be >=0 (it will be clipped if it is not).

        +--------+----------------------+
        | ``p=`` |    equivalent norm   |
        +========+======================+
        | 0      |    Hamacher (``hh``) |
        +--------+----------------------+
        | 50     |    product (``pp``)  |
        +--------+----------------------+
        | 100    |    drastic (``dd``)  |
        +--------+----------------------+

    """

    def __str__(self):
        return str(f"A parameterized Hamacher norm, p = {self.up:.4g} (strictness ≈ {self.strictness:.4g}).")

    def __init__(self, p=0.0):
        """Maps the user parameter on [0,100] to the behavior described in :class:`ParameterizedHamacher`:
        50 = Prod, 0 and 100 are very close to Hamacher and drastic, respectively."""
        self.up = max(p, 0)  # (user_parameter, self._p) =  (0, .001), (50, 1), (100, 1000)
        self._p = 0 if p < 0 else 10 ** (.06 * p - 3)
        if p < 50:
            self.strictness = -49.78 + (p / 50) * 27.36
        else:
            self.strictness = -22.42 + ((p - 50) / 50) * 122.42

    def _and(self, a: Operand, b: Operand) -> np.ndarray:
        # equivalent:   0 if self._p == a == b == 0 else a * b / (self._p + (1 - self._p) * (a + b - a * b))
        c = a * b / (self._p + (1 - self._p) * (a + b - a * b))
        return np.nan_to_num(c, nan=0.0, posinf=1.0, neginf=0.0)

    def _or(self, a: Operand, b: Operand) -> np.ndarray:
        c = (a + b + (self._p - 2) * a * b) / (1 + (self._p - 1) * a * b)
        return np.nan_to_num(c, nan=1.0, posinf=1.0, neginf=0.0)


class CompoundNorm(Norm):
    """Defines norm pair as a linear combination of two others according to a weight on [0,100].

    Args:
        n1, n2: a :class:`Norm` or its name (a 2--3 letter code).
        w: a parameter on [0,100].  If ``n2`` is stricter than ``n1``,
           then ``w`` may be thought of as a strictness parameter."""

    def __init__(self, n1: Norm | str, n2: Norm | str, w: float):
        """In practice, the results of each norm are obtained separately, and combined as for a weighted average."""
        self._n1 = Norm._simple_factory(n1) if isinstance(n1, str) else n1
        self._n2 = Norm._simple_factory(n2) if isinstance(n2, str) else n2
        self._w = w
        self.strictness = ((100 - w) * self._n1.strictness + w * self._n2.strictness) / 100

    def __str__(self):
        return str(f"A compound norm (strictness = {self.strictness:.4g}): "
                   f"\n {100 - self._w}% {self._n1}, and\n {self._w}% {self._n2}")

    def _combination(self, r1: Operand, r2: Operand) -> Operand:
        return ((100 - self._w) * r1 + self._w * r2) / 100  # (w, result) = (0, n1), (50, avg(n1,n2)), (100, n2)

    def _and(self, a: Operand, b: Operand) -> np.ndarray:
        return self._combination(self._n1.and_(a, b), self._n2.and_(a, b))

    def _or(self, a: Operand, b: Operand) -> np.ndarray:
        return self._combination(self._n1.or_(a, b), self._n2.or_(a, b))

    def or_integral(self, z: np.ndarray) -> float:
        """In practice, the results of each norm are obtained separately, and combined as for a weighted average."""
        return self._combination(self._n1.or_integral(z), self._n2.or_integral(z))


# noinspection PyAbstractClass
class StrictnessNorm(Norm):
    """Defines a norm of given strictness, on a range from [-100,100] (hard limit).

    Really, it's a factory for creating :class:`.CompoundNorm`\\ s that achieve a given strictness.

    Arg:
        strictness: the tendency to extreme values---**and** more false and **or** more true.

            The provided norm pairs are, in increasing strictness,
            proportional to volume of the unit cube under the co-norm surface (r = p∨q):

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

            The norm is simply obtained by a linear combination of the two norms adjacent to the desired strictness.

    """

    def __new__(cls, strictness: float = 0):
        """Creates a :class:`CompoundNorm` to achieve the desired strictness as described in :class:`StrictnessNorm`,
        by linear interpolation between the "landmark" :class:`SimpleNorm`\\s.  The mapping is proportional to volume
        of the unit cube under the co-norm curve."""
        strictness = max(min(strictness, 100), -100)
        name = ["lx", "mm", "hh", "pp", "ee", "nn", "lb", "dd"]
        x = [-100.0, -75.69, -49.78, -22.42, 5.00, 33.20, 63.63, 100.0]
        y = np.arange(0, 8)
        w = np.interp(strictness, x, y)
        n = CompoundNorm(name[floor(w)], name[ceil(w)], 100 * (w % 1))
        return n

    def __str__(self):
        return str(f"A parameterized strictness norm (strictness = {self.strictness}).")


# It would be nice to put this module variable at the top of the file, but it seems to need to be here:

default_norm = Norm.define(norm="pp")  # The default Prod t-norm is probably all you need.
"""The Norm used by operators as a default, and by all overloaded operators.  Unless overridden 
by :meth:`fuzzy.operator.fuzzy_ctrl`, it is the product norm (``pp``)."""
