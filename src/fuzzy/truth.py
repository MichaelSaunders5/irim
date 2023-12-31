"""provides logical operators for *degrees of truth*.

A :class:`Truth` object is essentially an immutable ``float`` presumed to be on [0,1]---0 is false, 1 is true, and
the intermediate values indicate partial truth.  It may be used to describe the truth of a proposition, the strength
of an opinion, the certainty of a fact, the preference for a choice, or any other measured quantity of physical
or mental reality that might vary between established limits.

The class provides many methods and overloaded operators for working with logic:

* The methods for mapping between a :class:`Truth` and a range of real numbers, linearly or logarithmically:

    * :meth:`Truth` (the constructor), and
    * :meth:`.to_float`, which does the inverse.

* The methods:

    * :meth:`.is_valid` to check validity (presence on the range [0,1]), and
    * :meth:`.clip` to ensure it by clipping.

* Three basic logical connectives (the underscores are to difference them from Python keywords; the overloaded
  operators are rubricated as code; truth tables are shown in brackets):

    * :meth:`.Truth.not_` (¬, ``~``) [10],
    * :meth:`.Truth.and_` (∧, ``&``) [0001], and
    * :meth:`.Truth.or_` (∨, ``|``) [0111].

* The other eight non-trivial logical connectives familiar from propositional calculus and electronic logic
  gates:

    * :meth:`.Truth.imp` (→, "implies", ``>>``) [1101],
    * :meth:`.Truth.con` (←, *converse*, ``<<``) [1011],
    * :meth:`.Truth.iff` (↔, *equivalence*, "if and only if", "xnor", ``~(a @ b)``) [1001], and its inverse---
    * :meth:`.Truth.xor` (⨁, *exclusive or*, ``@``) [0110]; and the other inverses:
    * :meth:`.Truth.nand` (↑, non-conjunction, ``~(a & b)``) [1110],
    * :meth:`.Truth.nor` (↓, non-disjunction, ``~(a | b)``) [1000],
    * :meth:`.Truth.nimp` (:math:`\\nrightarrow`, non-implication, ``~(a >> b)``) [0010], and
    * :meth:`.Truth.ncon` (:math:`\\nleftarrow`, non-converse implication, ``~(a << b)``) [0100].

  All eleven logical connective methods can be defined by a :class:`.Norm` optionally indicated in the call;
  otherwise, they use the :attr:`fuzzy.norm.default_norm` by default.

* The six comparisons: ``<``, ``>`` , ``<=``, ``>=``, ``==``, ``!=``.
* A :meth:`.weight` method (``a // w``), for emphasizing or deëmphasizing the contribution of a truth to whatever
  expression it is in---equivalent to partially defuzzifying or refuzzifying it.
* Two methods for making final decisions---"defuzzifying" the *fuzzy* :class:`Truth` to a *crisp* ``bool``
  by comparing it to a threshold:

    * :meth:`.Truth.crisp`, which allows the threshold to be given in the call, or
    * ``bool()``, which refers to a global default:  :attr:`fuzzy.truth.default_threshold`.

  (Consider, though:  the best solution to your problem mightn't be crisp,
  but a variable mapped from the nuanced, ineffably beautiful, fuzzy truth.)

Note:
    As described above, six Python operators are overloaded for logic: ``~``, ``&``, ``|``, ``>>``, ``<<``, ``@``.
    So, for example, ``a & b`` is equivalent to ``Truth.and_(a, b)`` and ``a.and_(b)``.  The method forms can take an
    optional argument: keywords defining the norm.  This will override the default norm for
    the operator and any expressions contained in its operands, unless they contain their own norm arguments.

    For the operands of the above, we may use :class:`.Truth`, :class:`.FuzzyNumber`, and built-in Python numerical
    types (``float``, ``int``, ``bool``, or :class:`numpy.ndarray`\\s of them) interchangeably without worrying too much
    about it---a system of automatic promotions handles this intuitively.  Since an analogous set of logic operators
    exist for fuzzy numbers, the presence of a :class:`.FuzzyNumber` operand in an expression will promote the entire
    expression to one of fuzzy arithmetic.  Because the :class:`.FuzzyNumber` class has some methods with similar names,
    and because the numerical methods do not, there is some danger of confusion that the following points should
    clarify:

    * In calls of the form ``Truth.and_(a, b)``, a :class:`.Truth` method is indicated explicitly.
    * In calls of the form ``a.and_(b)``, ``a`` must be a :class:`.Truth` object for a :class:`.Truth` method to
      be indicated.  (If ``a`` is a :class:`.FuzzyNumber`, then a method from that class is indicated;  if ``a`` is
      a numerical type, an error will result, because those types have no such method.)
    * In expressions of the form ``a & b`` (i.e., with overloaded operators), at least one of the operands must be
      a :class:`.Truth` object for a :class:`.Truth` method to be indicated.  (But, if one is a :class:`.FuzzyNumber`,
      the method from that class will be used;  if *both* operands are numerical, the operator retains its meaning from
      standard Python.)
    * In the above, numerical operands on [0,1] are treated as :class:`.Truth` objects.
    * Numerical operands outside [0,1] become :class:`.Exactly` objects---fuzzy numbers only true at a single value.
      This, in turn, forces all the operators in the expression to be interpreted as :class:`.FuzzyNumber` methods,
      and the result will be a :class:`.FuzzyNumber` object.
    * The :meth:`.Truth.not_` method and its overloaded operator, ``~``, are unary.  The methods :meth:`.Truth.and_`
      and :meth:`.Truth.or_` are associative---so they may take any number of operands (zero or more).  The other
      logical methods, and all other overloaded operators are strictly binary.
    * Operands of :class:`numpy.ndarray` allow many logic operations to be carried out in a single call---by one
      truth on all elements or element-wise between two arrays of the same length.  This functionality was included
      mainly for the convenience of :class:`.FuzzyNumber` operators, but it may be useful more broadly.

    Additionally, the  :meth:`.weight` method overloads ``//``.  Its first operand must be a :class:`.Truth` object.
    Its second can be the usual weight parameter, a ``float`` on [-100,100], but if it is a :class:`.Truth` as well,
    it is translated onto that range (i.e., so that truth {0, .5, 1} becomes parameter {-100, 0 , 100}).

    You may have noticed that trivial truth tables are not implemented.
    They are:  contradiction [0000], tautology [1111];
    and the propositions and negations of single arguments:  [0011], [0101], [1100], [1010].


Caution:
    * In keeping with the weak-typing spirit of Python, the presumption of validity (range restricted to [0,1]) is not
      checked or enforced unless the user does so explicitly, with the methods :meth:`.is_valid`, :meth:`.clip`,
      or the constructor argument ``clip=True``.
    * Operands (the arguments of the logical connectives) may be of types
      :class:`Truth` | ``float`` | ``int`` | ``bool`` | :class:`numpy.ndarray`, with the usual assumption of validity.
    * Remember that, while the arguments of logical connectives may be of various types, they may only be called on
      :class:`Truth` objects, so, e.g., if ``a = Truth(.5)`` and ``b = .5``, then ``a.and_(b)`` works,
      but ``b.and_(a)`` fails.
    * The :meth:`Truth.and_` and :meth:`Truth.or_` methods may take zero or more arguments, because they are
      associative (although their overloaded operators, ``&`` and ``|``, are strictly binary).
      The other binary methods must take exactly two arguments or an error will result.
      Yes, :meth:`.Truth.iff` and :meth:`.Truth.xor` are mathematically associative, but, in these
      cases, "a * b * c" is too easily mistaken for the more desirable "(a * b) ∧ (b * c)" or "(a * b) * (b * c)",
      none of which are equivalent, so their programmatic associativity is disallowed to avoid confusion.
    * The precedence and associativity (in the programming sense) of the overloaded operators is
      inherited from Python, cannot be changed, and is slightly different from tradition, which is usually:
      ``~``, ``&``, ``|``, ``>>``, ``iff``.  Here, in :mod:`truth` it is: ``@``, ``~``, ``<<``, ``>>``, ``&``,
      ``^``, ``|``; followed by the six comparators.  Use parentheses if confusion might be a problem.
    * The boolean ``and``, ``or``, and ``not`` operators are unaffected---they return a ``bool``:
      the result of operating on the crisped (defuzzified by :func:`.bool`) versions of the operands.

"""

from __future__ import annotations

from dataclasses import dataclass, field
# from collections import Iterable
from math import log
from typing import Union, Tuple

import numpy as np

import fuzzy.norm

# from fuzzy.norm import Norm, default_norm

TruthOperand = Union[float, np.ndarray, int, bool, 'Truth']
"""A type indicating {:class:`Truth` | ``float`` | ``int`` | ``bool`` | :class:`numpy.ndarray`}.  
The range of values is presumed to be on [0,1]."""

default_threshold: float = .5
"""The default threshold for defuzzification, used by :meth:`.Truth.crisp` and ``bool()``.   

It is also the point of inflection for the :meth:`.weight` method, and the default value for ``Truth`` objects.  
It should be the number on [0,1] that you consider to be exactly "maybe".  It may be set directly or 
by :meth:`fuzzy.operator.fuzzy_ctrl` with the keyword ``threshold=``, e.g., ``fuzzy.truth.default_threshold = .8`` 
and ``fuzzy.operator.fuzzy_ctrl(threshold=.8)`` have the same effect.
I like the intuitive default of .5 because it gives equal scope and resolution to either side of "maybe"."""


def _prepare_norm(**kwargs) -> Norm:
    """A helper used in logical operator methods to get a norm defined by
    keyword parameters, or left undefined, the default norm.
    """
    kw = kwargs.get('norm', getattr(fuzzy.norm, 'default_norm'))
    if isinstance(kw, fuzzy.norm.Norm):
        return kw
    return fuzzy.norm.Norm.define(**kw)


def _promote_and_call(a, b, t_name: str, fn_name: str):
    """Appropriately promote the operand types and call the operator.

    This exists so that we may use :class:`.Truth`, :class:`.FuzzyNumber`, and built-in Python numerical types
    (``float``, ``int``, ``bool``) interchangeably in overloaded fuzzy expressions without worrying too much about it.
    An analogous function exists in :mod:`fuzzy.operator`, to deal with :class:`.FuzzyNumber` operations.

    Here, in :mod:`truth`, the operators are all logical and one operand must be :class:`.Truth`, or we would not have
    arrived at this function (e.g., an operation on two ``float`s would be handled by Python in the usual way).

    * If the other operand is also a :class`.Truth`, we simply call the class's method for the correct operation.
    * If it is a number on [0,1], we upgrade it to  a :class`.Truth`, and make the same call.
    * In all other cases, both operands must be upgraded to :class:`.FuzzyNumber`\\ s, and called as if it were
      written that way, i.e., referred to this method's analogue in :mod:`operator`.

      * If it (the non-Truth operand) is a number outside [0,1], it becomes an :class:`.Exactly` fuzzy number.
      * If it is a :class:`.FuzzyNumber`, it remains as it is.
      * In either case, the other operand (of type :class:`.Truth`), is passed on as-is, to be dealt with by
        :meth:`.operator._promote_and_call`.

    It is an added complication that we can't predict which of the two operands, if either, will require promotion,
    so this must be remembered.  Not all operands are commutative, so the order of a and b must be preserved.

    Args:
        a, b: operands which may be :class`.Truth`, :class`.FuzzyNumber`, ``float``, ``int``, or ``bool``.
        t_name: in a string, the name of the method for the :class`.Truth` operator, e.g., ``imp``.
        fn_name: in a string, the name of the overloaded magic method for the operator symbol.  It will be called in
            :mod:`operator` and end up in :meth:`.operator._promote_and_call`.

    Returns:
        A :class`.Truth`, or :class`.FuzzyNumber` that is the result of the indicated operation on ``a`` and ``b``.
        """
    a_truth, b_truth = isinstance(a, Truth), isinstance(b, Truth)
    if a_truth and b_truth:
        return eval("Truth." + t_name + "(a, b)")  # Call the operator for Truths
    a_num = (not a_truth) and isinstance(a, (float, int, bool))
    b_num = (not b_truth) and isinstance(b, (float, int, bool))
    if a_num or b_num:
        if a_num:
            a_prom, p = True, float(a)
        else:
            a_prom, p = False, float(b)
        if (p >= 0) and (p <= 1):
            p = Truth(p)
        else:
            from fuzzy.literal import Exactly
            p = Exactly(p)  # make it a FuzzyNumber: Exactly
        a = p if a_prom else a
        b = p if not a_prom else b
    if isinstance(a, Truth) and isinstance(b, Truth):
        return eval("Truth." + t_name + "(a, b)")
    else:
        return eval("Operator." + fn_name + "(a, b)")  # Call the operator for FuzzyNumbers


@dataclass
class Truth(float):
    """A fuzzy unit (a "fit"), representing a degree of truth.

    It is represented as an immutable float presumed to be on [0,1].
    """
    # default_threshold: ClassVar[float] = .5.  I moved TruthOperand, default_threshold up there^.  Trouble???
    s: float = field(default=default_threshold)

    # This works (maybe because it's a dataclass, but I've read that the following is normally necessary:
    # class Foo(float):
    #     def __new__(self, value, extra):
    #         return float.__new__(self, value)
    #     def __init__(self, value, ...):
    #         float.__init__(value)

    # The methods that deal with getting and setting:

    @classmethod
    def scale(cls, x: TruthOperand = None, dir: str = "in", r: Tuple[float, float] = None,
              map: str = "lin", clip: bool = False) -> TruthOperand:
        """A helper function to map between external variables and fuzzy units.

        In general, a complete workflow might resemble the following:

            #. You have a crisp variable, in some units, on a range of real numbers.  (E.g., oven temperatures on
               [70, 550]°F.)
            #. You map its range (expected or actual) to fuzzy units, [0,1].
            #. You do fuzzy reasoning and calculation with it.
            #. Finally, you translate the result from fuzzy units back into crisp units by a mapping
               that inverts the original one.  (E.g., a truth result of .71 becomes 411°F.)

        Of course, the first two steps might well be imaginary.

        Args:
            x: A number (or an array of them) to be mapped from one range to another.
            dir: The direction of the mapping, in to or out of fuzzy units.

                * ``"in"`` maps the given range, ``r``, onto the *fit* range, [0,1].
                * ``"out"`` maps the *fit* range, [0,1], onto the given range, ``r``.

            r: The expected input or desired output range, as a tuple, (min, max).  Its width, max-min,
                presents special concerns.  There are three modes:

                Output (``dir="out"``):
                    The range must be declared and have some non-zero width or an exception will be raised.

                Input: (``dir="in"``, ``r`` declared):
                    If the range has zero width, three results are possible:

                        * x<r:  -∞
                        * x=r:  :attr:`default_threshold`,
                        * x>r:  ∞

                    If ``clip`` is ``True`` this becomes {0, :attr:`default_threshold`, 1}.

                Automatic Input: : ``dir="in"``, ``r`` not declared (``= None``):
                    The range is set to the actual range found in ``x``, so the input is normalized to fill [0,1].
                    If ``x`` does not vary (e.g., if it is a single float), the result
                    is :attr:`default_threshold`.

                In any mode:
                    In normal operation, the width of ``r`` is non-zero and positive.  If it is negative (min > max),
                    then the sense of the input is flipped---higher numbers become lower.

            map: There are three mappings:

                * ``"exp"`` for exponential, ``"lin"`` for linear (the default), and ``"log"`` for logarithmic.
                * "exp" and "log" are inverses of each other, "lin" of itself;
                * ``"invexp"``, ``"invlin"``, and ``"invlog"`` are provided as mnemonics.

                I.e., if a fuzzy variable was created by ``dir="in"``, ``map="log"``, and you want to recover its
                original units, you would unmap it with ``dir="out"`` and ``map="exp"`` or ``map="invlog"``.

                Why use anything other than linear?  In nature, many variables have non-linear characteristics.
                By mapping them accordingly, you can preserve their significance and resolution across the range of the
                fuzzy unit.  E.g., audio amplitude can be thought of as linear, but the human ear perceives it
                logarithmically, hence the decibel.  If you mapped amplitude to a fuzzy unit linearly, the
                difference between .8 and .9 would appear equal in significance to that between .1 and .2.
                Perceptually, the latter is much more important than the former---most of the information is stuffed
                into the bottom of the fuzzy unit.  So, in this case, it would be better to use a logarithmic
                mapping---it would spread the information more evenly across [0,1].  (Exponential mapping is available
                for variables that would be more significant on the higher end.)  When the differences in value are
                in their proper proportion, fuzzy reasoning is more effective.

            clip:  if ``True``, clipping prevents out-of-range results.

        Returns:
            A number (or an array of them), mapped onto the indicated range.

        Caution:
            * Input mode with a zero-width expected range doesn't make a lot of sense, but it produces the limit of
              the usual behavior as the width approaches zero.  With clipping in use, this resembles trinary logic
              and might be good for something.
            * Automatic input mode can be risky:  if ``x`` is constant, the result, :attr:`.default_threshold`,
              may be unexpected, but it makes some sense:  whatever ``x`` is, it's assumed to *be* "dead center",
              instead of varying *about* dead center.
            * Output mode without a well-defined, non-zero output range is meaningless.
            * If you want to reverse a scaling, say by mapping in then out, remember that ``"log"`` and ``"exp"`` are
              inverse operations of each other and ``lin`` is the inverse of itself.  Or, to avoid confusion, use
              ``"log"`` with ``"invlog"`` and ``"exp"`` with ``"invexp"``.

        Note:
            * This method is used by :meth:`.Truth.__init__`, :meth:`.Truth.to_float`; :meth:`.Literal.CPoints`,
              :meth:`.DPoints.DPoints`; and :class:`.Map`.  It might be useful to you if you are dealing with raw data
              or creating your own subclasses of :class:`.Literal` to represent parameterized fuzzy numbers.

            """
        if x is None:
            x = default_threshold
        fl = (not isinstance(x, np.ndarray))  # To return the same type given.
        if dir == "in":  # map from an input variable on [min,max] to a fit representation on [0,1]:
            if r is None:
                r = (np.min(x), np.max(x))  # Map the whole *used* range to [0,1]
            if r[0] == r[1]:  # This is the limit of the behavior as the condition is approached, but why do it?:
                x = default_threshold if (x == r[0]) else inf if (x > r[0]) else -inf
            elif (map == "lin") or (map == "invlin"):
                x = (x - r[0]) / (r[1] - r[0])  # A
            elif (map == "log") or (map == "invexp"):
                x = np.log(1 + np.fabs(x - r[0])) / log(1 + abs(r[1] - r[0]))  # B
            else:  # "exp" or "invlog"
                x = 2 ** ((r[0] - x) / (r[0] - r[1])) - 1  # C this does not reconstruct the log onto [0,1]
        if clip:
            x = Truth.clip(x)
        if dir == "out":  # map from a fit representation on [0,1] to an output variable on [min,max]:
            if r is None:
                raise ValueError("The output range, r, must be defined.")
            if r[0] == r[1]:
                x = r[0]
            elif (map == "lin") or (map == "invlin"):
                x = r[0] + (r[1] - r[0]) * x  # inv A
            elif (map == "log") or (map == "invexp"):
                x = r[0] + np.log(1 + x) * (r[1] - r[0]) * 1.4426950408889634  # inv C (1/log(2))
            else:  # "exp" or "invlog"
                sign = 1 if (max > min) else -1
                x = sign * (np.exp(x * log(1 + abs(r[1] - r[0]))) - 1) + r[0]  # inv B
        return float(x) if fl else x

    def __init__(self, x: float = None, range: Tuple[float, float] = (0, 1),
                 map: str = "lin", clip: bool = False):
        """The usual initialization behavior is to simply set the ``self`` to the input, ``x``,
        or to :attr:`default_threshold`, if no ``x`` input is given.

        The parameters allow for the mapping of the input onto [0,1] based on its extreme limits, ``range``,
        and a mapping type, ``map``.  There is also an option ``clip``, to clip to [0,1] for safety
        (in case ``x`` might fall outside ``range``).

        The inverse of this constructor is :meth:`to_float`, i.e.: ``x == Truth(x).to_float()`` as long as their other
        parameters are compatible (equal ``range``, complementary mapping, clipping not required).

        Args:
            x: an input number.

        Other Parameters:
            range, map, clip:  relate to mapping fuzzy units.  See :meth:`scale`.

        Return:
            A :class:`Truth` object, with value on [0,1].
        """
        self.s = Truth.scale(x, "in", range, map, clip)
        if not Truth.is_valid(self):
            raise ValueError(f"A truth must be on [0,1]. {self.s} is invalid.")

    def to_float(self, range: Tuple[float, float] = (0, 1), map: str = "lin", clip: bool = False) -> float:
        """Map ``self``'s value from [0,1] to a given range.

        This is the inverse of the :class:`Truth` constructor.  When called with similar arguments
        (the same ``range`` and complementary ``map``), it will reproduce the input that initialized
        the :class:`Truth`, unless clipping was necessary, i.e., ``x == Truth(x).to_float()``.

        Other Parameters:
            range, map, clip:  relate to mapping fuzzy units.  See :meth:`scale`.


        Return:
            A number mapped onto the given range from the :class:`Truth`'s value."""
        return Truth.scale(self.s, "out", range, map, clip)

    @staticmethod
    def is_valid(s: TruthOperand) -> bool:
        """``True`` if and only if ``self`` is on [0,1]."""
        return True if (s <= 1) and (s >= 0) else False

    @staticmethod
    def clip(x: TruthOperand) -> Union[float, np.ndarray]:
        """Restricts the input's value to the domain of fuzzy truth, [0,1], by clipping.

        Arg:
            x: an input :attr:`TruthOperand` to be clipped.

        Return:
            a similar type certain to be on [0,1]."""
        r = np.clip(x, 0, 1)
        if isinstance(x, np.ndarray):
            return r
        else:
            return float(r)

    # Here I implement logical connectives from propositional calculus / electronic logic gates:---
    # References to truth tables assume s = [0,1] (unary) or (p,q) = [00,01,10,11] (binary):

    # One unary operator (ignoring insistence (11), denial (00), and proposition (01))
    # ---we only need "¬", negation (10), accessed via: "s.not_()" or "~s".

    def not_(self, **kwargs) -> Union[Truth, np.ndarray]:
        """The negation ("not", ¬) unary operator [10].

        The truth of such expressions as, "``self`` is not so".
        It may be accessed by ``Truth.not_(a)``, or ``a.not_()``, or ``~a``.

        Keyword Parameters:
            An optional :class:`.Norm` defined by keywords.  See :meth:`.Norm.define`.
            (N.B.: in the case of this operator, the calculation will probably always be the same for all
            norms:  :math:`1 - t`, the standard fuzzy negation.)

        Returns:
            The opposite of ``self``, the extent to which ``self`` is false."""
        # This will probably always be the standard fuzzy negation: 1 - self.s, but I refer to the norm
        # for consistency, and in case one chooses to implement non-standard negations.
        norm = _prepare_norm(**kwargs)
        r = norm.not_(self)
        r = Truth(r) if isinstance(r, float) else r
        return r

    # Ten binary operators:---
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
    # ↛⇸ material nonimplication              nimp         0010    p&~q            2
    # ↚⇷ converse nonimplication              ncon         0100    ~p&q            4
    # I need a basic function for each.  Since "not, and, or" must be differenced with an underscore, all have one.
    # I tried providing a |xxx| form for all the operators via https://pypi.org/project/infix/ .
    # It worked but produced spurious warnings, cautioned against its own use,
    # and looked inelegant on the page, so I dropped it.
    # The 3 Boolean operators (not, and, or) cannot be overloaded---they're stuck in the crisp world, for the best,
    # since that functionality is a little different.
    # In Python ^ is bitwise xor, but I'm not using it here because I'll need it for the sharpening (focus) operator.
    # I decided to add floor division, //, for weight as well.

    # First, the basic functions that provide the calculation via the `default_norm`:
    # and_, or_, imp_, con_, iff_, xor_;  nand_, nor_, nimp_, ncon_:

    def and_(self, *other: TruthOperand, **kwargs) -> Union[Truth, np.ndarray]:
        """The conjunction ("and", ∧) binary operator [0001].

        The truth of such expressions as, "both ``self`` and ``other``".  
        It may be accessed by ``Truth.and_(a, b,...)``, or ``a.and_(b,...)``, or ``a & b``.

        Args:
            other: zero or more :attr:`TruthOperand`\\s, since the operation is associative.

        Keyword Parameters:
            An optional :class:`.Norm` defined by keywords.  See :meth:`.Norm.define`.

        Returns:
            The conjunction of ``self`` and ``other``, the extent to which they are both true
            (to which all are simultaneously true)."""
        norm = _prepare_norm(**kwargs)
        r = norm.and_(self, *other)
        r = Truth(r) if isinstance(r, float) else r
        return r

    def or_(self, *other: TruthOperand, **kwargs) -> Union[Truth, np.ndarray]:
        """The disjunction ("inclusive or", ∨)
        binary operator [0111].

        The truth of such expressions as, "either ``self`` or ``other``".  
        It may be accessed by ``Truth.or_(a, b,...)``, or ``a.or_(b,...)``, or ``a | b``.

        Args:
            other: zero or more :attr:`TruthOperand`\\s, since the operation is associative.

        Keyword Parameters:
            An optional :class:`.Norm` defined by keywords.  See :meth:`.Norm.define`.

        Returns:
            The disjunction of ``self`` and ``other``, the extent to which either or both is true
            (to which any number are true)."""
        norm = _prepare_norm(**kwargs)
        r = norm.or_(self, *other)
        r = Truth(r) if isinstance(r, float) else r
        return r

    def imp(self, other: TruthOperand, **kwargs) -> Union[Truth, np.ndarray]:
        """The material implication ("imply", →) binary operator [1101], the converse of :meth:`con`.

        The truth of such expressions as, "``self`` implies ``other``", or "if ``self`` then ``other``".  
        It may be accessed by ``Truth.imp(a, b)``, or ``a.imp(b)``, or ``a >> b``.

        Args:
            other: one :attr:`TruthOperand`, since the operation is binary.

        Keyword Parameters:
            An optional :class:`.Norm` defined by keywords.  See :meth:`.Norm.define`.

        Returns:
            The implication of ``self`` to ``other``, the extent to which ``self`` must result in ``other``."""
        norm = _prepare_norm(**kwargs)
        r = norm.or_(norm.not_(self), other)
        r = Truth(r) if isinstance(r, float) else r
        return r

    def con(self, other: TruthOperand, **kwargs) -> Union[Truth, np.ndarray]:
        """The converse implication ("con", ←) binary operator [1011], the converse of :meth:`imp`.

        The truth of such expressions as, "``other`` implies ``self``", or "if ``other`` then ``self``".  
        It may be accessed by ``Truth.con(a, b)``, or ``a.con(b)``, or ``a << b``.

        Args:
            other: one :attr:`TruthOperand`, since the operation is binary.

        Keyword Parameters:
            An optional :class:`.Norm` defined by keywords.  See :meth:`.Norm.define`.

        Returns:
            The implication of ``other`` to ``self``,
            the extent to which ``other`` must indicate that ``self`` was true."""
        norm = _prepare_norm(**kwargs)
        r = norm.or_(self, norm.not_(other))
        r = Truth(r) if isinstance(r, float) else r
        return r

    def iff(self, other: TruthOperand, **kwargs) -> Union[Truth, np.ndarray]:
        """The equivalence or "biconditional" ("iff", ↔) binary operator [1001], the inverse of :meth:`xor`.

        It is familiar in Mathematics as "if and only if" and in Electronics as "xnor"

        The truth of such expressions as, "``self`` and ``other`` imply each other", or
        "``self`` and ``other`` are true only to the same extent".  
        It may be accessed by ``Truth.iff(a, b)``, or ``a.iff(b)``, or ``~(a @ b)``.

        Args:
            other: one :attr:`TruthOperand`, since the operation is binary.

        Keyword Parameters:
            An optional :class:`.Norm` defined by keywords.  See :meth:`.Norm.define`.

        Returns:
            The equivalence of ``self`` and ``other``, the extent to which they have the same degree of truth,
            the extent to which they occur together but not apart."""
        norm = _prepare_norm(**kwargs)
        r = norm.or_(norm.and_(self, other), norm.and_(norm.not_(self), norm.not_(other)))
        r = Truth(r) if isinstance(r, float) else r
        return r

    def xor(self, other: TruthOperand, **kwargs) -> Union[Truth, np.ndarray]:
        """The non-equivalence or "exclusive disjunction" ("exclusive or", ⨁) binary operator [0110],
        the inverse of :meth:`iff`.

        It is familiar in Electronics as "xor".

        The truth of such expressions as, "either ``self`` or ``other``, but not both together".  
        It may be accessed by ``Truth.xor(a, b)``, or ``a.xor(b)``, or ``a @ b``.

        Args:
            other: one :attr:`TruthOperand`, since the operation is binary.

        Keyword Parameters:
            An optional :class:`.Norm` defined by keywords.  See :meth:`.Norm.define`.

        Returns:
            The non-equivalence of ``self`` and ``other``, the extent to which their degrees of truth differ,
            the extent to which they occur apart but not together."""
        norm = _prepare_norm(**kwargs)
        r = norm.and_(norm.or_(self, other), norm.not_(norm.and_(self, other)))
        r = Truth(r) if isinstance(r, float) else r
        return r

    def nand(self, other: TruthOperand, **kwargs) -> Union[Truth, np.ndarray]:
        """The alternative denial ("nand", ↑) binary operator [1110], the inverse of :meth:`and_`.

        The truth of such expressions as, "``self`` and ``other`` cannot occur together".
        It may be accessed by ``Truth.nand(a, b)``, or ``a.nand(b)``, or ``~(a & b)``.

        Args:
            other: one :attr:`TruthOperand`, since the operation is binary.

        Keyword Parameters:
            An optional :class:`.Norm` defined by keywords.  See :meth:`.Norm.define`.

        Returns:
            The inverse conjunction of `self` and ``other``, the extent to which they are not both true."""
        norm = _prepare_norm(**kwargs)
        r = norm.not_(norm.and_(self, other))
        r = Truth(r) if isinstance(r, float) else r
        return r

    def nor(self, other: TruthOperand, **kwargs) -> Union[Truth, np.ndarray]:
        """The joint denial ("nor", ↓) binary operator [1000], the inverse of :meth:`or_`.

        The truth of such expressions as, "neither ``self`` nor ``other``".
        It may be accessed by ``Truth.nor(a, b)``, or ``a.nor(b)``, or ``~(a | b)``.

        Args:
            other: one :attr:`TruthOperand`, since the operation is binary.

        Keyword Parameters:
            An optional :class:`.Norm` defined by keywords.  See :meth:`.Norm.define`.

        Returns:
            The inverse disjunction of ``self`` and ``other``, the extent to which both are false."""
        norm = _prepare_norm(**kwargs)
        r = norm.not_(norm.or_(self, other))
        r = Truth(r) if isinstance(r, float) else r
        return r

    def nimp(self, other: TruthOperand, **kwargs) -> Union[Truth, np.ndarray]:
        """The material nonimplication ("nimply", :math:`\\nrightarrow`) binary operator [0010],
        the inverse of :meth:`imp`.

        The truth of such expressions as, "``self`` does not imply ``other``", or "if ``self`` then not ``other``";

        It may be accessed by ``Truth.nimp(a, b)``, or ``a.nimp(b)``, or ``~(a >> b)``.

        Args:
            other: one :attr:`TruthOperand`, since the operation is binary.

        Keyword Parameters:
            An optional :class:`.Norm` defined by keywords.  See :meth:`.Norm.define`.

        Returns:
            The nonimplication of ``self`` to ``other``; the extent to which ``self`` suppresses or inhibits ``other``;
            the extent to which the presence of ``self`` indicates that ``other`` will not occur."""
        norm = _prepare_norm(**kwargs)
        r = norm.and_(self, norm.not_(other))
        r = Truth(r) if isinstance(r, float) else r
        return r

    def ncon(self, other: TruthOperand, **kwargs) -> Union[Truth, np.ndarray]:
        """The converse non-implication ("ncon", :math:`\\nleftarrow`) binary operator [0100],
        the inverse of :meth:`con`.

        The truth of such expressions as, ``other`` does not imply ``self``", or "if ``other`` then not ``self``".  
        It may be accessed by ``Truth.ncon(a, b)``, or ``a.ncon(b)``, or ``~(a << b)``.

        Args:
            other: one :attr:`TruthOperand`, since the operation is binary.

        Keyword Parameters:
            An optional :class:`.Norm` defined by keywords.  See :meth:`.Norm.define`.

        Returns:
            The non-implication of ``other`` to ``self``,
            the extent to which ``other`` indicates that ``self`` was false."""
        norm = _prepare_norm(**kwargs)
        r = norm.and_(norm.not_(self), other)
        r = Truth(r) if isinstance(r, float) else r
        return r

    # One more binary operator peculiar to fuzzy technique:  weight:

    def weight(self, w: Union[Truth | float]) -> Truth:
        """Emphasizes or deëmphasizes a :class:`Truth`.

        One may think of a weight as a partial de-fuzzification (or, if ``w<0``, a partial fuzzification).
        It may be accessed by ``Truth.weight(t, w)``, or ``t.weight(w)``, or ``(t // w)``.

        Use weighting when you want to modify a truth's contribution to an expression.
        It applies a sigmoid transfer function to the truth's value.
        If the parameter, ``w``, is negative (de-emphasis), results tend to the :attr:`default_threshold`.
        If it is positive (emphasis), they tend to 0 or 1, depending which side of the threshold they are on.
        In any case, ``w==0`` does nothing; and, the input values {0, :attr:`default_threshold`, 1}
        are never transformed.  Note: as ``w`` approaches either infinity, the logic becomes trinary.


        Arg:
            w: The weight parameter, presumed to be on [-100, 100], though this is not a hard limit.

                To describe its effect, consider the width of the input window that varies in an interesting way.
                For ``w>0``, this is the width that gives results between .01 and .99.  For ``w<0``, it is the width
                that gives results below .49 and above .51.  We will call this the width that is "in play".
                The rest of the inputs yield results near one of the three constants---they are out of play.
                At ``w==0``, 98% of the input is in play:  inputs on (.01, .99) result in outputs on (.01, .99).

                At the extremes, only 2% is in play.  E.g.:  if ``w==100``, only inputs on (.49, .51) yield outputs on
                (.01, .99)---all other inputs yield more extreme outputs and the transfer function approaches a step
                function (the same one that decides ``bool()``, except for the excluded middle).

                If ``w==-100``, inputs on (.01, .99) yield outputs on (.49, .51)---only the outer 2% of inputs vary
                beyond this.  The transfer function approaches a constant at :attr:`default_threshold`,
                with exceptional points when the input is 0 or 1.

                (All of the above assumes that :attr:`default_threshold`  ``==.5``.)

        Returns:
            The ``self`` emphasized (made more extreme, if ``w>0``) or deëmphasized (trending towards a "maybe",
            if ``w<0``) by the weight, ``w``.

        Note:

            Concerning the weight operand:  if it is a ``float``, it is treated as above.  However, if it is
            a :class:`.Truth` object, its value on [0,1] is scaled to [-100,100] linearly, so that, e.g., .5 becomes 0.
            This is so :class:`.Truth`-valued expressions might conveniently be used as weights.

              """
        th = .5 if default_threshold <= 0 or default_threshold >= 1 else default_threshold
        if isinstance(w, Truth):
            w = w.to_float(range=(-100, 100))
        # The old formula made the in-play region vary linearly, as in the statement below:
        # k = -3.912023005428146 / np.log(.0096 * abs(w)  + .02)
        # The size of the "in play" region varies with ``w``.  At ``w==50``, it has a width of .5,
        # that is, inputs on (.25, .75)---covering 50% of the input range---yield outputs on (.01, .99).
        # But, this didn't look so good when I graphed it.  The new formula seems to vary more intuitively:
        k = -3.912023005428146 / np.log(.0096 * abs(w / 10) ** 2 + .02)
        k = 1 / k if w < 0 else k
        s = self.s if isinstance(self, Truth) else self
        r = np.where(s <= th, th * (s / th) ** k, 1 - (1 - th) * ((1 - s) / (1 - th)) ** k)
        if isinstance(r, float):
            return Truth(r)
        else:
            return r

    # Two methods for making a decision (defuzzification):

    def crisp(self, threshold: float = None) -> bool:
        """Decides the crisp value of a fuzzy truth on the basis of ``threshold``.

            Args:
                threshold:  Presumably on [0,1].  If ``self`` is as true as this, we can call it "True".

            Returns:
                The crisp version of a fuzzy truth, a defuzzification of its logical proposition,
                a final decision about its suitability.

            Note:
                * You might consider simply using the ``Truth`` in its fuzzy form, as a ``float``, in order to take
                  advantage of its nuance, i.e., by mapping it to some practical unit, perhaps with
                  the help of :meth:`.to_float`.
                * The built-in function ``bool()`` has been overridden to yield the same result as :meth:`crisp`
                  called without a threshold."""
        if threshold is None:
            threshold = Truth.default_threshold
        if self.s < threshold:
            return False
        else:
            return True

    def __bool__(self) -> bool:
        """Crisps the ``Truth`` on the basis of the class's global threshold, :attr:`Truth.default_threshold`."""
        return Truth.crisp(self)

    # Operator symbol overrides:  these are for the most common operators that have a suitable overridable symbol:
    # `~  &  |  >>  <<  @  //` (not, and, or, implies, converse, weight).

    def __invert__(self) -> Truth:
        """Overloads the unary ``~`` (bitwise not) operator."""
        # There's no way to make this operate on ``float``, ``int``, or ``bool`` objects because there is no
        # __rinvert__ method in which to put the Truth(float(self)).clip().not_(), because it's unary.
        # But ``Norm().not_(x)`` should do for that!
        return self.not_()

    def __and__(self, other: Truth) -> Truth:
        """Overloads the binary ``&`` (bitwise and) operator."""
        return _promote_and_call(self, other, "and_", "__and__")

    def __rand__(self, other: Truth) -> Truth:
        """Ensures that the overloading of ``&`` works as long as one operand is a ``Truth`` object."""
        return _promote_and_call(other, self, "and_", "__and__")

    def __or__(self, other: Truth) -> Truth:
        """Overloads the binary ``|`` (bitwise or) operator."""
        return _promote_and_call(self, other, "or_", "__or__")

    def __ror__(self, other: Truth) -> Truth:
        """Ensures that the overloading of ``|`` works as long as one operand is a ``Truth`` object."""
        return _promote_and_call(other, self, "or_", "__or__")

    def __rshift__(self, other: Truth) -> Truth:
        """Overloads the binary ``>>`` (bitwise right shift) operator."""
        return _promote_and_call(self, other, "imp", "__rshift__")

    def __rrshift__(self, other: Truth) -> Truth:
        """Ensures that the overloading of ``>>`` works as long as one operand is a ``Truth`` object."""
        print(f"in here for {self}---{other}")
        return _promote_and_call(other, self, "imp", "__rshift__")

    def __lshift__(self, other: Truth) -> Truth:
        """Overloads the binary ``<<`` (bitwise left shift) operator."""
        return _promote_and_call(self, other, "con", "__lshift__")

    def __rlshift__(self, other: Truth) -> Truth:
        """Ensures that the overloading of ``<<`` works as long as one operand is a ``Truth`` object."""
        return _promote_and_call(other, self, "con", "__lshift__")

    def __matmul__(self, other: Truth) -> Truth:
        """Overloads the binary ``@`` (matrix product) operator."""
        return _promote_and_call(self, other, "xor", "__matmul__")

    def __rmatmul__(self, other: Truth) -> Truth:
        """Ensures that the overloading of ``@`` works as long as one operand is a ``Truth`` object."""
        return _promote_and_call(other, self, "xor", "__matmul__")

    def __floordiv__(self, other: Union[Truth | float]) -> Truth:
        """Overloads the binary ``//`` (floor division) operator with :meth`.weight`.

        Arg:
            other: a ``float`` presumed to be on the range [-100,100], or a Truth scaled to this.
            See the note under :meth`.weight`.

        Returns:
            A weighted version of ``self``.
        """
        return Truth.weight(self, other)

    def __rfloordiv__(self, other: float) -> Truth:
        """Ensures that the overloading of ``//`` works as long as one operand is a ``Truth`` object."""
        return Truth.weight(Truth(other), self)

    # a trivial point:

    def __str__(self):
        """Returns just the truth value."""
        return str(f"truth: {self.s}")
