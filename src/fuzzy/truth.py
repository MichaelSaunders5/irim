"""A Truth represents a *degree of truth*, from perfect falsehood (0), to absolute truth (1) or anything in-between.

A :class:`Truth` object is essentially an immutable ``float`` presumed to be on [0,1].  It may be used to describe
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
    * :meth:`.iff` (↔, *equivalence*, "if and only if", "xnor"), and its inverse---
    * :meth:`.xor` (⨁, *exclusive or*); and the other inverses:
    * :meth:`.nand` (↑, not conjunction),
    * :meth:`.nor` (↓, not disjunction),
    * :meth:`.nimp` (⇸, not implication), and
    * :meth:`.ncon` (⇷, not converse implication).

* The 6 comparisons: ``<``, ``>`` , ``<=``, ``>=``, ``==``, ``!=``.
* Methods:

    * :meth:`.isValid` to check validity (presence on the range [0,1]), and
    * :meth:`.clip` to ensure it by clipping.

* Methods:

    * :meth:`.to_float` and
    * :meth:`Truth` (the constructor),

    to translate truths to and from a range of real numbers, linearly or logarithmically.
* A :meth:`.weight` method, for emphasizing or deëmphasizing the contribution of a truth to whatever expression
  it is in---equivalently, partially defuzzifying or refuzzifying it.
* A :meth:`.crisp` "defuzzification" method for making final decisions, i.e., for converting from a
  *fuzzy* :class:`Truth` to a *crisp* ``bool``, by comparison to a threshold (given or by global default).
  (Consider, though, the utility of simply using a :class:`Truth` in its nuanced, ineffably beautiful,
  fuzzy form.)


In keeping with the weak-typing spirit of Python, the presumption of validity (range restricted to [0,1]) is not
checked or enforced unless the user does so explicitly, with the methods :meth:`.isValid`, :meth:`.clip`, or the
constructor argument ``clip=True``.

You will notice that, as with :class:`Norm`, the operands may generally be of type ``float`` (:class:`Truth`
objects) or :class:`numpy.ndarray` with the same presumption of validity.  The facility with arrays,
though not private, is intended for use by the :class:`Value` class to work with fuzzy numbers.
Most users will need only :class:`Truth` and  :class:`Value` objects.

"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Union, Iterable, ClassVar, Tuple

import numpy as np


@dataclass
class Truth(float):
    """A "fuzzy unit" or "fit".

    As binary units are *bits* with a domain of ``bool`` ({0,1}), fuzzy units are *fits* with a domain of
    [0,1]---which, by analogy, we might call "fool".  Its numerical value measures a degree of truth on the
    continuum between perfect
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

    Operand = Union[float, np.ndarray, 'Truth']
    AssociativeOperand = Iterable[Operand]
    global_threshold: ClassVar[float] = .5
    s: float = field(default=global_threshold)

    def __str__(self):
        """Just the truth value."""
        return str(self.s)

    # The methods that deal with getting and setting:

    def is_valid(self) -> bool:
        """``True`` only if this truth is on [0,1]."""
        return True if (self.s <= 1) and (self.s >= 0) else False

    @staticmethod
    def clip(x: Operand) -> Operand:
        """Restricts the Truth's value to the domain of fuzzy truth, [0,1], by clipping.

        Arg:
            x: """
        return np.clip(x, 0, 1)

    def __init__(self, x: float = global_threshold, range: Tuple[float] = (0, 1),
                 clip: bool = False, logarithmic: bool = False):
        """Maps a ``float`` onto [0,1] based on extreme limits and mapping type.

            Args:
                x: input ``float``.
                range: range of input ``float``.  If min>max, the result ranges from 1 to 0;
                    if min==max, the result is a default ``Truth`` (a perfect maybe).
                logarithmic: whether to use logarithmic rather than linear mapping.

            Return:
                A number mapped onto the range [0,1] from the range [min,max]."""
        if range[0] == range[1]:
            self.s = Truth.global_threshold
        if logarithmic:
            self.s = math.log(1 + abs(x - range[0]), 1 + abs(range[1] - range[0]))
        else:
            self.s = (x - range[0]) / (range[1] - range[0])
        if clip:
            self.s = Truth.clip(self.s)

    def to_float(self, range: Tuple[float] = (0, 1), exponential: bool = False) -> float:
        """The reverse of :meth:`map`: maps the ``Truth`` on [0,1] onto a range of real numbers, [min,max].

            Args:
                range: range of output ``float``.  If min>max, truer truths produce lower results;
                    if min==max, the result is that.
                exponential: whether to map the result exponentially (to reverse logarithmic mapping of the truth)
                    rather than linearly.

            Return:
                A number mapped onto the range [0,1] from the range [min,max]."""
        sign = 1
        if range[0] == range[1]:
            return range[0]
        elif range[0] > range[1]:
            sign = -1
        if exponential:
            return sign * (math.exp(self.s * math.log(1 + abs(range[1] - range[0]))) - 1) + range[0]
        else:
            return self.s * (range[1] - range[0]) + range[0]

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

    def not_(self, norm=None) -> 'Truth':
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

    def and_(self, other: 'Truth' | float | int | bool, norm=None) -> 'Truth':
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

    def or_(self, other: 'Truth' | float | int | bool, norm=None) -> 'Truth':
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

    def imp(self, other: 'Truth' | float | int | bool, norm=None) -> 'Truth':
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

    def con(self, other: 'Truth' | float | int | bool, norm=None) -> 'Truth':
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

    def iff(self, other: 'Truth' | float | int | bool, norm=None) -> 'Truth':
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
        return norm.or_(norm.and_(self, other), self.nor(other, norm))

    def xor(self, other: 'Truth' | float | int | bool, norm=None) -> 'Truth':
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

    def nand(self, other: 'Truth' | float | int | bool, norm=None) -> 'Truth':
        """The alternative denial ("nand", ↑) binary operator (1110), the inverse of :meth:`and_`,
        accessed by ``a.nand(b)``.

        Returns:
            The inverse conjunction of `self` and ``other``, the extent to which they are not both true.

        Warning:
            When calling ``a.nand(b)``, ``b`` may be ``Truth | float | int | bool``,
            but ``a`` must be a ``Truth`` object."""
        if not norm:
            norm = global_norm
        return norm.not_(norm.and_(self, other))

    def nor(self, other: 'Truth' | float | int | bool, norm=None) -> 'Truth':
        """The joint denial ("nor", ↓) binary operator (1000), the inverse of :meth:`or_`,
        accessed by ``a.nor(b)``.

        Returns:
            The inverse disjunction of ``self`` and ``other``, the extent to which both are false.

        Warning:
            When calling ``a.nor(b)``, ``b`` may be ``Truth | float | int | bool``,
            but ``a`` must be a ``Truth`` object."""
        if not norm:
            norm = global_norm
        return norm.not_(norm.or_(self, other))  # self.or_(other).not_()

    def nimp(self, other: 'Truth' | float | int | bool, norm=None) -> 'Truth':
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
        return norm.and_(self, other.not_())

    def ncon(self, other: 'Truth' | float | int | bool, norm=None) -> 'Truth':
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
        return norm.and_(norm.not_(self), other)

    # Operator symbol overrides:  these are for the most common operators that have a suitable overridable symbol:
    # `~  &  |  >>  <<` (not, and, or, implies, converse).

    def __invert__(self) -> 'Truth':
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

    def __and__(self, other: 'Truth') -> 'Truth':
        """Overloads the binary ``&`` (bitwise and) operator.

        Returns:
            The fuzzy-logical conjunction (∧) of the operands.

        Warning:
            One of the operands may be ``float | int | bool``, but at least one must be a ``Truth`` object,
            or the expected fuzzy result will not be produced.
            Also, the Boolean ``and`` operator is unaffected---it returns a ``bool``:
            the result of crisping, then anding the operands."""
        return Truth.and_(self, other)

    def __rand__(self, other: 'Truth') -> 'Truth':
        """Ensures that the overloading of ``&`` works as long as one operand is a ``Truth`` object."""
        return Truth.and_(Truth(float(other), clip=True), self)

    def __or__(self, other: 'Truth') -> 'Truth':
        """Overloads the binary ``|`` (bitwise or) operator.

        Returns:
            The fuzzy-logical disjunction (∨) of the operands.

        Warning:
            One of the operands may be ``float | int | bool``, but at least one must be a ``Truth`` object,
            or the expected fuzzy result will not be produced.
            Also, the Boolean ``or`` operator is unaffected---it returns a ``bool``:
            the result of crisping, then oring the operands."""
        return Truth.or_(self, other)

    def __ror__(self, other: 'Truth') -> 'Truth':
        """Ensures that the overloading of ``|`` works as long as one operand is a ``Truth`` object."""
        return Truth.or_(Truth(float(other), clip=True), self)

    def __rshift__(self, other: 'Truth') -> 'Truth':
        """Overloads the binary ``>>`` (bitwise right shift) operator.

        Returns:
            The fuzzy-logical implication (→) of the operands.

        Warning:
            One of the operands may be ``float | int | bool``, but at least one must be a ``Truth`` object,
            or the expected fuzzy result will not be produced."""
        return Truth.imp(self, other)

    def __rrshift__(self, other: 'Truth') -> 'Truth':
        """Ensures that the overloading of ``>>`` works as long as one operand is a ``Truth`` object."""
        return Truth.imp(Truth(float(other), clip=True), self)

    def __lshift__(self, other: 'Truth') -> 'Truth':
        """Overloads the binary ``<<`` (bitwise left shift) operator.

        Returns:
            The fuzzy-logical converse implication (←) of the operands.

        Warning:
            One of the operands may be ``float | int | bool``, but at least one must be a ``Truth`` object,
            or the expected fuzzy result will not be produced."""
        return Truth.con(self, other)

    def __rlshift__(self, other: 'Truth') -> 'Truth':
        """Ensures that the overloading of ``<<`` works as long as one operand is a ``Truth`` object."""
        return Truth.con(Truth(float(other), clip=True), self)

    # One more binary operator:  weight:

    def weight(self, w: float) -> 'Truth':
        """Emphasizes or deëmphasizes a :class:`Truth`.

        Use if you want to emphasize or deëmphasize a truth's contribution to whatever expression it's in.
        It applies a sigmoid transfer function to the truth's value.
        If the parameter ``w`` is negative (de-emphasis), results tend to the :attr:`global_threshold`.
        If it is positive (emphasis), they tend to 0 or 1, depending which side of the threshold they are on.
        In any case, values of {0, global_threshold, 1} remain the same and ``w==0`` does nothing.
        One may think of a weight as a partial de-fuzzification (or, if ``w<0``, a partial fuzzification).

        Args:
            w: The weight parameter, presumed to be on [-100, 100], but this is not a hard limit.
                At ``w==0``, 98% of the input window is "in play", i.e., has results between .01 and .99.
                At the extremes, only 2% is in play.  E.g.:  if ``w==100``, inputs on (.49, .51) yield outputs on
                (.01, .99)---all other results are more extreme.  If ``w==-100``, inputs on (.01, .99) yield outputs on
                (.49, .51)---only the outer 2% of inputs vary beyond this.  The behavior of the "in play" region
                is linear with ``w``:  at ``w==50``, it has a width of .5, that is, inputs on (.25, .75) yield outputs
                on (.01, .99).  (All of the above assumes that :attr:`global_threshold`  ``==.5``.)

        Returns:
            A :class:`Truth` emphasized (made more extreme, if ``w>0``) or deëmphasized (trending towards a "maybe",
            if ``w<0``) by the weight, ``w``."""
        th = .5 if Truth.global_threshold <= 0 or Truth.global_threshold >= 1 else Truth.global_threshold
        k = -3.912023005428146 / np.log(.0096 * abs(w) + .02)
        k = 1 / k if w < 0 else k
        if self.s <= th:
            return Truth(th * (self.s / th) ** k)
        else:
            return Truth(1 - (1 - th) * ((1 - self.s) / (1 - th)) ** k)
