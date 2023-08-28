"""A fuzzy Truth represents a *degree of truth*, from perfect falsehood (0),
to absolute truth (1) or anything in-between.

A :class:`Truth` object is essentially an immutable ``float`` presumed to be on [0,1].  It may be used to describe
the truth of a proposition, the strength of an opinion, the certainty of a fact, the preference for a choice,
or any other measured quantity of physical or mental reality that might vary between established limits.

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

* A :meth:`.weight` method, for emphasizing or deëmphasizing the contribution of a truth to whatever expression
  it is in---equivalent to partially defuzzifying or refuzzifying it.
* Two methods for making final decisions---"defuzzifying" the *fuzzy* :class:`Truth` to a *crisp* ``bool``
  by comparing it to a threshold:

    * :meth:`.Truth.crisp`, which allows the threshold to be given in the call, or
    * :func:`.bool`, which refers to a global default:  :attr:`.Truth.global_threshold`.

  (Consider, though:  the best solution to your problem mightn't be crisp,
  but a variable mapped from the nuanced, ineffably beautiful, fuzzy truth.)

Note:
    As described above, six Python operators are overloaded: ``~``, ``&``, ``|``, ``>>``, ``<<``, ``@``.
    So, for example, ``a & b`` is equivalent to ``a.and_(b)``.  They function as :class:`.Truth` methods
    as long as one of their operands is a :class:`.Truth` object.

    Additionally, the  :meth:`.weight` method overloads ``^``.  (N.B.: in :mod:`.value` it is overloaded by
    :meth:`Value.focus`.)  Its first operand must be a :class:`.Truth` object.  Its second can be the usual weight
    parameter on [-100,100], but if it is a :class:`.Truth` as well, it is translated onto that range.

    Overriding supersedes the usual behavior of the operators.  Internal checks could avoid that, but I don't think
    it is worthwhile for such rare functions---I doubt they will be needed in the same scope.

    You may have noticed that trivial truth tables are not implemented.
    They are:  contradiction [0000], tautology [1111];
    and the propositions and negations of single arguments:  [0011], [0101], [1100], [1010].


Caution:
    * In keeping with the weak-typing spirit of Python, the presumption of validity (range restricted to [0,1]) is not
      checked or enforced unless the user does so explicitly, with the methods :meth:`.is_valid`, :meth:`.clip`,
      or the constructor argument ``clip=True``.
    * Operands (the arguments of the logical connectives) may be
      :class:`Truth` | float | int | bool | :class:`numpy.ndarray`, with the usual assumption of validity.
    * Although operands may be :class:`numpy.ndarray`, this is intended for private use by :class:`Value`.
    * Remember that, while the arguments of logical connectives may be of various types, they may only be called on
      :class:`Truth` objects, so, e.g., if ``a = Truth(.5)`` and ``b = .5``, then ``a.and_(b)`` works,
      but ``b.and_(a)`` fails.
    * The :meth:`Truth.and_` and :meth:`Truth.or_` methods may take zero or more arguments, because they are
      associative (although their overloaded operators, ``&`` and ``|``, are strictly binary).
      The other binary methods must take exactly one argument or an error will result.
      Yes, :meth:`.iff` and :meth:`.xor` are mathematically associative, but, in these cases, "a * b * c" is
      too easily mistaken for the more desirable "(a * b) ∧ (b * c)" or "(a * b) * (b * c)",
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
from math import log, exp
from typing import Union, ClassVar, Tuple

import numpy as np

from fuzzy.norm import Norm, global_norm


@dataclass
class Truth(float):
    """A fuzzy unit (or "fit"), representing a degree of truth.
    """
    Operand = Union[float, np.ndarray, int, bool, 'Truth']
    """A type indicating {:class:`Truth` | ``float`` | ``int`` | ``bool`` | :class:`numpy.ndarray`}.  
    Arrays are intended for private use by :class:`Value`.  The range of values is presumed to be on [0,1]."""
    global_threshold: ClassVar[float] = .5
    """The default threshold for defuzzification, used by :meth:`crisp` and ``bool()``.   It is also 
    the point of inflection for the :meth:`weight` method, and the default value for ``Truth`` objects.  
    It should be what you consider as exactly "maybe".  It may be set directly.  
    I like the intuitive default of .5 because it gives equal scope and resolution to either side of "maybe"."""
    s: float = field(default=global_threshold)

    # The methods that deal with getting and setting:

    def __init__(self, x: float = None, range: Tuple[float, float] = (0, 1),
                 logarithmic: bool = False, clip: bool = False):
        """The usual initialization behavior is to simply set the ``self`` to the input, ``x``,
        or to :attr:`global_threshold`, if no ``x`` input is given.

        The parameters allow for the mapping of the input onto [0,1] based on its extreme limits, ``range``,
        and a mapping type, ``logarithmic``.  There is also an option ``clip``, to clip to [0,1] for safety
        (in case ``x`` might fall outside ``range``).

        The inverse of this constructor is :meth:`to_float`, i.e.: ``x == Truth(x).to_float()`` as long as their other
        parameters are compatible (equal ``range``, complementary mapping, clipping not required).

        Args:
            x: an input number.
            range: the expected range, [min,max], of the input.  The default is to expect a fuzzy unit.
                If min>max, lower inputs produce truer results; if min==max, the result is a default :class:`Truth`
                (i.e., equal to :attr:`.global_threshold`---a perfect maybe).
            logarithmic: use logarithmic rather than the default linear mapping.
            clip: clip the result to [0,1]---useful if the input might fall outside the expected range.

        Return:
            A :class:`Truth` object, the value of which is mapped onto [0,1] from the range [min,max] with optional
            logarithmic mapping and clipping.
        """
        if (x is None) or (range[0] == range[1]):
            x = Truth.global_threshold
        if logarithmic:
            self.s = log(1 + abs(x - range[0]), 1 + abs(range[1] - range[0]))
        else:
            self.s = (x - range[0]) / (range[1] - range[0])
        if clip:
            self.s = Truth.clip(self.s)

    def to_float(self, range: Tuple[float, float] = (0, 1), exponential: bool = False) -> float:
        """Map ``self``'s value from [0,1] to [min,max].

        It is the inverse of the :class:`Truth`: constructor.  When called with similar arguments
        (the same ``range``, ``logarithmic == exponential``), it will reproduce the input that initialized
        the :class:`Truth`, unless clipping was necessary, i.e., ``x == Truth(x).to_float()``.

        Args:
            range: the desired extreme range, [min,max], of the output.
                If min>max, truer truths produce lower results; if min==max, that is the result.
            exponential: use exponential rather than the default linear mapping.  This is needed if
                logarithmic mapping was used in the constructor.


        Return:
            A number mapped onto the range [min,max] from the :class:`Truth`'s value."""
        sign = 1
        if range[0] == range[1]:
            return range[0]
        elif range[0] > range[1]:
            sign = -1
        if exponential:
            return sign * (exp(self.s * log(1 + abs(range[1] - range[0]))) - 1) + range[0]
        else:
            return self.s * (range[1] - range[0]) + range[0]

    @staticmethod
    def is_valid(s: Operand) -> bool:
        """``True`` if and only if ``self`` is on [0,1]."""
        return True if (s <= 1) and (s >= 0) else False

    @staticmethod
    def clip(x: Operand) -> Union[float, np.ndarray]:
        """Restricts the input's value to the domain of fuzzy truth, [0,1], by clipping.

        Arg:
            x: an input :attr:`Operand` to be clipped.

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

    def not_(self, norm: Norm = None) -> Truth:
        """The negation ("not", ¬) unary operator [10].

        The truth of such expressions as, "``self`` is not so".
        It may be accessed by ``s.not_()`` or ``~s``.

        Arg:
            norm: an optional norm.  The default is :attr:`Norm.global_norm`

        Returns:
            The opposite of ``self``, the extent to which ``self`` is false."""
        # This will probably always be the standard fuzzy negation: 1 - self.s, but I refer to the norm
        # for consistency, and in case one chooses to implement non-standard negations.
        if norm is None:
            norm = global_norm
        return Truth(norm.not_(self.s))

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

    # First, the basic functions that provide the calculation via the `global_norm`:
    # and_, or_, imp_, con_, iff_, xor_;  nand_, nor_, nimp_, ncon_:

    def and_(self, *other: Operand, norm: Norm = None) -> Truth:
        """The conjunction ("and", ∧) binary operator [0001].

        The truth of such expressions as, "both ``self`` and ``other``".  
        It may be accessed by ``a.and_(b...)`` or ``a & b``.

        Args:
            other: zero or more :attr:`Operand`, since the operation is associative.
            norm: an optional norm.  The default is :attr:`Norm.global_norm`

        Returns:
            The conjunction of ``self`` and ``other``, the extent to which they are both true
            (to which all are simultaneously true)."""
        if isinstance(other[-1], Norm):
            norm = other[-1]
            other = other[0:-1]
        if norm is None:
            norm = global_norm
        return Truth(norm.and_(self.s, *other))

    def or_(self, *other: Operand, norm: Norm = None) -> Truth:
        """The disjunction ("inclusive or", ∨) binary operator [0111].

        The truth of such expressions as, "either ``self`` or ``other``".  
        It may be accessed by ``a.or_(b...)`` or ``a | b``.

        Args:
            other: zero or more :attr:`Operand`, since the operation is associative.
            norm: an optional norm.  The default is :attr:`Norm.global_norm`

        Returns:
            The disjunction of ``self`` and ``other``, the extent to which either or both is true
            (to which any number are true)."""
        if isinstance(other[-1], Norm):
            norm = other[-1]
            other = other[0:-1]
        if norm is None:
            norm = global_norm
        return Truth(norm.or_(self.s, *other))

    def imp(self, other: Operand, norm: Norm = None) -> Truth:
        """The material implication ("imply", →) binary operator [1101], the converse of :meth:`con`.

        The truth of such expressions as, "``self`` implies ``other``", or "if ``self`` then ``other``".  
        It may be accessed by ``a.imp(b)`` or ``a >> b``.

        Args:
            other: one :attr:`Operand`, since the operation is binary.
            norm: an optional norm.  The default is :attr:`Norm.global_norm`

        Returns:
            The implication of ``self`` to ``other``, the extent to which ``self`` must result in ``other``."""
        if norm is None:
            norm = global_norm
        return Truth(norm.or_(norm.not_(self.s), other.s))

    def con(self, other: Operand, norm: Norm = None) -> Truth:
        """The converse implication ("con", ←) binary operator [1011], the converse of :meth:`imp`.

        The truth of such expressions as, "``other`` implies ``self``", or "if ``other`` then ``self``".  
        It may be accessed by ``a.con_(b)`` or ``a << b``.

        Args:
            other: one :attr:`Operand`, since the operation is binary.
            norm: an optional norm.  The default is :attr:`Norm.global_norm`

        Returns:
            The implication of ``other`` to ``self``,
            the extent to which ``other`` must indicate that ``self`` was true."""
        if norm is None:
            norm = global_norm
        return Truth(norm.or_(self.s, norm.not_(other.s)))

    def iff(self, other: Operand, norm: Norm = None) -> Truth:
        """The equivalence or "biconditional" ("iff", ↔) binary operator [1001], the inverse of :meth:`xor`.

        It is familiar in Mathematics as "if and only if" and in Electronics as "xnor"

        The truth of such expressions as, "``self`` and ``other`` imply each other", or
        "``self`` and ``other`` are true only to the same extent".  
        It may be accessed by ``a.iff(b)`` or ``~(a @ b)``.

        Args:
            other: one :attr:`Operand`, since the operation is binary.
            norm: an optional norm.  The default is :attr:`Norm.global_norm`

        Returns:
            The equivalence of ``self`` and ``other``, the extent to which they have the same degree of truth,
            the extent to which they occur together but not apart."""
        if norm is None:
            norm = global_norm
        return Truth(norm.or_(norm.and_(self.s, other.s), norm.and_(norm.not_(self.s), norm.not_(other.s))))

    def xor(self, other: Operand, norm: Norm = None) -> Truth:
        """The non-equivalence or "exclusive disjunction" ("exclusive or", ⨁) binary operator [0110],
        the inverse of :meth:`iff`.

        It is familiar in Electronics as "xor".

        The truth of such expressions as, "either ``self`` or ``other``, but not both together".  
        It may be accessed by ``a.xor(b)`` or ``a @ b``.

        Args:
            other: one :attr:`Operand`, since the operation is binary.
            norm: an optional norm.  The default is :attr:`Norm.global_norm`

        Returns:
            The non-equivalence of ``self`` and ``other``, the extent to which their degrees of truth differ,
            the extent to which they occur apart but not together."""
        if norm is None:
            norm = global_norm
        return Truth(norm.and_(norm.or_(self.s, other.s), norm.not_(norm.and_(self.s, other.s))))

    def nand(self, other: Operand, norm: Norm = None) -> Truth:
        """The alternative denial ("nand", ↑) binary operator [1110], the inverse of :meth:`and_`.

        The truth of such expressions as, "``self`` and ``other`` cannot occur together".
        It may be accessed by ``a.nand(b)`` or ``~(a & b)``.

        Args:
            other: one :attr:`Operand`, since the operation is binary.
            norm: an optional norm.  The default is :attr:`Norm.global_norm`

        Returns:
            The inverse conjunction of `self` and ``other``, the extent to which they are not both true."""
        if norm is None:
            norm = global_norm
        return Truth(norm.not_(norm.and_(self.s, other.s)))

    def nor(self, other: Operand, norm: Norm = None) -> Truth:
        """The joint denial ("nor", ↓) binary operator [1000], the inverse of :meth:`or_`.

        The truth of such expressions as, "neither ``self`` nor ``other``".
        It may be accessed by ``a.nor(b)`` or ``~(a | b)``.

        Args:
            other: one :attr:`Operand`, since the operation is binary.
            norm: an optional norm.  The default is :attr:`Norm.global_norm`

        Returns:
            The inverse disjunction of ``self`` and ``other``, the extent to which both are false."""
        if norm is None:
            norm = global_norm
        return Truth(norm.not_(norm.or_(self.s, other.s)))

    def nimp(self, other: Operand, norm: Norm = None) -> Truth:
        """The material nonimplication ("nimply", :math:`\\nrightarrow`) binary operator [0010],
        the inverse of :meth:`imp`.

        The truth of such expressions as, "``self`` does not imply ``other``", or "if ``self`` then not ``other``";

        It may be accessed by ``a.nimp(b)`` or ``~(a >> b)``.

        Args:
            other: one :attr:`Operand`, since the operation is binary.
            norm: an optional norm.  The default is :attr:`Norm.global_norm`

        Returns:
            The nonimplication of ``self`` to ``other``; the extent to which ``self`` suppresses or inhibits ``other``;
            the extent to which the presence of ``self`` indicates that ``other`` will not occur."""
        if norm is None:
            norm = global_norm
        return Truth(norm.and_(self.s, norm.not_(other.s)))

    def ncon(self, other: Operand, norm: Norm = None) -> Truth:
        """The converse non-implication ("ncon", :math:`\\nleftarrow`) binary operator [0100],
        the inverse of :meth:`con`.

        The truth of such expressions as, ``other`` does not imply ``self``", or "if ``other`` then not ``self``".  
        It may be accessed by ``a.ncon(b)`` or ``~(a << b)``.

        Args:
            other: one :attr:`Operand`, since the operation is binary.
            norm: an optional norm.  The default is :attr:`Norm.global_norm`

        Returns:
            The non-implication of ``other`` to ``self``,
            the extent to which ``other`` indicates that ``self`` was false."""
        if norm is None:
            norm = global_norm
        return Truth(norm.and_(norm.not_(self.s), other.s))

    # One more binary operator peculiar to fuzzy technique:  weight:

    def weight(self, w: float) -> Truth:
        """Emphasizes or deëmphasizes a :class:`Truth`.

        One may think of a weight as a partial de-fuzzification (or, if ``w<0``, a partial fuzzification).
        It may be accessed by ``s.weight(w)`` or ``(s^w)``.

        Use if you want to modify a truth's contribution to whatever expression it's in.
        It applies a sigmoid transfer function to the truth's value.
        If the parameter, ``w``, is negative (de-emphasis), results tend to the :attr:`global_threshold`.
        If it is positive (emphasis), they tend to 0 or 1, depending which side of the threshold they are on.
        In any case, ``w==0`` does nothing; and, the input values {0, :attr:`global_threshold`, 1}
        are never transformed.  Note: as ``w`` approaches either infinity, the logic becomes trinary.


        Arg:
            w: The weight parameter, presumed to be on [-100, 100], though this is not a hard limit.

                To describe its effect, consider the width of the input window that varies in an interesting way.
                For ``w>0``, this is the width that gives results between .01 and .99.  For ``w<0``, it is the width
                that gives results below .49 and above .51.  We will call this the width that is "in play".
                The rest of the inputs yield results near one of the three constants---they are out of play.
                At ``w==0``, 98% of the input is in play:  inputs on (.01, .99) result in outputs on (.01, .99).

                At the extremes, only 2% is in play.  E.g.:  if ``w==100``, only inputs on (.49, .51) yield outputs on
                (.01, .99)---all other inputs yeild more extreme outputs and the transfer function approaches a step
                function (the same one that decides ``bool()``, except for the excluded middle).

                If ``w==-100``, inputs on (.01, .99) yield outputs on (.49, .51)---only the outer 2% of inputs vary
                beyond this.  The transfer function approaches a constant at :attr:`global_threshold`,
                with exceptional points when the input is 0 or 1.

                The size of the "in play" region varies linearly with ``w``.  At ``w==50``, it has a width of .5,
                that is, inputs on (.25, .75)---covering 50% of the input range---yield outputs on (.01, .99).

                (All of the above assumes that :attr:`global_threshold`  ``==.5``.)

        Returns:
            The ``self`` emphasized (made more extreme, if ``w>0``) or deëmphasized (trending towards a "maybe",
            if ``w<0``) by the weight, ``w``.

        Note:
            * Because the ``^`` operator inherits very low precedence from Python, it is advised to enclose the
              weighted term in parentheses.
            * Concerning the weight operand used with the overloaded operator, ``^``:  if it is a ``float``, it is
              treated no differently than in function call ``s.weight(w)``.  However, if it is a :class:`.Truth`
              object, its value on [0,1] is scaled to [-100,100] linearly, so that, e.g., .5 becomes 0.
              This is so :class:`.Truth`-valued expressions might conveniently be used as weights.

              """
        th = .5 if Truth.global_threshold <= 0 or Truth.global_threshold >= 1 else Truth.global_threshold
        k = -3.912023005428146 / np.log(.0096 * abs(w) + .02)
        k = 1 / k if w < 0 else k
        if self.s <= th:
            return Truth(th * (self.s / th) ** k)
        else:
            return Truth(1 - (1 - th) * ((1 - self.s) / (1 - th)) ** k)

    # Two methods for making a decision (defuzzification):

    def crisp(self, threshold: float = None) -> bool:
        """Decides the crisp value of a fuzzy truth on the basis of ``threshold``.

            Args:
                threshold:  Presumably on [0,1].  If the ``Truth`` is as true as this, we can call it "true".

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
            threshold = Truth.global_threshold
        if self.s < threshold:
            return False
        else:
            return True

    def __bool__(self) -> bool:
        """Crisps the ``Truth`` on the basis of the class's global threshold, :attr:`Truth.global_threshold`."""
        return Truth.crisp(self)

    # Operator symbol overrides:  these are for the most common operators that have a suitable overridable symbol:
    # `~  &  |  >>  <<  @  ^` (not, and, or, implies, converse, weight).

    def __invert__(self) -> Truth:
        """Overloads the unary ``~`` (bitwise not) operator."""
        # There's no way to make this operate on ``float``, ``int``, or ``bool`` objects because there is no
        # __rinvert__ method in which to put the Truth(float(self)).clip().not_(), because it's unary.
        # But ``Norm().not_(x)`` should do for that!
        return self.not_()

    def __and__(self, other: Truth) -> Truth:
        """Overloads the binary ``&`` (bitwise and) operator."""
        return Truth.and_(self, other)

    def __rand__(self, other: Truth) -> Truth:
        """Ensures that the overloading of ``&`` works as long as one operand is a ``Truth`` object."""
        return Truth.and_(Truth(float(other), clip=True), self)

    def __or__(self, other: Truth) -> Truth:
        """Overloads the binary ``|`` (bitwise or) operator."""
        return Truth.or_(self, other)

    def __ror__(self, other: Truth) -> Truth:
        """Ensures that the overloading of ``|`` works as long as one operand is a ``Truth`` object."""
        return Truth.or_(Truth(float(other), clip=True), self)

    def __rshift__(self, other: Truth) -> Truth:
        """Overloads the binary ``>>`` (bitwise right shift) operator."""
        return Truth.imp(self, other)

    def __rrshift__(self, other: Truth) -> Truth:
        """Ensures that the overloading of ``>>`` works as long as one operand is a ``Truth`` object."""
        return Truth.imp(Truth(float(other), clip=True), self)

    def __lshift__(self, other: Truth) -> Truth:
        """Overloads the binary ``<<`` (bitwise left shift) operator."""
        return Truth.con(self, other)

    def __rlshift__(self, other: Truth) -> Truth:
        """Ensures that the overloading of ``<<`` works as long as one operand is a ``Truth`` object."""
        return Truth.con(Truth(float(other), clip=True), self)

    def __matmul__(self, other: Truth) -> Truth:
        """Overloads the binary ``@`` (matrix product) operator."""
        return Truth.xor(self, other)

    def __rmatmul__(self, other: Truth) -> Truth:
        """Ensures that the overloading of ``@`` works as long as one operand is a ``Truth`` object."""
        return Truth.xor(Truth(float(other), clip=True), self)

    def __xor__(self, other: Union[Truth | float]) -> Truth:
        """Overloads the binary ``^`` (bitwise xor) operator with :meth`.weight`.

        Arg:
            other: a ``float`` presumed to be on the range [-100,100], or a Truth scaled to this.
            See the note under :meth`.weight`.

        Returns:
            A weighted version of ``self``.
        """
        if isinstance(other, Truth):
            other = other.to_float(range=(-100, 100))
        return Truth.weight(self, other)

    def __rxor__(self, other: float) -> Truth:
        """Ensures that the overloading of ``^`` works as long as one operand is a ``Truth`` object."""
        # This gets sent to the magic method version to deal with the truth scaling.
        return Truth.__xor__(Truth(float(other), clip=True), self)

    # a trivial point:

    def __str__(self):
        """Returns just the truth value."""
        return str(self.s)
