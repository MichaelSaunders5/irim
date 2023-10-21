"""Come on, let's *do* somthin'."""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import pi, sqrt, ceil, floor  # pi, is finite, log
from sys import float_info
from typing import Union, Tuple  # , ClassVar,

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import fuzzy.norm  # ???
from fuzzy.crisp import Interpolator
from fuzzy.norm import Norm, default_norm
from fuzzy.number import FuzzyNumber, Domain, default_sampling_method, default_interpolator, _Numerical
from fuzzy.truth import Truth

LogicOperand = Union[FuzzyNumber, Truth, float, int, bool]  # but not np.ndarray, because it has no domain
# --> numbers get promoted to Truthy with  e=truth

MathOperand = Union[FuzzyNumber, float, int, bool]  # --> numbers get promoted to Exact with  v=value

Operand = Union[LogicOperand, MathOperand]  # = LogicOperand.  Not really used, I think


def _handle_defaults(**kwargs) -> Tuple:
    """Used for setting obscure defaults, globally by :meth:`.fuzzy_ctrl` or in a calculation or on
    individual :class:`.Operator`\\ s.  See :meth:`.fuzzy_ctrl`."""
    norm_args = kwargs.get('norm', None)
    if norm_args is not None:
        norm = Norm.define(**norm_args)
    else:
        norm = getattr(fuzzy.norm, 'default_norm')
    threshold = kwargs.get('threshold', None)  # float on [0,1]
    sampling = kwargs.get('sampling', None)  # str: either "Chebyshev" or "uniform"
    interpolator = kwargs.get('interpolator', None)  # Interpolator
    interpolator = Interpolator(interpolator)
    resolution = kwargs.get('resolution', None)  # float
    # crisp = kwargs.get('crisper', None)  # Crisper
    crisper = None  # Crisper()  # TODO: decode the dict "crisp" and construct a crisper here as it instructs.
    return norm, threshold, sampling, interpolator, resolution, crisper


def fuzzy_ctrl(**kwargs):
    """Set global defaults.

    This is a convenience for controlling default attributes of the :mod:`.fuzzy` package with one simple interface,
    and without having to import everything.  You set the keywords to the parameter you want or, in more complex
    cases, a dictionary with various keys.  Its counterpart, :meth:`.fuzzy_ctrl_show`, prints them out.

    Keyword Parameters:
        norm= {dict}
            Sets :attr:`.norm.default_norm`, the :class:`.Norm` (a t-norm, co-norm pair) that is the basic definition
            for all fuzzy operations.

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

        threshold= float on [0,1]
            The threshold for crisping :class:`.Truth`---the fuzzy truth defining the boundary between Boolean ``True``
            and not ``False``, used by :meth:`.Truth.crisp` and ``bool``.  The default, .5, is probably best left alone.

        sampling= {"Chebyshev", "uniform"}
            Used when :class:`._Numerical`\\ s are constructed internally for calculations.  The default, "Chebyshev",
            creates near-minimax polynomials and probably introduces less error, but "uniform" might be appropriate for
            some highly-structured t(v) functions with narrow peaks.
        interpolator= str | tuple
                Defines the default :class:`.Interpolator` used for constructing t(v) functions in :class:`.CPoints`
                and in evaluating any :class:`._Numerical`---when internal resampling is unavoidable.

            +--------------------+-------------------------------------------------------------------+
            |  ``kind=``         | note                                                              |
            +====================+===================================================================+
            | ``"linear"``       | linear                                                            |
            +--------------------+-------------------------------------------------------------------+
            | ``"akima"``        | due to Hiroshi Akima                                              |
            +--------------------+-------------------------------------------------------------------+
            | ``"bary"``         | barycentric                                                       |
            +--------------------+-------------------------------------------------------------------+
            | ``"krogh"``        | due to Fred T. Krogh                                              |
            +--------------------+-------------------------------------------------------------------+
            |  ``"pchip"``       | piecewise cubic Hermite interpolating polynomial                  |
            +--------------------+-------------------------------------------------------------------+
            |  cubic splines     |                                                                   |
            +--------------------+-------------------------------------------------------------------+
            | ``"not-a-knot"``   | Last two segments at each end are the same polynomial.            |
            +--------------------+-------------------------------------------------------------------+
            | ``"periodic"``     | The first and last suitabilities must be identical.               |
            +--------------------+-------------------------------------------------------------------+
            | ``"clamped"``      | ((1, 0), (1, 0))                                                  |
            +--------------------+-------------------------------------------------------------------+
            | ``"natural"``      | ((2, 0), (2, 0))                                                  |
            +--------------------+-------------------------------------------------------------------+
            | ``((a,b), (c,d))`` | (*left, right*): (*derivative order* {1,2}, *derivative value*)   |
            +--------------------+-------------------------------------------------------------------+

        resolution= float
            The default resolution for creating :class:`.Map`\\ s and for :meth:`.FuzzyNumber.crisp`\\ ing.  You
            really shouldn't rely on *any* default for this, since it depends on the units of your calculation and
            how precise you need them to be---things which only you know.
        crisper= Crisper
            I'll let you know when I write some.  They'll probably take parameters.

    """
    norm, threshold, sampling, interpolator, resolution, crisper = _handle_defaults(**kwargs)
    if norm is not None:
        setattr(fuzzy.norm, 'default_norm', norm)
    if threshold is not None:
        setattr(fuzzy.truth, 'default_threshold', threshold)
    if sampling is not None:
        setattr(fuzzy.number, 'default_sampling_method', sampling)
    if interpolator is not None:
        setattr(fuzzy.number, 'default_interpolator', interpolator)
    if resolution is not None:
        setattr(fuzzy.number, 'default_resolution', resolution)
    if crisper is not None:
        setattr(fuzzy.number, 'default_crisper', crisper)


def fuzzy_ctrl_show() -> None:
    """Prints out the current global defaults of the :mod:`.fuzzy` package.

    You can set them with its counterpart, :meth:`.fuzzy_ctrl`."""
    print("These are the current global defaults for the fuzzy package:")
    print(f"norm:  {getattr(fuzzy.norm, 'default_norm')}")
    print(f"truth threshold:  {getattr(fuzzy.truth, 'default_threshold')} == 'maybe' ")
    print(f"sampling method:  {getattr(fuzzy.number, 'default_sampling_method')}")
    print(f"interpolator:  {getattr(fuzzy.number, 'default_interpolator')}")
    print(f"resolution:  {getattr(fuzzy.number, 'default_resolution')}")
    print(f"crisper:  {getattr(fuzzy.number, 'default_crisper')}")


def safe_div(x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Division by zero results in the largest possible ``float``; 0/0 returns 0."""
    # np.seterr(all='ignore')  # guilt-free division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.divide(x, y)
    r = np.nan_to_num(r, nan=0, posinf=float_info.max / 2, neginf=-float_info.max / 2)
    return r


def _get_sample_points(cd: Domain, cn: int, sampling_method: str) -> np.ndarray:
    if sampling_method == "Chebyshev":  # Find the values to sample at, Chebyshev or uniform.
        m = cn + 4
        cv = np.arange(2, m - 1, 1)
        cv = .5 * cd.span() * (-np.cos((cv - .5) * pi / m)) / np.cos(1.5 * pi / m) + cd.center()
    else:
        cv = np.linspace(cd[0], cd[1], num=(cn + 2))
    return cv


def _promote_and_call(a, b, op_name, logic: bool = True):
    """Appropriately promote the operand types and call the operator.

    This exists so that we may use :class:`.Truth`, :class:`.FuzzyNumber`, and built-in Python numerical types
    (``float``, ``int``, ``bool``) interchangeably in overloaded fuzzy expressions without worrying too much about it.
    An analogous function exists in :mod:`truth`, to deal with :class:`.Truth` operations.

    Here, in :mod:`operator`, the operators may be logical or arithmetic and one operand must be :class:`.FuzzyNumber`,
    or we would not have arrived at this function (e.g., an operation on two ``float`s would be handled by Python in
    the usual way).  (We might well have ended up here because the analogous method :meth:`.truth._promote_and_call`
    promoted something to a :class:`.FuzzyNumber` and called one of :class:`.Operator`'s magic methods to handle it.)

    * If the other operand is also a :class`.FuzzyNumber`, no promotion is necessary.
    * If it is a :class:`.Truth`, it is promoted to :class:`.Truthy`.
    * If it is a number, the promotion behavior depends on whether the operator is logical or mathematical:

        * If the operator is logical:

            * And the number is on [0,1], it is promoted to :class:`.Truthy`.
            * But if it is not on [0,1], it is promoted to :class:`.Exactly`.

        * If the operator is mathematical, it is promoted to :class:`.Exactly`.

    In any case, the appropriate operator method is called with the resulting promotions, with the order of ``a``
    and ``b`` preserved, since not all operands are commutative.

    Args:
        a, b: operands which may be :class`.Truth`, :class`.FuzzyNumber`, ``float``, ``int``, or ``bool``.
        op_name: in a string, the name of the :class`.Operator` method called for, e.g., ``imp``.
        logic: true if the operator being called is logical.

    Returns:
        A :class`.Truth`, or :class`.FuzzyNumber` that is the result of the indicated operation on ``a`` and ``b``.
        """
    a_fuz, b_fuz = isinstance(a, FuzzyNumber), isinstance(b, FuzzyNumber)
    if a_fuz and b_fuz:
        return eval("Operator." + op_name + "(a, b)")  # Call the operator if both are FuzzyNumbers.
    p = a if b_fuz else b  # Promote the one that isn't a FuzzyNumber.
    if isinstance(p, Truth):  # Even if the op is math, using a Truth operand suggests that you mean a Truthy.
        from fuzzy.literal import Truthy
        p = Truthy(p)
    else:  # It must be a number.
        p = float(p)
        if logic:  # Using a logic operator...
            if (p >= 0) and (p <= 1):  # ... with a number that looks like a Truth; or...
                from fuzzy.literal import Truthy
                p = Truthy(p)
            else:  # ... with a number that looks like a crisp value.
                from fuzzy.literal import Exactly
                p = Exactly(p)
        else:  # Using a math operator, assume the number is a crisp value.
            from fuzzy.literal import Exactly
            p = Exactly(p)
    a = p if b_fuz else a  # Put a and b back in the original order.
    b = b if b_fuz else p
    return eval("Operator." + op_name + "(a, b)")  # Call the operator for FuzzyNumbers


class Operator(FuzzyNumber, ABC):
    """The base class for operators.  It handles obscure calculation parameters usually left to global defaults.

    It sets any attributes an operator *might* need to consult.  If an operator needs one, it looks in itself, and it
    hasn't been set, it takes the global default, e.g.:  ``n = getattr(self, 'norm', default_norm)')``.
    Possible attributes are:

    * fuzzy.norm.default_norm --- needed for most math an logic operators.
    * fuzzy.truth.default_threshold --- needed for :class:`.Weight`.
    * fuzzy.number.default_sampling_method --- maybe; used when resampling in some operators.
    * fuzzy.number.default_interpolator --- maybe; used when resampling in some operators.
    * fuzzy.number.default_crisper  --- not used in operators; given for :meth:`.FuzzyNumber.crisp`.
    * fuzzy.number.default_resolution  --- not used in operators; given for :meth:`._get_numerical`
      and related calls (:meth:`.FuzzyNumber.map`, :meth:`.FuzzyNumber.crisp`).

    If you set an attribute of an operator via ``**kwargs``, it sets the attribute in itself, its operands,
    their operands, and so on---its whole sub-tree---except where these have already been explicitly set.

    How do you make an operator?  I think by just:

    1. implement __str__
    2. implement d_op (if the superclass doesn't do the right thing with it).
    3. define either:

        a. l_op if it's a logic operator, or
        b. m_op if it's a math operator

    4. implement t (can I automate that too?)

    The unary/binary/associative classes should handle the handling of 1,2,many operands in their get_numericals
    I think that's it."""

    def __init__(self, *operands: Operand, **kwargs):
        """See :meth:`.fuzzy_ctrl` for parameter docs.

        Subclasses of Operator will access these by, e.g.:
        ``n = getattr(self, 'norm', default_norm)``, i.e. See if one has been set for this expression/operator.
        If not, use the global default.

        ook: override_operand_kwargs:  if true, the operator passes on its attributes to its operands:
        """
        super().__init__()
        self.operands = list(operands)
        self._set_attributes(**kwargs)

    def _get_numerical(self, precision: int, allowed_domain: Domain = None) -> _Numerical:
        """operate on the operand over its consequential domain with the required precision.
        :class:`.UnaryOperator`, :class:`.BinaryOperator`, and :class:`.AssociativeOperator` must each implement
        :meth:`._operate(precision, allowed_domain)` to handle one, two , or many operands."""
        # if allowed_domain is not None:      # TODO this is not the place to impose domain  on binaries
        #     allowed_domain = self.d_op(True, allowed_domain)  # Only sample where it matters.-------------
        # iT LOOKS LIKE I've just renamed "_get_numerical" "_operate" in this neighborhood!
        # Ah, at least this can be the one place I clean up the exceptional points with:
        return _Numerical._impose_domain(self._operate(precision, allowed_domain), allowed_domain)

    @abstractmethod
    def _operate(self, precision: int, allowed_domain: Domain = None):  # -> _Numerical
        """Every operator must be able to get numerical representations of its operands,
        disassemble them, operate on their parts, return a numerical result.

            * fuzzy.norm.default_norm --- needed for most math an logic operators.
            * fuzzy.truth.default_threshold --- needed for :class:`.Weight`.
            * fuzzy.number.default_sampling_method --- maybe; used when resampling in some operators.
            * fuzzy.number.default_interpolator --- maybe; used when resampling in some operators.
            * fuzzy.number.default_crisper  --- not used in operators; given for :meth:`.FuzzyNumber.crisp`.
            * fuzzy.number.default_resolution"""
        # attribute setups look like:
        # n = getattr(self, 'norm', default_norm)
        # threshold = getattr(self, 'threshold', default_threshold)
        # sampling = getattr(self, 'sampling', default_threshold)
        # interpolator = getattr(self, 'interpolator', default_interpolator)
        # resolution = getattr(self, 'resolutionresolution', default_resolution)    # probably never needed.
        # crisper = getattr(self, 'crisper', default_crisper)    # probably never needed.

        # Associative operators could be set up like:
        # cd, cn, cv, ct, xv, xt, e = [], [], [], [], [], [], []
        # for a in self.operands:
        #     na = a._get_numerical(precision, allowed_domain)  # a Numerical version of each operand
        #     cd.append(na.cd)
        #     cn.append(na.cn)
        #     cv.append(na.cv)
        #     ct.append(na.ct)
        #     xv.append(na.xv)
        #     xt.append(na.xt)
        #     e.append(na.e)

    def _set_attributes(self, **kwargs):
        """Take any attribute indicated in ``kwargs``, set it, if it has not already been set, and do this for all
        contained operands that are also operators.  I.e., the operator at the head of the tree pushes the kwargs
        down into the tree, overriding the defaults but respecting those given in its operands.  I.e., one can use
        a special norm for one operator, or for part of the expression, or for the whole expression."""
        norm, threshold, sampling, interpolator, resolution, crisper = _handle_defaults(**kwargs)
        if (norm is not None) and not hasattr(self, "norm"):
            setattr(self, 'norm', norm)  # Used by most Operators.
        if (threshold is not None) and not hasattr(self, "threshold"):
            setattr(self, 'threshold', threshold)  # Used by Weight.
        if (sampling is not None) and not hasattr(self, "sampling"):
            setattr(self, 'sampling', sampling)  # Used by non-unary operators
        if (interpolator is not None) and not hasattr(self, "interpolator"):
            setattr(self, 'interpolator', interpolator)  # Used by non-unary operators
        if (resolution is not None) and not hasattr(self, "resolution"):
            setattr(self, 'resolution', resolution)  # Probably never needed by an Operator.
        if (crisper is not None) and not hasattr(self, "crisper"):
            setattr(self, 'crisper', crisper)  # Probably never needed by an Operator.
        if self.operands is not None:
            for a in self.operands:
                if isinstance(a, Operator):
                    a._set_attributes(**kwargs)

    def _prepare_operand(self, a: Operand, **kwargs) -> None:
        """Operators get their rare attributes set here (or they will keep those given,
        or they will find the global defaults)."""
        if isinstance(a, Operator):
            a._set_attributes(**kwargs)

    @abstractmethod
    def _op(self, *args) -> Union[np.ndarray, float]:
        """In each Operator class, this meth defines it.  You give it what it needs and it  gives you the result.
        Who calls it?  _operate calls something that calls it"""

    def d_op(self, inv: bool, *d: Domain) -> Domain:
        """Tells how a given domain will be transformed by the operation.

        When calling down the tree with :meth:`._get_domain`, we want to know how the operand domains will be
        transformed into the result domain.  When calling with :meth:`._get_numerical`, we use the inverse on the
        ``allowed_domain`` to which we seek to restrict the result---and so tell the operand
        (ultimately a :class:`.Literal`) to only sample the domain that will matter---thus conserving precision.

        E.g.  Suppose an operator is simple "+5"---it adds a five to everything.  It's operand has a domain of
        d = [0,10].  We want to know the resulting domain: ``d_op(d, False)`` tells us: [5,15].  We are only interested
        in results on a = [0,10], so when we want to do the operation by getting the numerical result, we tell the
        operator not to sample where it won't matter to us, so it does ``d_op(a, True)`` to find that this means [-5,5].
        Its business is then on the intersection of that and what it has: [-5,5].intersection[0,10], so it only
        samples [0,5] of its operand to create the numerical it will return.

        This idea only works for unary operators!  with binaries, the operands affect each other.
        """

    # Here is where all the operator functions used in expressions go:

    def not_(self, **kwargs) -> Operator:
        """Returns a :class:`.Not` object with ``self`` as the operand (for logical negation, ¬a).

        Call by:

        * ``a.not_(**kwargs)``
        * ``Operator.not_(a, **kwargs)``
        * ``~a``

        Where

        * ``a`` is a :class:`.FuzzyNumber` (e.g., an :class:`.Operator` or :class:`.Literal`); and
        * ``b`` is a :class:`.FuzzyNumber`, or ``float`` or ``int`` or ``bool``.
        * For ``kwargs`` see :meth:`.fuzzy_ctrl`.
        """
        if isinstance(self, And) and (len(self.operands) == 2):
            return Nand(*self.operands, **kwargs)
        elif isinstance(self, Or) and (len(self.operands) == 2):
            return Nor(*self.operands, **kwargs)
        elif isinstance(self, Xor):
            return Iff(*self.operands, **kwargs)
        elif isinstance(self, Imp):
            return Nimp(*self.operands, **kwargs)
        elif isinstance(self, Con):
            return Ncon(*self.operands, **kwargs)
        elif isinstance(self, Con):
            return Ncon(*self.operands, **kwargs)
        elif isinstance(self, Nand):
            return And(*self.operands, **kwargs)
        elif isinstance(self, Nor):
            return Or(*self.operands, **kwargs)
        elif isinstance(self, Nimp):
            return Imp(*self.operands, **kwargs)
        elif isinstance(self, Ncon):
            return Con(*self.operands, **kwargs)
        elif isinstance(self, Iff):
            return Xor(*self.operands, **kwargs)
        else:  # The above cruft may make the tree more efficient
            return Not(self, **kwargs)  # This line is all that's necessary.

    def neg(self, **kwargs) -> Negative:
        """Returns a :class:`.Negative` object with ``self`` as the operand (for arithmetic negation, -a).

        Call by:

        * ``a.neg(**kwargs)``
        * ``Operator.neg(a, **kwargs)``
        * ``-a``

        Where

        * ``a`` is a :class:`.FuzzyNumber` (e.g., an :class:`.Operator` or :class:`.Literal`); and
        * ``b`` is a :class:`.FuzzyNumber`, or ``float`` or ``int`` or ``bool``.
        * For ``kwargs`` see :meth:`.fuzzy_ctrl`.
        """
        return Negative(self, **kwargs)

    def reciprocal(self, **kwargs) -> Reciprocal:
        """Returns a :class:`.Reciprocal` object with ``self`` as the operand (for 1/a).

        Call by:

        * ``a.reciprocal(**kwargs)``
        * ``Operator.reciprocal(a, **kwargs)``

        Where

        * ``a`` is a :class:`.FuzzyNumber` (e.g., an :class:`.Operator` or :class:`.Literal`); and
        * ``b`` is a :class:`.FuzzyNumber`, or ``float`` or ``int`` or ``bool``.
        * For ``kwargs`` see :meth:`.fuzzy_ctrl`.
        """
        return Reciprocal(self, **kwargs)

    def abs(self, **kwargs) -> Absolute:
        """Returns a :class:`.Absolute` object with ``self`` as the operand (for absolute value, :math:`|a|`).

        Call by:

        * ``a.abs(**kwargs)``
        * ``Operator.abs(a, **kwargs)``
        * ``+a``

        Where

        * ``a`` is a :class:`.FuzzyNumber` (e.g., an :class:`.Operator` or :class:`.Literal`); and
        * ``b`` is a :class:`.FuzzyNumber`, or ``float`` or ``int`` or ``bool``.
        * For ``kwargs`` see :meth:`.fuzzy_ctrl`.
        """
        return Absolute(self, **kwargs)

    # TODO: make and_ and or_ associative.  Is this possible for overloads?
    # TODO: passing kwargs to symbolic calls?
    def and_(self, b: Operand, **kwargs) -> And:
        """Returns an :class:`.And` object with ``self`` and ``b`` as operands (for self ∧ b).

        Call by:

        * ``a.and_(b, **kwargs)``
        * ``Operator.and_(a, b, **kwargs)``
        * ``a & b``

        Where

        * ``self``, or ``a`` is a :class:`.FuzzyNumber` (e.g., an :class:`.Operator` or :class:`.Literal`); and
        * ``b`` is a :class:`.FuzzyNumber`, or :class:`.Truth`, or ``float`` or ``int`` or ``bool``.
        * For ``kwargs`` see :meth:`.fuzzy_ctrl`.
        """
        return And(self, b, **kwargs)

    def or_(self, *b: Operand, **kwargs) -> Or:
        """Returns an :class:`.Or` object with ``self`` and ``b`` as operands (for self ∨ b).

        Call by:

        * ``a.or_(b, **kwargs)``
        * ``Operator.or_(a, b, **kwargs)``
        * ``a | b``

        Where

        * ``self``, or ``a`` is a :class:`.FuzzyNumber` (e.g., an :class:`.Operator` or :class:`.Literal`); and
        * ``b`` is a :class:`.FuzzyNumber`, or :class:`.Truth`, or ``float`` or ``int`` or ``bool``.
        * For ``kwargs`` see :meth:`.fuzzy_ctrl`.
        """
        return Or(self, *b, **kwargs)

    def imp(self, b: Operand, **kwargs) -> Imp:
        """Returns an :class:`.Imp` object with ``self`` and ``b`` as operands (for self → b).

        Call by:

        * ``a.imp(b, **kwargs)``
        * ``Operator.imp(a, b, **kwargs)``
        * ``a >> b``

        Where

        * ``self``, or ``a`` is a :class:`.FuzzyNumber` (e.g., an :class:`.Operator` or :class:`.Literal`); and
        * ``b`` is a :class:`.FuzzyNumber`, or :class:`.Truth`, or ``float`` or ``int`` or ``bool``.
        * For ``kwargs`` see :meth:`.fuzzy_ctrl`.
        """
        return Imp(self, b, **kwargs)

    def con(self, b: Operand, **kwargs) -> Con:
        """Returns a :class:`.Con` object with ``self`` and ``b`` as operands (for self ← b).

        Call by:

        * ``a.con(b, **kwargs)``
        * ``Operator.con(a, b, **kwargs)``
        * ``a << b``

        Where

        * ``self``, or ``a`` is a :class:`.FuzzyNumber` (e.g., an :class:`.Operator` or :class:`.Literal`); and
        * ``b`` is a :class:`.FuzzyNumber`, or :class:`.Truth`, or ``float`` or ``int`` or ``bool``.
        * For ``kwargs`` see :meth:`.fuzzy_ctrl`.
        """
        return Con(self, b, **kwargs)

    def xor(self, b: Operand, **kwargs) -> Xor:
        """Returns a :class:`.Xor` object with ``self`` and ``b`` as operands (for self ⨁ b).

        Call by:

        * ``a.xor(b, **kwargs)``
        * ``Operator.xor(a, b, **kwargs)``
        * ``a @ b``

        Where

        * ``self``, or ``a`` is a :class:`.FuzzyNumber` (e.g., an :class:`.Operator` or :class:`.Literal`); and
        * ``b`` is a :class:`.FuzzyNumber`, or :class:`.Truth`, or ``float`` or ``int`` or ``bool``.
        * For ``kwargs`` see :meth:`.fuzzy_ctrl`.
        """
        return Xor(self, b, **kwargs)

    def nand(self, b: Operand, **kwargs) -> Nand:
        """Returns a :class:`.Nand` object with ``self`` and ``b`` as operands (for self ↑ b).

        Call by:

        * ``a.nand(b, **kwargs)``
        * ``Operator.nand(a, b, **kwargs)``
        * ``~(a & b)``

        Where

        * ``self``, or ``a`` is a :class:`.FuzzyNumber` (e.g., an :class:`.Operator` or :class:`.Literal`); and
        * ``b`` is a :class:`.FuzzyNumber`, or :class:`.Truth`, or ``float`` or ``int`` or ``bool``.
        * For ``kwargs`` see :meth:`.fuzzy_ctrl`.
        """
        return Nand(self, b, **kwargs)

    def nor(self, b: Operand, **kwargs) -> Nor:
        """Returns a :class:`.Nor` object with ``self`` and ``b`` as operands (for self ↓ b).

        Call by:

        * ``a.imp(b, **kwargs)``
        * ``Operator.imp(a, b, **kwargs)``
        * ``~(a | b)``

        Where

        * ``self``, or ``a`` is a :class:`.FuzzyNumber` (e.g., an :class:`.Operator` or :class:`.Literal`); and
        * ``b`` is a :class:`.FuzzyNumber`, or :class:`.Truth`, or ``float`` or ``int`` or ``bool``.
        * For ``kwargs`` see :meth:`.fuzzy_ctrl`.
        """
        return Nor(self, b, **kwargs)

    def nimp(self, b: Operand, **kwargs) -> Nimp:
        """Returns a :class:`.Nimp` object with ``self`` and ``b`` as operands (for self :math:`\\nrightarrow` b).

        Call by:

        * ``a.nimp(b, **kwargs)``
        * ``Operator.nimp(a, b, **kwargs)``
        * ``~(a >> b)``

        Where

        * ``self``, or ``a`` is a :class:`.FuzzyNumber` (e.g., an :class:`.Operator` or :class:`.Literal`); and
        * ``b`` is a :class:`.FuzzyNumber`, or :class:`.Truth`, or ``float`` or ``int`` or ``bool``.
        * For ``kwargs`` see :meth:`.fuzzy_ctrl`.
        """
        return Nimp(self, b, **kwargs)

    def ncon(self, b: Operand, **kwargs) -> Ncon:
        """Returns a :class:`.Ncon` object with ``self`` and ``b`` as operands (for self :math:`\\nleftarrow` b).

        Call by:

        * ``a.ncon(b, **kwargs)``
        * ``Operator.ncon(a, b, **kwargs)``
        * ``~(a << b)``

        Where

        * ``self``, or ``a`` is a :class:`.FuzzyNumber` (e.g., an :class:`.Operator` or :class:`.Literal`); and
        * ``b`` is a :class:`.FuzzyNumber`, or :class:`.Truth`, or ``float`` or ``int`` or ``bool``.
        * For ``kwargs`` see :meth:`.fuzzy_ctrl`.
        """
        return Ncon(self, b, **kwargs)

    def iff(self, b: Operand, **kwargs) -> Iff:
        """Returns an :class:`.Iff` object with ``self`` and ``b`` as operands (for self ↔ b).

        Call by:

        * ``a.iff(b, **kwargs)``
        * ``Operator.iff(a, b, **kwargs)``
        * ``~(a @ b)``

        Where

        * ``self``, or ``a`` is a :class:`.FuzzyNumber` (e.g., an :class:`.Operator` or :class:`.Literal`); and
        * ``b`` is a :class:`.FuzzyNumber`, or :class:`.Truth`, or ``float`` or ``int`` or ``bool``.
        * For ``kwargs`` see :meth:`.fuzzy_ctrl`.
        """
        return Iff(self, b, **kwargs)

    def add(self, b: Operand, **kwargs) -> Add:
        """Returns an :class:`.Add` object with ``self`` and ``b`` as operands (for self + b).

        Call by:

        * ``a.add(b, **kwargs)``
        * ``Operator.add(a, b, **kwargs)``
        * ``a + b``

        Where

        * ``self``, or ``a`` is a :class:`.FuzzyNumber` (e.g., an :class:`.Operator` or :class:`.Literal`); and
        * ``b`` is a :class:`.FuzzyNumber`, or :class:`.Truth`, or ``float`` or ``int`` or ``bool``.
        * For ``kwargs`` see :meth:`.fuzzy_ctrl`.
        """
        return Add(self, b, **kwargs)

    # Here is where all the operator overloads used in expressions go:

    def __invert__(self) -> Operator:
        """Overloads the unary ``~`` (bitwise not) operator to perform logical negation on ``self``."""
        # There's no way to make this operate on ``float``, ``int``, or ``bool`` objects because there is no
        # __rinvert__ method in which to put the Operator.not_(Exact(float(self))), because it's unary.
        return self.not_()

    def __neg__(self) -> Operator:
        """Overloads the unary ``-`` (minus) operator to perform arithmetic negation on ``self``."""
        # There's no way to make this operate on ``float``, ``int``, or ``bool`` objects because there is no
        # __neg__ method in which to put the Operator.neg(Exact(float(self))), because it's unary.
        return self.neg()

    def __pos__(self) -> Operator:
        """Overloads the unary ``+`` (positive) operator to perform absolute value on ``self``."""
        # There's no way to make this operate on ``float``, ``int``, or ``bool`` objects because there is no
        # __pos__ method in which to put the Operator.abs(Exact(float(self))), because it's unary.
        return self.abs()

    def __and__(self, other: Operand) -> And:
        """Overloads the binary ``&`` (bitwise and) operator."""
        return _promote_and_call(self, other, "and_")

    def __rand__(self, other: Operand) -> And:
        """Ensures that the overloading of ``&`` works as long as one operand is a ``Truth`` object."""
        return _promote_and_call(other, self, "and_")

    def __or__(self, other: Operand) -> Or:
        """Overloads the binary ``|`` (bitwise or) operator."""
        return _promote_and_call(self, other, "or_")

    def __ror__(self, other: Operand) -> Or:
        """Ensures that the overloading of ``|`` works as long as one operand is a ``Truth`` object."""
        return _promote_and_call(other, self, "or_")

    def __rshift__(self, other: Operand) -> Imp:
        """Overloads the binary ``>>`` (bitwise right shift) operator."""
        return _promote_and_call(self, other, "imp")

    def __rrshift__(self, other: Operand) -> Imp:
        """Ensures that the overloading of ``>>`` works as long as one operand is a ``Truth`` object."""
        return _promote_and_call(other, self, "imp")

    def __lshift__(self, other: Operand) -> Con:
        """Overloads the binary ``<<`` (bitwise left shift) operator."""
        return _promote_and_call(self, other, "con")

    def __rlshift__(self, other: Operand) -> Con:
        """Ensures that the overloading of ``<<`` works as long as one operand is a ``Truth`` object."""
        return _promote_and_call(other, self, "con")

    def __matmul__(self, other: Operand) -> Xor:
        """Overloads the binary ``@`` (matrix product) operator."""
        return _promote_and_call(self, other, "xor")

    def __rmatmul__(self, other: Operand) -> Xor:
        """Ensures that the overloading of ``@`` works as long as one operand is a ``Truth`` object."""
        return _promote_and_call(other, self, "xor")

    # overloads for math operators

    def __add__(self, other: Operand) -> Add:
        """Overloads the binary ``+`` (addition) operator."""
        return _promote_and_call(self, other, "add", False)

    def __radd__(self, other: Operand) -> Add:
        """Ensures that the overloading of ``+`` works as long as one operand is a ``Truth`` object."""
        return _promote_and_call(other, self, "add", False)

    # overloads for modifiers

    def __floordiv__(self, other: Union[Truth | float]) -> Truth:
        """Overloads the binary ``//`` (floor division) operator with :meth`.weight`.

        Arg:
            other: a ``float`` presumed to be on the range [-100,100], or a Truth to be scaled to this.
            See the note under :meth`.weight`.

        Returns:
            A weighted version of ``self``.
        """
        if isinstance(other, Truth):
            other = other.to_float(range=(-100, 100))
        return _promote_and_call(self, other, "weight")

    def __rfloordiv__(self, other: float) -> Truth:
        """Ensures that the overloading of ``//`` works as long as one operand is a ``Truth`` object."""
        # This gets sent to the magic method version to deal with the truth scaling.
        s = self.to_float(range=(-100, 100)) if isinstance(self, Truth) else self
        return _promote_and_call(other, s, "weight")

    def __xor__(self, other: Union[Truth | float]) -> Truth:
        """Overloads the binary ``^`` (bitwise xor) operator with :meth`.focus`.

        Arg:
            other: a ``float`` presumed to be on the range [-100,100], or a Truth scaled to this.
            See the note under :meth`.focus`.

        Returns:
            A weighted version of ``self``.
        """
        if isinstance(other, Truth):
            other = other.to_float(range=(-100, 100))
        return _promote_and_call(self, other, "focus")

    def __rxor__(self, other: float) -> Truth:
        """Ensures that the overloading of ``^`` works as long as one operand is a ``Truth`` object."""
        # This gets sent to the magic method version to deal with the truth scaling.
        s = self.to_float(range=(-100, 100)) if isinstance(self, Truth) else self
        return _promote_and_call(other, s, "focus")

    # abstract methods:  _get_domain, _get_numerical, t.  Can I do anything here?


class LogicOperator(Operator, ABC):
    def _prepare_operand(self, a: TruthOperand, **kwargs) -> FuzzyNumber:
        """If an operand is a crisp number, it is "promoted" to an :class:`.Truthy`.  If it is already
        a :class:`.FuzzyNumber`, it will get its rare operator attributes set, as appropriate.
        """
        if isinstance(a, (int, float, bool, Truth)):  # A :class:`.Truth` is a float, so they are covered.
            a = Truthy(a)  # It's a Literal, so it doesn't have operator attributes.
        else:  # It's an operator, so it does have operator attributes.
            super()._prepare_operand(a, **kwargs)
        return a

    def _binary_logic_op(self, n: Norm, sampling_method: str, interp: Interpolator,
                         a: _Numerical, b: _Numerical) -> _Numerical:
        """general service for binary logic operators, just send it the Truth operator."""

        def op(t1: TruthOperand, t2: TruthOperand) -> TruthOperand:
            return self._op(n, t1, t2)

        r = _Numerical(self.d_op(False, a.cd, b.cd), 1, None, None, None, None, op(a.e, a.e))
        if (a.xv is not None) or (b.xv is not None):
            axv, bxv = a.xv if a.xv is not None else np.empty(0), b.xv if b.xv is not None else np.empty(0)
            r.xv = np.sort(np.unique(np.append(axv, bxv)))
            r.xt = np.empty_like(r.xv)
            r.xt = op(a.t(r.xv, interp), b.t(r.xv, interp))
        if (a.cd is not None) or (b.cd is not None):
            precision = max(a.cn, b.cn)
            points = np.empty(0)
            points = points if a.cd is None else np.append(points, np.array(a.cd))
            points = points if b.cd is None else np.append(points, np.array(b.cd))
            points = np.sort(np.unique(points))
            r.cv = np.empty(0)
            for interval in range(len(points) - 1):
                subdomain = Domain((points[interval], points[interval + 1]))
                new_v = _get_sample_points(subdomain, precision, sampling_method)
                r.cv = np.append(r.cv, new_v)
            approx_res_left = r.cv[1] - r.cv[0]
            approx_res_right = r.cv[-1] - r.cv[-2]
            gp = np.array(r.cv[0] - approx_res_left)
            r.cv = np.append(gp, r.cv)
            r.cv = np.append(r.cv, r.cv[-1] + approx_res_right)
            r.ct = np.empty_like(r.cv)
            r.ct = op(a._sample(r.cv, interp), b._sample(r.cv, interp))
        r.cn = 0 if r.cv is None else len(r.cv)
        r.e = op(a.e, b.e)
        return r

    def _associative_logic_op(self, precision: int, allowed_domain: Domain, n: Norm,
                              sampling_method: str, interp: Interpolator) -> _Numerical:
        """general service for associative logic operators, just send it the Truth operator."""

        def op(truths: np.ndarray) -> float:  # the op to use on arrays of truths
            return self._op(n, truths)

        d, e = [], []
        for a in self.operands:
            e.append(a.e)
            domain = a._get_domain()
            if domain is None:
                continue
            d.append(domain[0])
            d.append(domain[1])
        e = op(np.array(e))  # the final elsewhere truth
        if len(d) == 0:
            cd, cn, cv, ct = None, 0, None, None
        else:
            d = np.array(d)
            cd = Domain((np.min(d), np.max(d)))  # the total union domain
            d = np.sort(np.unique(d))  # the partition boundaries
            cv = np.empty(0)
            for i in range(0, len(d) - 1):  # iterate through the subdomains
                subdomain = Domain((d[i], d[i + 1]))
                new_cv = _get_sample_points(subdomain, precision, sampling_method)
                if new_cv is None:
                    continue
                new_cv = np.delete(new_cv, -1)  # Remove the last point to avoid duplication of inner points.
                cv = np.append(cv, new_cv)  # compile sample points
            gp_left, gp_right = 2 * cv[0] - cv[1], 2 * cd[1] - cv[-1]  # the guard points
            cv = np.insert(cv, 0, gp_left)
            cv = np.append(cv, cd[1])
            cv = np.append(cv, gp_right)
            cn = len(cv)  # the total number of sample points
            ct = np.empty([len(self.operands), cn])  # to hold array of truths (operand, sample point)
        xv = np.empty(0)  # to hold exceptional point values
        for i, a in enumerate(self.operands):
            num = a._get_numerical(precision, allowed_domain)  # a numerical version of each
            if ct is not None:
                ct[i] = num._sample(cv, interp)  # matrix: ct(operand, cv)
            if num.xv is None:
                continue
            xv = np.append(xv, num.xv)  # compile their exception point values
        if ct is not None:
            for j in range(cn):
                ct[0][j] = op(ct[:, j])  # op together all the sample points at cv for each operand
            ct = ct[0]  # now the (cv, ct) define a continuous function t(v)
        xv = np.sort(np.unique(xv))
        xt = np.empty([len(self.operands), len(xv)])  # to hold array of truths (operand, exceptional point value)
        for i, a in enumerate(self.operands):
            xt[i] = a.t(xv)  # evaluating with .t includes exceptionals, continuous and elsewhere
        for j in range(len(xv)):
            xt[0][j] = op(xt[:, j])  # op together all the sample points at each xv for each operand
        xt = xt[0]  # now the (cv, ct) define a complete set of exceptional points
        if len(xv) == 0:
            xv, xt = None, None
        return _Numerical(cd, cn, cv, ct, xv, xt, e)


class MathOperator(Operator, ABC):
    def _prepare_operand(self, a: MathOperand, **kwargs) -> FuzzyNumber:
        """If an operand is a crisp number, it is "promoted" to an :class:`.Exactly`.  If it is already
        a :class:`.FuzzyNumber`, it will get its rare operator attributes set, as appropriate.
        """
        if isinstance(a, (int, float, bool)):
            a = Exactly(a)  # It's a Literal, so it doesn't have operator attributes.
        else:  # It's an operator, so it does have operator attributes.
            super()._prepare_operand(a, **kwargs)
        return a

    # form Cartesian product AND, calc lines, OR-integrate, resample?

    def _binary_math_op(self, n: Norm, sampling_method: str, interp: Interpolator,
                        a: _Numerical, b: _Numerical, allowed_domain: Domain = None) -> _Numerical:
        """general service for binary logic operators"""

        # self._op(x, y) is simply the arithmetic operator, needed to find the result's xv.
        # self.none_op(a.xv, b.xv, a.xt, b.xt) handles operands without xv, xt
        # self._line(n, t1, t2) describes result lines, but how to call it?
        r = _Numerical(self.d_op(False, a.cd, b.cd), 1, None, None, None, None, n.and_(a.e, b.e))  # The result to be built.
        # cn, cv, ct;   xv, xt need attention.  First xv, xt:
        use_r, xv, xt = self.none_op(a.xv, b.xv, a.xt, b.xt)    # self needs none_op to deal with empty sets...
        if use_r:
            r.xv, r.xt = xv, xt     # ...because each operator may have to do different things with them.
        else:       # But, if both operands have exceptional points, find the resulting ones here:
            x, y = np.meshgrid(a.xv, b.xv)  # every combination of values, one from a, one from b
            xabv = self._op(x, y)  # a *op* b for each of those.
            x, y = np.meshgrid(a.xt, b.xt)  # every combination of truths, one from a, one from b
            xabt = np.ndarray.flatten(n.and_(x, y))  # a.truth & b.truth for each of those, flat.
            r.xv, i = np.unique(xabv, return_inverse=True)  # all the unique result values and where each occurs
            rlen = len(xv)  # np.max(i) + 1           # how many there are
            r.xt = np.ndarray((rlen,))  # an array to hold truths for them
            for j in range(0, rlen):  # for each
                k = np.atleast_1d(np.where(i == j))[0]  # indices where its truths are in xabt
                r.xt[j] = n.or_(xabt[k])  # or them all together for that value's truth
        # That settles the new xv, xt.  Now: products of one operand's xp with the other's continuous function.
        x_c_products = []
        if (a.xv is not None) and (b.cv is not None):
            for i in range(len(a.xv)):
                cd = self.d_op(False, Domain((a.xv[i], a.xv[i])), b.cd)
                cv = self._op(a.xv[i], b.cv)
                ct = n.and_(a.xt[i], b.ct)
                x_c_products.append(_Numerical(cd, b.cn, cv, ct, None, None))
        if (a.cv is not None) and (b.xv is not None):
            for i in range(len(b.xv)):
                cd = self.d_op(False, a.cd, Domain((b.xv[i], b.xv[i])))
                cv = self._op(a.cv, b.xv[i])
                ct = n.and_(a.ct, b.xt[i])
                x_c_products.append(_Numerical(cd, a.cn, cv, ct, None, None))
        # Now to find the product of the operands' continuous functions:  cd, cn, cv, ct
        use_r, cv, ct = self.none_op(a.cv, b.cv, a.ct, b.ct)  # self needs none_op to deal with empty sets...
        if not use_r:  # If both operands have continuous parts:
            x, y = np.meshgrid(a.ct, b.ct, indexing = 'ij')  # every combination of truths, one from a, one from b
            cart = n.and_(x, y)  # a.truth & b.truth for each of those: AND of the Cartesian product.
            rgi = RegularGridInterpolator((a.cv, b.cv), cart, bounds_error=False, fill_value=0, method="cubic")
            X, Y = np.meshgrid(a.cv, b.cv, indexing = 'ij')
            rv_all = np.unique(self._op(X, Y))   # Concentrates samples where there is structure.
            if allowed_domain is not None:
                r0 = np.argmax(rv_all >= allowed_domain[0]) - 1
                r1 = np.argmax(rv_all > allowed_domain[1]) + 1
                rv_all = rv_all[r0:r1]
            want = max(a.cn, b.cn)   # But this set must be decimated lest sample sets balloon.
            step = (len(rv_all)-1) / float(want - 1)
            indices = [int(round(x * step)) for x in range(want)]    # The decimation is as even as possible.
            cv = np.unique(np.take(rv_all, indices, mode='clip'))  # The sample points to describe the result.
            ct = np.empty_like(cv)
            for i, v in enumerate(cv):
                x, y, arclen = self._line(a.cv[0], a.cv[-1], b.cv[0], b.cv[-1], v)
                t_at_each_xy = Truth.clip(rgi((x, y)))
                ct[i] = n._or_integral(t_at_each_xy, arclen)
        cn = 0 if cv is None else len(cv)
        cd = self.d_op(False, a.cd, b.cd).intersection(allowed_domain)
        # Append that to x_c_products, or all the continuous functions together, and put them in the result.
        x_c_products.append(_Numerical(cd, cn, cv, ct, None, None))
        if x_c_products is not None:
            r_partial = (Operator.or_(*x_c_products))._get_numerical(cn)
            r.cd, r.cn, r.cv, r.ct = r_partial.cd, r_partial.cn, r_partial.cv, r_partial.ct
        return r


# UnaryOperator:--- Not, Negative, Reciprocal, Absolute

class UnaryOperator(Operator, ABC):
    """A base class: An :class:`.Operator` has one :class:`.TruthOperand` operand."""

    def __init__(self, a: Operand, **kwargs):
        """It just gets its rare attributes set and prepares its operand."""
        super().__init__(a, **kwargs)
        self.operands = [a]
        super()._prepare_operand(a, **kwargs)

    def _get_domain(self) -> Union[Domain, None]:
        """What the domain of the result will be: the domain of the operand, transformed."""
        a_d = self.operands[0]._get_domain()
        a_d = self.d_op(False, a_d)
        return a_d

    def _operate(self, precision: int, allowed_domain: Domain = None) -> _Numerical:
        """Use the operator's _op on the three parts of the operand's numerical form."""

        if isinstance(self, LogicOperator):
            a = self.operands[0]._get_numerical(precision, allowed_domain)
            n = getattr(self, 'norm', default_norm)
            a.ct = None if a.ct is None else self._op(n, a.ct)
            a.xt = None if a.xt is None else self._op(n, a.xt)
            a.e = self._op(n, a.e)
            return a
        # It must be a math operator, for which domain restrictions are more complicated.
        natural_domain = self.operands[0]._get_domain()
        imposed_domain = self.d_op(True, allowed_domain).intersection(natural_domain)
        a = self.operands[0]._get_numerical(precision, imposed_domain)
        if isinstance(self, Absolute):
            n = getattr(self, 'norm', default_norm)
            interp = getattr(self, 'interpolator', default_interpolator)
            return self._op(n, interp, a)
        if isinstance(self, MathOperator):
            a.cd = None if a.cd is None else self.d_op(False, a.cd)
            a.cv = None if a.cv is None else self._op(a.cv)
            a.xv = None if a.xv is None else self._op(a.xv)
            return a

    def t(self, v: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """"""
        if isinstance(self, LogicOperator):
            n = getattr(self, 'norm', default_norm)
            return self._op(n, self.operands[0].t(v))
        if isinstance(self, MathOperator):
            return self.operands[0].t(self._op(v))


class Not(LogicOperator, UnaryOperator):
    def __str__(self):
        a = self.operands[0].__str__()
        return str(f"NOT ({a})")

    def d_op(self, inv: bool, *d: Domain) -> Domain:
        """It only works on one domain, but Pycharm complains if I leave off the star.
        Pycharm doesn't complain when l_op or m_op do the same thing."""
        return d[0]

    def _op(self, n: Norm, t: TruthOperand) -> TruthOperand:
        return n.not_(t)


class Negative(MathOperator, UnaryOperator):
    def __str__(self):
        a = self.operands[0].__str__()
        return str(f"NEGATIVE ({a})")

    def d_op(self, inv: bool, *d: Domain) -> Domain:
        return Domain.sort(-d[0][0], -d[0][1])

    def _op(self, v: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return -v


class Reciprocal(MathOperator, UnaryOperator):
    def __str__(self):
        a = self.operands[0].__str__()
        return str(f"RECIPROCAL ({a})")

    def d_op(self, inv: bool, *d: Domain) -> Domain:
        d = d[0]  # Only ever one, but the call needs multiples.
        d0 = safe_div(1, d[0])
        d1 = safe_div(1, d[1])
        return Domain.sort(d0, d1)

    def _op(self, v: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return safe_div(1, v)


class Absolute(MathOperator, UnaryOperator):
    def __str__(self):
        a = self.operands[0].__str__()
        return str(f"ABS ({a})")

    def d_op(self, inv: bool, *d: Domain) -> Domain:
        d = d[0]  # Only ever one, but the call needs multiples.
        d0, d1 = abs(d[0]), abs(d[1])
        lo, hi = min(d0, d1), max(d0, d1)
        if (d[0] * d[1]) < 0:
            lo = 0
        return Domain((lo, hi))

    def _op(self, n: Norm, interp: Interpolator, a: _Numerical) -> _Numerical:
        """t and this return different numbers for v < 0.
        This is because I can't represent a double-sided "elsewhere".
        This is also how I handle Inequality and Sigmoid."""
        r = _Numerical(self.d_op(False, a.cd), a.cn, None, None, None, None, n.or_(a.e, a.e))
        if a.xv is not None:
            r.xv = np.sort(np.unique(np.fabs(a.xv)))
            r.xt = np.empty_like(r.xv)
            r.xt = n.or_(a.t(r.xv), a.t(-r.xv))
        if a.cd is not None:
            r.cv = np.sort(np.unique(np.fabs(a.cv)))
            r.ct = np.empty_like(r.cv)
            r.ct = n.or_(a._sample(r.cv, interp), a._sample(-r.cv, interp))
        return r

    def t(self, v: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """The result at v could come from the truths at v or -v, so I OR those truths together."""
        operand = self.operands[0]
        a = isinstance(v, np.ndarray)
        v = np.atleast_1d(v)
        n = getattr(self, 'norm', default_norm)
        r = n.or_(operand.t(v), operand.t(-v))
        less_that_zero = np.where(v < 0)
        r[less_that_zero] = 0
        return r if a else r[0]


# BinaryOperator:---  logic: imp, con, nimp, ncon, nand, nor, xor, iff; math: mul, div

class BinaryOperator(Operator, ABC):
    """A base class: An :class:`.Operator` has one :class:`.TruthOperand` operand."""

    def __init__(self, a: Operand, b: Operand, **kwargs):
        """It just gets its rare attributes set and prepares its operand."""
        super()._prepare_operand(a, **kwargs)
        super()._prepare_operand(b, **kwargs)
        super().__init__(a, b, **kwargs)

    def __str__(self):
        a = self.operands[0].__str__()
        b = self.operands[1].__str__()
        return str(f" ({a})\n {self.name} \n ({b})")

    def _get_domain(self) -> Union[Domain, None]:
        """What the domain of the result will be: the domain of the operand, transformed."""
        a_d = self.operands[0]._get_domain()
        b_d = self.operands[1]._get_domain()
        d = self.d_op(False, a_d, b_d)
        return d

    def d_op(self, inv: bool, a: Domain, b: Domain) -> Domain:  # noqa
        """It only works on one domain, but Pycharm complains if I leave off the star.
        Pycharm doesn't complain when l_op or m_op do the same thing.

        This works for the 8 binary logic ops, so I'll put it here.  The 2 binary math ops are different."""
        if a is None:
            return b
        else:
            return a.union(b)

    def t(self, v: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """This should do it for all binary functions"""
        if isinstance(self, LogicOperator):
            n = getattr(self, 'norm', default_norm)
            return self._op(n, self.operands[0].t(v), self.operands[1].t(v))
        if isinstance(self, MathOperator):
            n = getattr(self, 'norm', default_norm)
            return self._op(n, self.operands[0].t(v), self.operands[1].t(v))  # TODO Something like this? combine?

    def _operate(self, precision: int, allowed_domain: Domain = None) -> _Numerical:
        """Use the operator's d_op and l_op or m_op on the three parts of the operand's numerical form."""
        n = getattr(self, 'norm', default_norm)
        sampling = getattr(self, 'sampling', default_sampling_method)
        interp = getattr(self, 'interpolator', default_interpolator)
        if isinstance(self, LogicOperator):
            a = self.operands[0]._get_numerical(precision, allowed_domain)
            b = self.operands[1]._get_numerical(precision, allowed_domain)
            return self._binary_logic_op(n, sampling, interp, a, b)  # does the partitioning and sampling behavior
        else:  # It's a MathOperator.
            a_natural_domain = self.operands[0]._get_domain()
            b_natural_domain = self.operands[1]._get_domain()
            a_imposed_domain, b_imposed_domain = self.inv_d_op(allowed_domain, a_natural_domain, b_natural_domain)
            a = self.operands[0]._get_numerical(precision, a_imposed_domain)
            b = self.operands[1]._get_numerical(precision, b_imposed_domain)
            return self._binary_math_op(n, sampling, interp, a, b) # TODO impose domain on a, b, r here?


class Imp(LogicOperator, BinaryOperator):
    name = str("→ IMPLIES →")

    def _op(self, n: Norm, a: TruthOperand, b: TruthOperand) -> TruthOperand:
        return Truth.imp(a, b, norm=n)


class Con(LogicOperator, BinaryOperator):
    name = str("← CONVERSE IMPLIES ←")

    def _op(self, n: Norm, a: TruthOperand, b: TruthOperand) -> TruthOperand:
        return Truth.con(a, b, norm=n)


class Iff(LogicOperator, BinaryOperator):
    name = str("↔ IF AND ONLY IF ↔")

    def _op(self, n: Norm, a: TruthOperand, b: TruthOperand) -> TruthOperand:
        return Truth.iff(a, b, norm=n)


class Xor(LogicOperator, BinaryOperator):
    name = str("⨁ EXCLUSIVE OR ⨁")

    def _op(self, n: Norm, a: TruthOperand, b: TruthOperand) -> TruthOperand:
        return Truth.xor(a, b, norm=n)


class Nand(LogicOperator, BinaryOperator):
    name = str("↑ NAND ↑")

    def _op(self, n: Norm, a: TruthOperand, b: TruthOperand) -> TruthOperand:
        return Truth.nand(a, b, norm=n)


class Nor(LogicOperator, BinaryOperator):
    name = str("↓ NOR ↓")

    def _op(self, n: Norm, a: TruthOperand, b: TruthOperand) -> TruthOperand:
        return Truth.nor(a, b, norm=n)


class Nimp(LogicOperator, BinaryOperator):
    name = str("↛ NON-IMPLICATION ↛")

    def _op(self, n: Norm, a: TruthOperand, b: TruthOperand) -> TruthOperand:
        return Truth.nimp(a, b, norm=n)


class Ncon(LogicOperator, BinaryOperator):
    name = str("↚ CONVERSE NON-IMPLICATION ↚")

    def _op(self, n: Norm, a: TruthOperand, b: TruthOperand) -> TruthOperand:
        return Truth.ncon(a, b, norm=n)


class AssociativeOperator(Operator, ABC):
    """A base class: An :class:`.Operator` has one :class:`.TruthOperand` operand."""

    def __init__(self, *operands: Operand, **kwargs):
        """It just gets its rare attributes set and prepares its operand."""
        for a in operands:
            super()._prepare_operand(a, **kwargs)
        super().__init__(*operands, **kwargs)

    def __str__(self):
        top = str(f"\n {self.name} together the following: [\n")
        for i, a in enumerate(self.operands):
            top = top + str(f"{i}:  ") + a.__str__() + str(f"\n")
        top = top + str(f"]\n")
        return top

    def _get_domain(self) -> Union[Domain, None]:
        """What the domain of the result will be: the domain of the operand, transformed."""
        d, i, n = None, 0, len(self.operands)
        for i in range(0, n):  # Find the first operand that has a domain.
            d = self.operands[i]._get_domain()
            if d is not None:
                break
        if d is None:
            return None  # If none do, return None.
        for j in range(i + 1, n):
            a = self.operands[j]._get_domain()
            d = self.d_op(False, d, a)
        return d

    def d_op(self, inv: bool, *a: Domain) -> Domain:
        """This works for the 2 associative logic ops, so I'll put it here.
        The 2 associative math ops are different, and must override this."""
        return a[0].union(a[1])

    def t(self, v: float) -> float:
        """This should do it for all associative functions"""
        if isinstance(self, LogicOperator):
            n = getattr(self, 'norm', default_norm)
            t = []
            for a in self.operands:
                t.append(a.t(v))
            t = np.array(t)
            return self._op(n, t)
        if isinstance(self, MathOperator):  # TODO
            norm = getattr(self, 'norm', default_norm)
            s, n = self.operands[0], len(operands)
            for i in range(1, n):
                a = self.operands[i]
                s = self._op(norm, s, a)
            return s

    def _operate(self, precision: int, allowed_domain: Domain = None) -> _Numerical:
        n = getattr(self, 'norm', default_norm)
        sampling = getattr(self, 'sampling', default_sampling_method)
        interp = getattr(self, 'interpolator', default_interpolator)
        if isinstance(self, LogicOperator):
            return self._associative_logic_op(precision, allowed_domain, n, sampling, interp)
            # does the partitioning and sampling behavior
        else:  # It's a MathOperator---but this simple routine would work for logical associatives too:
            # TODO re-rwrite this as a chain of binaries
            # TODO this works, but go over it.
            number = len(self.operands)
            a = self.operands[0]
            for i in range(1, number):
                b = self.operands[i]
                a_natural_domain = a._get_domain()
                b_natural_domain = b._get_domain()
                r_natural_domain = self.d_op(False, a_natural_domain, b_natural_domain)
                allowed_domain = allowed_domain.intersection(r_natural_domain)
                if allowed_domain is None:
                    a_imposed_domain, b_imposed_domain = a_natural_domain, b_natural_domain
                else:
                    a_imposed_domain, b_imposed_domain = self.inv_d_op(allowed_domain, a_natural_domain, b_natural_domain)
                a = a._get_numerical(precision, a_imposed_domain)
                b = b._get_numerical(precision, b_imposed_domain)
                a = self._binary_math_op(n, sampling, interp, a, b, allowed_domain)
                a = _Numerical._impose_domain(a, allowed_domain)
            return a


class And(LogicOperator, AssociativeOperator):
    name = str("∧ AND ∧")

    def _op(self, n: Norm, *operands: TruthOperand) -> TruthOperand:
        return n.and_(*operands)


class Or(LogicOperator, AssociativeOperator):
    name = str("∨ OR ∨")

    def _op(self, n: Norm, *operands: TruthOperand) -> TruthOperand:
        return n.or_(*operands)


# All the math operators are described as binary, even if they are associative, because I must do the associative
# ones as chains of binaries because the continuous part is so involved.

# r = a+b:    r - a = b   parallel NW to SE lines, slope = -1
# r = a-b:    a - r = b   parallel SW to NE lines, slope = 1
# r = a*b:    r / a = b   +ve r: -\ + \_; -ve r: _/ + /- farther from origin as r increases
# r = a/b:    a / r = b   straight line through origin, slope = r

class Add(MathOperator, AssociativeOperator):
    name = str("+ ADD +")

    def d_op(self, inv: bool, a: Domain, b: Domain) -> Domain:  # noqa
        """The natural resulting domain when a + b."""
        if a is None:
            return b
        if b is None:
            return a
        else:
            return Domain((a[0] + b[0], a[1] + b[1]))

    def inv_d_op(self, r: Domain, a: Domain, b: Domain) -> (Domain, Domain, float):  # noqa
        """The extreme domains of a and b that could lead to that of r.
        r is imposed, is in terms of this operator's result."""
        # I know:  ai[0] + bi[0] = r[0], ai[1] + bi[1] = r[1]
        ai = Domain((max(a[0], r[0] - b[1]), min(a[1], r[1] - b[0])))
        bi = Domain((max(b[0], r[0] - a[1]), min(b[1], r[1] - a[0])))
        return ai, bi

    def none_op(self, axv: np.ndarray, bxv: np.ndarray, axt: np.ndarray, bxt: np.ndarray) \
            -> Union[Tuple[bool, np.ndarray, np.ndarray],  Tuple[bool, None, None]]:
        """Handles cases where one or both operands have no exceptional points."""
        if axv is None:
            return True, bxv, bxt
        if bxv is None:
            return True, axv, axt
        else:
            return False, None, None

    def _op(self, a: float, b: float) -> float:
        # add operands (for pairs of xv)
        # return b if (a is None) else a if (b is None) else a + b
        return a + b

    def _line(self, xmin: float, xmax: float, ymin: float, ymax: float,  r: float) -> (np.ndarray, np.ndarray, float):
        # Figure out what happens in here.  describe the lines on the cartprod. MathOperator._binary_math_op uses this.
        xmin_r = max(xmin, r - ymax)
        xmax_r = min(xmax, r - ymin)
        arclen = sqrt(2 * (xmax_r - xmin_r) ** 2)
        max_arclen = sqrt(2 * (xmax - xmin) ** 2)
        n = max(3, floor(100 * arclen / max_arclen))
        x = np.linspace(xmin_r, xmax_r, n)   # The number doesn't matter much, but shouldn't it depend on arclen???
        y = r - x
        return x, y, arclen

# AssociativeOperator:---  logic: and, or;  math:  add, mul
# Qualifiers:  normalize, weight, focus
# FuzzyExpression---with all the static methods;  overloaded operators

# class Op(Operator):
#
#
#
