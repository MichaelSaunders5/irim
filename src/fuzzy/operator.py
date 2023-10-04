"""Come on, let's *do* somthin'."""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import pi  # ceil, floor, pi, is finite, log, sqrt
from sys import float_info
from typing import Union, Tuple  # , ClassVar,

import numpy as np

import fuzzy.norm           # ???
from fuzzy.norm import Norm, default_norm
from fuzzy.crisp import Interpolator
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
        # r = x / y
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


def _promote_and_call(s, b, logic, fn_name, flip):
    if flip:
        s, b = b, s
    if isinstance(b, Truth):  # If it is a Truth, make it a Truthy
        from fuzzy.literal import Truthy
        b = Truthy(b)
    elif isinstance(b, (float, int, bool)):  # If it is a number, make it a FuzzyNumber
        b = float(b)
        if logic and b>=0 and b<=1:         # If it's logical and in range, a Truthy, otherwise, Exactly
            from fuzzy.literal import Truthy
            b = Truthy(b)
        else:
            from fuzzy.literal import Exactly
            b = Exactly(b)
    if flip:
        s, b = b, s
    from fuzzy.operator import Operator     # Now both are FuzzyNumber, so prepare to call Operator
    return eval("Operator." + fn_name + "(s, b)")   # Call the operator for FuzzyNumbers


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
        if allowed_domain is not None:
            allowed_domain = self.d_op(True, allowed_domain)  # Only sample where it matters.
        return self._operate(precision, allowed_domain)


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
        """

    # Here is where all the operator functions used in expressions go:

    def not_(self, **kwargs) -> Operator:
        """Returns a :class:`.Not` object with ``self`` as the operand (for logical negation, ¬a).
        For ``kwargs`` see :meth:`.fuzzy_ctrl`.

        Call by:

        * ``a.not_()``
        * ``Operator.not_(b)``
        * ``~a``

        Where

        * ``a`` is a :class:`.FuzzyNumber` (e.g., an :class:`.Operator` or :class:`.Literal`); and
        * ``b`` is a :class:`.FuzzyNumber`, or ``float`` or ``int`` or ``bool``.
        """
        return Not(self, **kwargs)
    def neg(self, **kwargs) -> Operator:
        """Returns a :class:`.Negative` object with ``self`` as the operand (for arithmetic negation, -a).
        For ``kwargs`` see :meth:`.fuzzy_ctrl`.

        Call by:

        * ``a.neg()``
        * ``Operator.neg(b)``
        * ``-a``

        Where

        * ``a`` is a :class:`.FuzzyNumber` (e.g., an :class:`.Operator` or :class:`.Literal`); and
        * ``b`` is a :class:`.FuzzyNumber`, or ``float`` or ``int`` or ``bool``.
        """
        return Negative(self, **kwargs)
    def reciprocal(self, **kwargs) -> Operator:
        """Returns a :class:`.Reciprocal` object with ``self`` as the operand (for 1/a).
        For ``kwargs`` see :meth:`.fuzzy_ctrl`.

        Call by:

        * ``a.reciprocal()``
        * ``Operator.reciprocal(b)``

        Where

        * ``a`` is a :class:`.FuzzyNumber` (e.g., an :class:`.Operator` or :class:`.Literal`); and
        * ``b`` is a :class:`.FuzzyNumber`, or ``float`` or ``int`` or ``bool``.
        """
        return Reciprocal(self, **kwargs)
    def abs(self, **kwargs) -> Operator:
        """Returns a :class:`.Absolute` object with ``self`` as the operand (for absolute value, :math:`|a|`).
        For ``kwargs`` see :meth:`.fuzzy_ctrl`.

        Call by:

        * ``a.abs()``
        * ``Operator.abs(b)``
        * ``+a``

        Where

        * ``a`` is a :class:`.FuzzyNumber` (e.g., an :class:`.Operator` or :class:`.Literal`); and
        * ``b`` is a :class:`.FuzzyNumber`, or ``float`` or ``int`` or ``bool``.
        """
        return Absolute(self, **kwargs)


    def imp(self, b: Operand, **kwargs) -> Operator:
        """Returns an :class:`.And` object with ``self`` and ``b`` as operands (for self → b).
        For ``kwargs`` see :meth:`.fuzzy_ctrl`.

        Call by:

        * ``a.imp()``
        * ``Operator.imp(a, b)``
        * ``a >> b``

        Where

        * ``self``, or ``a`` is a :class:`.FuzzyNumber` (e.g., an :class:`.Operator` or :class:`.Literal`); and
        * ``b`` is a :class:`.FuzzyNumber`, or :class:`.Truth`, or ``float`` or ``int`` or ``bool``.
        """
        return Imp(self, b, **kwargs)
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


    def __and__(self, other: Truth) -> Truth:
        """Overloads the binary ``&`` (bitwise and) operator."""
        return Truth.and_(self, Truth(float(other), clip=True))

    def __rand__(self, other: Truth) -> Truth:
        """Ensures that the overloading of ``&`` works as long as one operand is a ``Truth`` object."""
        return Truth.and_(Truth(float(other), clip=True), self)

    def __or__(self, other: Truth) -> Truth:
        """Overloads the binary ``|`` (bitwise or) operator."""
        return Operator.or_(self, _promote_operand(True, other))

    def __ror__(self, other: Truth) -> Truth:
        """Ensures that the overloading of ``|`` works as long as one operand is a ``Truth`` object."""
        return Operator.or_(_promote_operand(True, other), self)

    def __rshift__(self, other: Truth) -> Truth:
        """Overloads the binary ``>>`` (bitwise right shift) operator."""
        # return Operator.imp(self, _promote_operand(True, other))
        return _promote_and_call(self, other, True, "imp", False)

    def __rrshift__(self, other: Truth) -> Truth:
        """Ensures that the overloading of ``>>`` works as long as one operand is a ``Truth`` object."""
        # return Operator.imp(_promote_operand(True, other), self)
        return _promote_and_call(other, self, True, "imp", True)

    def __lshift__(self, other: Truth) -> Truth:
        """Overloads the binary ``<<`` (bitwise left shift) operator."""
        return Truth.con(self, Truth(float(other), clip=True))

    def __rlshift__(self, other: Truth) -> Truth:
        """Ensures that the overloading of ``<<`` works as long as one operand is a ``Truth`` object."""
        return Truth.con(Truth(float(other), clip=True), self)

    def __matmul__(self, other: Truth) -> Truth:
        """Overloads the binary ``@`` (matrix product) operator."""
        return Truth.xor(self, Truth(float(other), clip=True))

    def __rmatmul__(self, other: Truth) -> Truth:
        """Ensures that the overloading of ``@`` works as long as one operand is a ``Truth`` object."""
        return Truth.xor(Truth(float(other), clip=True), self)

    def __floordiv__(self, other: Union[Truth | float]) -> Truth:
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

    def __rfloordiv__(self, other: float) -> Truth:
        """Ensures that the overloading of ``^`` works as long as one operand is a ``Truth`` object."""
        # This gets sent to the magic method version to deal with the truth scaling.
        s = self.to_float(range=(-100, 100)) if isinstance(self, Truth) else self
        return Truth.weight(Truth(float(other), clip=True), s)

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
        return Truth.weight(self, other)

    def __rxor__(self, other: float) -> Truth:
        """Ensures that the overloading of ``^`` works as long as one operand is a ``Truth`` object."""
        # This gets sent to the magic method version to deal with the truth scaling.
        if isinstance(self, Truth):
            self = self.to_float(range=(-100, 100))
        return Truth.weight(Truth(float(other), clip=True), self)

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
        d = np.array(d)
        cd = Domain((np.min(d), np.max(d)))  # the total union domian
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
            ct[i] = num._sample(cv, interp)  # matrix: ct(operand, cv)
            if num.xv is None:
                continue
            xv = np.append(xv, num.xv)  # compile their exception point values
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
        """Use the operator's d_op and l_op or m_op on the three parts of the operand's numerical form."""
        a = self.operands[0]._get_numerical(precision, allowed_domain)  # a Numerical version of the operand
        if isinstance(self, LogicOperator):
            n = getattr(self, 'norm', default_norm)
            a.ct = None if a.ct is None else self._op(n, a.ct)
            a.xt = None if a.xt is None else self._op(n, a.xt)
            a.e = self._op(n, a.e)
            return a
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
        a = self.operands[0]._get_numerical(precision, allowed_domain)  # a Numerical version of the a operand
        b = self.operands[1]._get_numerical(precision, allowed_domain)  # a Numerical version of the b operand
        if isinstance(self, LogicOperator):
            return self._binary_logic_op(n, sampling, interp, a, b)  # does the partitioning and sampling behavior
        else:  # It's a MathOperator.
            return self._binary_math_op(n, sampling, interp, a, b)  # TODO  defined by its line functions


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
            d = self.operands[0]._get_domain()
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
            number = len(self.operands)
            s = self.operands[0]._get_numerical(precision, allowed_domain)
            for i in range(1, number):
                a = self.operands[i]._get_numerical(precision, allowed_domain)
                s = self._op(n, s, a)  # sampling, interp,
            return s


class And(LogicOperator, AssociativeOperator):
    name = str("∧ AND ∧")

    def _op(self, n: Norm, *operands: TruthOperand) -> TruthOperand:
        return n.and_(*operands)


class Or(LogicOperator, AssociativeOperator):
    name = str("∨ OR ∨")

    def _op(self, n: Norm, *operands: TruthOperand) -> TruthOperand:
        return n.or_(*operands)

# AssociativeOperator:---  logic: and, or;  math:  add, mul
# Qualifiers:  normalize, weight, focus
# FuzzyExpression---with all the static methods;  overloaded operators

# class Op(Operator):
#
#
#
