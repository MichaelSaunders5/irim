from __future__ import annotations

from abc import ABC, abstractmethod
from math import pi  # ceil, floor, pi, is finite, log, sqrt
from sys import float_info
from typing import Union, Tuple  # , ClassVar,

import numpy as np

import fuzzy.norm
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
    individual :class:`.Operator`\\ s.  See. :meth:`.fuzzy_ctrl`."""
    norm, threshold, sampling, interpolator, resolution, crisper = None, None, None, None, None, None
    norm_args = kwargs.get('norm', None)
    if norm_args is not None:
        norm = Norm.define(**norm_args)
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
    r = np.nan_to_num(r, nan=0, posinf=float_info.max, neginf=-float_info.max)
    return r


def _get_sample_points(cd: Domain, cn: int, sampling_method: str) -> np.ndarray:
    if sampling_method == "Chebyshev":  # Find the values to sample at, Chebyshev or uniform.
        m = cn + 4
        cv = np.arange(2, m - 1, 1)
        cv = .5 * cd.span() * (-np.cos((cv - .5) * pi / m)) / np.cos(1.5 * pi / m) + cd.center()
    else:
        cv = np.linspace(cd[0], cd[1], num=(cn + 2))
    return cv


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

    If you set an attribute of an operator via ``**kwargs, it sets the attribute in itself, its operands,
    their operands, and so on---its whole sub-tree---except where these have already been explicitly set.

    How do you make an operator?  I think by just:
    1. implement __str__
    2. implement d_op (if the superclass doesn't do the right thing with it).
    3. define either:
        a. l_op if it's a logic operator, or
        b. m_op if it's a math operator
    4. implement t (can I automate that too?)

    The unary/binary/associative classes should handle the handling of 1,2,many operands in their get_numericals
    I think that's it.
        """

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

    def _prepare_operand(self, a: Operand, **kwargs) -> Operand:
        """Operators get their rare attributes set here (or they will keep those given,
        or they will find the global defaults)."""
        if isinstance(a, Operator):
            a._set_attributes(**kwargs)
        return a

    @abstractmethod
    def _op(self, *args) -> _Numerical:
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
        operator not to sample where it won't matter to us, so it does ``d_op(a, True) to find that this means [-5,5].
        Its business is then on the intersection of that and what it has: [-5,5].intersection[0,10], so it only
        samples [0,5] of its operand to create the numerical it will return.
         """

    # def t(self, v: float) -> float:
    #     t = []
    #     for a in self.operands:
    #         t.append(a.t(v))
    #     # Then do what _operate does?

    # abstract methods:  _get_domain, _get_numerical, t.  Can I do anything here?


class LogicOperator(Operator, ABC):
    def _prepare_operand(self, a: TruthOperand, **kwargs) -> FuzzyNumber:
        """If an operand is a crisp number, it is "promoted" to an :class:`.Truthy`.  If it is already
        a :class:`.FuzzyNumber`, it will get its rare operator attributes set, as appropriate.
        """
        if isinstance(a, (int, float, bool, Truth)):  # A :class:`.Truth` is a float, so they are covered.
            a = Truthy(a)  # It's a Literal, so it doesn't have operator attributes.
        else:  # It's an operator, so it does have operator attributes.
            a = super()._prepare_operand(a, **kwargs)
        return a

    def _binary_logic_op(self, n: Norm, sampling_method: str, interp: Interpolator,
                         a: _Numerical, b: _Numerical) -> _Numerical:
        """general service for binary logic operators, just send it the Truth operator."""

        # sorry.  To complicated to type hint. see below
        # So, it's going to get str code letter + four arrays, like: na.xv, na.xt, nb.xv, nb.xt, "x"|"c"
        # or, str code letter + two floats, like: na.e, nb.e, "e"
        def op(t1: TruthOperand, t2: TruthOperand) -> TruthOperand:
            return self._op(n, t1, t2)

        r = _Numerical(self.d_op(False, a.cd, b.cd), 1, None, None, None, None, op(a.e, a.e))
        print(f"r.cd:  {r.cd}")
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
        r.cn = len(r.cv)
        return r

    def ass_l_op(self, situation: str, precision: int, operator: Callable, *operand):
        """general service for associative logic operators, just send it the ``operator``."""
        # sorry.  To complicated to type hint. see below
        # So, it's going to get str code letter + op + N 4-tuples of arrays, like: (na.cv, na.ct, na.xv, na.xt) "x|c"
        # or, str code letter + op + N  floats, like: na.e, nb.e, ... "e"
        n = getattr(self, 'norm', default_norm)
        a, b = self.operands[0], self.operands[1]
        sampling_method = getattr(self, 'default_sampling_method', default_sampling_method)
        if situation == "x":
            av, at, bv, bt = operand[0], operand[1], operand[2], operand[3]
            rv, rt = None, None
            av, bv = np.empty(0) if av is None else av, np.empty(0) if bv is None else bv
            rt = np.empty(0) if rv is None else rv
            rv = np.unique(np.append(av, bv))
            for v in rv:
                rt = np.append(rt, operator(a.t(v), b.t(v), n))
            return rv, rt
        elif situation == "c":
            # av, at, bv, bt = operand[0], operand[1], operand[2], operand[3]
            points = np.array([a.d[0], a.d[1], b.d[0], b.d[1]])
            points = np.sort(np.unique(points))
            # sample each interval with precision pts, n.imp(a.t(v), self.b.t(v)) at those points
            # this is the same for all binary logic, so make it a helper with precision and operator arguments.
            # write a helper to choose the sample points---add guards on either end.
            rv = np.empty(0)
            for interval in range(len(points) - 1):
                subdomain = Domain((points[interval], points[interval + 1]))
                new_v = _get_sample_points(subdomain, precision, sampling_method)
                rv = np.append(rv, new_v)
            approx_res_left = rv[1] - rv[0]
            approx_res_right = rv[-1] - rv[-2]
            gp = np.array(rv[0] - approx_res_left)
            rv = np.append(gp, rv)
            rv = np.append(rv, rv[-1] + approx_res_right)
            rt = operator(a._sample(rv), b._sample(rv), n)
            return rv, rt
        else:  # elsewhere
            print(f"in ass lop now: {operand}")
            return operator(operand, n)


class MathOperator(Operator, ABC):
    def _prepare_operand(self, a: MathOperand, **kwargs) -> FuzzyNumber:
        """If an operand is a crisp number, it is "promoted" to an :class:`.Exactly`.  If it is already
        a :class:`.FuzzyNumber`, it will get its rare operator attributes set, as appropriate.
        """
        if isinstance(a, (int, float, bool)):
            a = Exactly(a)  # It's a Literal, so it doesn't have operator attributes.
        else:  # It's an operator, so it does have operator attributes.
            a = super()._prepare_operand(a, **kwargs)
        return a
    # form Cartesian product AND, calc lines, OR-integrate, resample?


# UnaryOperator:--- Not, Negative, Reciprocal, Absolute

class UnaryOperator(Operator, ABC):
    """A base class: An :class:`.Operator` has one :class:`.TruthOperand` operand."""

    def __init__(self, a: Operand, **kwargs):
        """It just gets its rare attributes set and prepares its operand."""
        super().__init__(a, **kwargs)
        a = super()._prepare_operand(a, **kwargs)
        self.operands = [a]

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
            a.ct, a.xt, a.e = self._op(n, a.ct), self._op(n, a.xt), self._op(n, a.e)
            return a
        if isinstance(self, Absolute):
            n = getattr(self, 'norm', default_norm)
            interp = getattr(self, 'interpolator', default_interpolator)
            return self._op(n, interp, a)
        if isinstance(self, MathOperator):
            a.ct, a.xt = self._op(a.ct), self._op(a.xt)
            return a


class Not(LogicOperator, UnaryOperator):
    def __str__(self):
        a = self.operands[0].__str__()
        return str(f"NOT ({a})")

    def d_op(self, inv: bool, d: Domain) -> Domain:  # noqa
        """It only works on one domain, but Pycharm complains if I leave off the star.
        Pycharm doesn't complain when l_op or m_op do the same thing."""
        return d

    def _op(self, n: Norm, t: TruthOperand) -> float:
        return n.not_(t)

    def t(self, v: float) -> float:
        """"""
        return self._op(default_norm, self.operands[0].t(v))


class Negative(MathOperator, UnaryOperator):
    def __str__(self):
        a = self.operands[0].__str__()
        return str(f"NEGATIVE ({a})")

    def d_op(self, inv: bool, *d: Domain) -> Domain:
        return Domain.sort(-d[0][0], -d[0][1])

    def _op(self, v: np.ndarray) -> np.ndarray:
        return -v

    def t(self, v: float) -> float:
        """"""
        return self.operands[0].t(-v)


class Reciprocal(MathOperator, UnaryOperator):
    def __str__(self):
        a = self.operands[0].__str__()
        return str(f"RECIPROCAL ({a})")

    def d_op(self, inv: bool, d: Domain) -> Domain:  # noqa
        d0 = safe_div(1, d[0])
        d1 = safe_div(1, d[1])
        return Domain.sort(d0, d1)

    def _op(self, v: np.ndarray) -> np.ndarray:
        return safe_div(1, v)

    def t(self, v: float) -> float:
        return self.operands[0].t(safe_div(1, v))


class Absolute(MathOperator, UnaryOperator):
    def __str__(self):
        a = self.operands[0].__str__()
        return str(f"ABS ({a})")

    def d_op(self, inv: bool, d: Domain) -> Domain:  # noqa
        d0, d1 = abs(d[0]), abs(d[1])
        lo, hi = min(d0, d1), max(d0, d1)
        if (d[0] * d[1]) < 0:
            lo = 0
        return Domain((lo, hi))

    def _op(self, n: Norm, interp: Interpolator, a: _Numerical) -> _Numerical:
        """_Numerical and this return different numbers for v < 0.
        This is because I can't represent a double-sided "elsewhere".
        This is also how I handle Inequality and Sigmoid."""
        r = _Numerical(self.d_op(False, a.cd), a.cn, None, None, None, None, n.or_(a.e, a.e))
        if a.vx is not None:
            r.xv = np.sort(np.unique(np.fabs(a.xv)))
            r.xt = np.empty_like(r.xv)
            for i in r.xv:
                v = r.xv[i]
                r.xt[i] = n.or_(a.t(v, interp), a.t(-v, interp))
        if a.cd is not None:
            r.cv = np.sort(np.unique(np.fabs(a.cv)))
            r.ct = np.empty_like(r.cv)
            for i in r.cv:
                v = r.cv[i]
                r.ct[i] = n.or_(a._sample(v, interp), a._sample(-v, interp))
        return r

    def t(self, v: float) -> float:
        """The result at v could come from the truths at v or -v, so I OR those truths together."""
        n = getattr(self, 'norm', default_norm)
        a = operands[0]
        return 0 if v < 0 else n.or_(a.t(v), a.t(-v))


# BinaryOperator:---  logic: imp, con, nimp, ncon, nand, nor, xor, iff; math: mul, div

class BinaryOperator(Operator, ABC):
    """A base class: An :class:`.Operator` has one :class:`.TruthOperand` operand."""

    def __init__(self, a: Operand, b: Operand, **kwargs):
        """It just gets its rare attributes set and prepares its operand."""
        a = super()._prepare_operand(a)
        b = super()._prepare_operand(b)
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
        return a.union(b)

    def t(self, v: float) -> float:
        """This should do it for all binary functions"""
        return self.l_op("e", 0, self.operands[0].t(v), self.operands[1].t(v))

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
            return self._binary_math_op(n, sampling, interp, a, b)  # defined by its line functions


class Imp(LogicOperator, BinaryOperator):
    name = str("→ IMPLIES →")

    def _op(self, n: Norm, a: TruthOperand, b: TruthOperand) -> TruthOperand:
        return Truth.imp(a, b, n)


class Con(LogicOperator, BinaryOperator):
    name = str("← CONVERSE IMPLIES ←")

    def _op(self, n: Norm, a: TruthOperand, b: TruthOperand) -> TruthOperand:
        return Truth.con(a, b, n)


class Iff(LogicOperator, BinaryOperator):
    name = str("↔ IF AND ONLY IF ↔")

    def _op(self, n: Norm, a: TruthOperand, b: TruthOperand) -> TruthOperand:
        return Truth.iff(a, b, n)


class Xor(LogicOperator, BinaryOperator):
    name = str("⨁ EXCLUSIVE OR ⨁")

    def _op(self, n: Norm, a: TruthOperand, b: TruthOperand) -> TruthOperand:
        return Truth.xor(a, b, n)


class Nand(LogicOperator, BinaryOperator):
    name = str("↑ NAND ↑")

    def _op(self, n: Norm, a: TruthOperand, b: TruthOperand) -> TruthOperand:
        return Truth.nand(a, b, n)


class Nor(LogicOperator, BinaryOperator):
    name = str("↓ NOR ↓")

    def _op(self, n: Norm, a: TruthOperand, b: TruthOperand) -> TruthOperand:
        return Truth.nor(a, b, n)


class Nimp(LogicOperator, BinaryOperator):
    name = str("↛ NON-IMPLICATION ↛")

    def _op(self, n: Norm, a: TruthOperand, b: TruthOperand) -> TruthOperand:
        return Truth.nimp(a, b, n)


class Ncon(LogicOperator, BinaryOperator):
    name = str("↚ CONVERSE NON-IMPLICATION ↚")

    def _op(self, n: Norm, a: TruthOperand, b: TruthOperand) -> TruthOperand:
        return Truth.ncon(a, b, n)


class BinaryOperator(Operator, ABC):
    """A base class: An :class:`.Operator` has one :class:`.TruthOperand` operand."""

    def __init__(self, a: Operand, b: Operand, **kwargs):
        """It just gets its rare attributes set and prepares its operand."""
        a = super()._prepare_operand(a, **kwargs)
        b = super()._prepare_operand(b, **kwargs)
        super().__init__(a, b, **kwargs)

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
        return a.union(b)

    def t(self, v: float) -> float:
        """This should do it for all binary functions"""
        return self.l_op("e", 0, self.operands[0].t(v), self.operands[1].t(v))

    def _operate(self, precision: int, allowed_domain: Domain = None) -> _Numerical:
        """Use the operator's d_op and l_op or m_op on the three parts of the operand's numerical form."""
        na = operands[0]._get_numerical(precision, allowed_domain)  # a Numerical version of the a operand
        nb = operands[1]._get_numerical(precision, allowed_domain)  # a Numerical version of the b operand
        e = self.l_op("e", precision, na.e, nb.e)
        # on [0,1] if l_op is defined (logic op); -1 otherwise (math op).
        ax, ac, = na.xv is not None, na.cv is not None  # What ``a`` and ``b`` have (c?, x?)
        bx, bc, = nb.xv is not None, nb.cv is not None
        logic = e >= 0  # A logic or math op?
        cd, cn, xv, xt, cv, ct = None, 0, None, None, None, None
        if (ax or bx) and logic:
            xv, xt = self.l_op("x", precision, na.xv, na.xt, nb.xv, nb.xt)
        if (ax or bx) and not logic:
            xv, xt = self.m_op("x", precision, na.xv, na.xt, nb.xv, nb.xt)
        if (ac or bc) and logic:
            cd = self.d_op(False, na.cd, nb.cd)  # What the domain of the result will be.
            cv, ct = self.l_op("c", precision, na.cv, na.ct, nb.cv, nb.ct)
        if (ac or bc) and not logic:
            cd = self.d_op(False, na.cd, nb.cd)  # What the domain of the result will be.
            cv, ct = self.m_op("c", precision, na.cv, na.ct, nb.cv, nb.ct)
        if not logic:
            e = self.m_op("e", precision, na.e, nb.e)
        return _Numerical(cd, precision, cv, ct, xv, xt, e)


class AssociativeOperator(Operator, ABC):
    """A base class: An :class:`.Operator` has one :class:`.TruthOperand` operand."""

    def __init__(self, *operands: TruthOperand, **kwargs):
        """It just gets its rare attributes set and prepares its operand."""
        super().__init__(**kwargs)  # bogus because Logic/MathOperator will call it.
        operands = list(operands)
        for i in range(0, len(operands)):
            operands[i] = super()._prepare_operand(operands[i], **kwargs)
        self.a = operands

    def __str__(self):
        out = str(f"( ({self.a[0].__str__()})")
        for i in range(1, len(self.a)):
            out = out + str(f"\n {self.name} \n ") + str(f"({self.a[i].__str__()})")
        return out + str(" )")

    def _get_domain(self) -> Union[Domain, None]:
        """What the domain of the result will be: the domain of the operand, transformed."""
        d = None
        for a in self.a:
            a_d = a._get_domain()
            d = self.d_op(False, a_d, d)
        return d

    def d_op(self, inv: bool, a: Domain, b: Domain) -> Domain:  # noqa
        """It only works on one domain, but Pycharm complains if I leave off the star.
        Pycharm doesn't complain when l_op or m_op do the same thing.

        This works for the 8 binary logic ops, so I'll put it here.  The 2 binary math ops are different."""
        return a.union(b)

    def t(self, v: float) -> float:
        """This should do it for all binary functions"""
        # for a in self.a:
        t = np.empty(0)
        for a in self.a:
            t = np.append(t, a.t(v))
        return self.l_op("e", 0, t)

    def _operate(self, precision: int, allowed_domain: Domain = None) -> _Numerical:
        """Use the operator's d_op and l_op or m_op on the three parts of the operand's numerical form."""
        na = self.a._get_numerical(precision, allowed_domain)  # a Numerical version of the a operand
        nb = self.b._get_numerical(precision, allowed_domain)  # a Numerical version of the b operand
        e = self.l_op("e", precision, na.e, nb.e)
        # on [0,1] if l_op is defined (logic op); -1 otherwise (math op).
        ax, ac, = na.xv is not None, na.cv is not None  # What ``a`` and ``b`` have (c?, x?)
        bx, bc, = nb.xv is not None, nb.cv is not None
        logic = e >= 0  # A logic or math op?
        cd, cn, xv, xt, cv, ct = None, 0, None, None, None, None
        if (ax or bx) and logic:
            xv, xt = self.l_op("x", precision, na.xv, na.xt, nb.xv, nb.xt)
        if (ax or bx) and not logic:
            xv, xt = self.m_op("x", precision, na.xv, na.xt, nb.xv, nb.xt)
        if (ac or bc) and logic:
            cd = self.d_op(False, na.cd, nb.cd)  # What the domain of the result will be.
            cv, ct = self.l_op("c", precision, na.cv, na.ct, nb.cv, nb.ct)
        if (ac or bc) and not logic:
            cd = self.d_op(False, na.cd, nb.cd)  # What the domain of the result will be.
            cv, ct = self.m_op("c", precision, na.cv, na.ct, nb.cv, nb.ct)
        if not logic:
            e = self.m_op("e", precision, na.e, nb.e)
        return _Numerical(cd, precision, cv, ct, xv, xt, e)


class And(LogicOperator, AssociativeOperator):
    name = str("∧ AND ∧")

    def _op(self, n: Norm, a: TruthOperand, b: TruthOperand) -> TruthOperand:
        return n.and_(a, b)


class Or(LogicOperator, AssociativeOperator):
    name = str("∨ OR ∨")

    def _op(self, n: Norm, a: TruthOperand, b: TruthOperand) -> TruthOperand:
        return n.or_(a, b)

# AssociativeOperator:---  logic: and, or;  math:  add, mul
# Qualifiers:  normalize, weight, focus
# FuzzyExpression---with all the static methods;  overloaded operators

# class Abs(UnaryOperator):
#     """"""
#
#     def __str__(self):
#         a = self.a.__str__()
#         return str(f"ABS({a})")
#
#     def evaluate(self, resolution: float) -> _Numerical:
#         """"""
#         na = self.a.evaluate(resolution)
#         na.d = (-na.d[1], -na.d[0])  # This will take resampling and ORing... but how far?  ...
#         if na.xp is not None:
#             xpv = -1 * xpv
#             na.xp = np.column_stack((xpv, xps))
#         na.v = -1 * na.v
#         return na
#
#     def suitability(self, v: float) -> float:
#         """"""
#         return abs(self.a.suitability(v))
