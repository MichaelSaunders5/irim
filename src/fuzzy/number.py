# re-write of Value with fixed precision

from __future__ import annotations

from abc import ABC, abstractmethod
from math import ceil, floor, pi, isfinite  # , log, sqrt
from typing import Union, Tuple  # , ClassVar,

import numpy as np

from fuzzy.crisp import Crisper, MedMax, Interpolator, Map
from fuzzy.norm import default_norm
from fuzzy.truth import Truth

default_resolution: float = .001  # Needed when calling float(Value).
"""The minimum difference in value that is considered by:

* The :meth:`crisp` method (as the default) and by ``float()``;
* All comparisons, e.g., :meth:`__gt__` (as the default) and by their operators, e.g., ``>``.
* The constructor of :class:`Numerical` (as the default), including when :meth:`.Value.evaluate` is called.

"""
default_interpolator: Interpolator = Interpolator(kind="linear")
"""The interpolator that is used when:

* Constructing a :class:`CPoints` fuzzy number (to interpolate the sample points between the knots).
* Calculating the suitability in the continuous domain of a :class:`Numerical` 
  (to interpolate between the sample points).
"""
default_crisper: Crisper = MedMax()
"""The :class:`.Crisper` (defuzzifier) that is used by the methods :meth:`.Value.crisp` (by default) and ``float()``."""
default_sampling_method: str = "Chebychev"
"""How Literals are sampled.  The options are Chebychev (producing near-minimax approximations); or "uniform"."""


def _guard(s: Operand) -> Union[float, np.ndarray]:
    """A private helper function to deal with exceptional suitabilities {-inf, nan, +inf}
    as {0, :attr:`Truth.default_threshold`, 1}.

    This is used internally in :class:`.Numerical` and :class:`Literal`, from which all user-defined fuzzy numbers
    will probably descend, so it is unlikely that you will need to use it directly.

    Args:
        s: A presumed suitability (or array of them), which should be on [0,1], but might be exceptional.

    Returns:
        The best equivalent, restricted to [0,1]."""
    r = np.nan_to_num(s, nan=Truth.default_threshold, posinf=1, neginf=0)
    r = np.clip(r, 0, 1)
    if isinstance(s, np.ndarray):
        return r
    else:
        return float(r)


class Domain(Tuple):
    def __new__(cls, d: Tuple[float, float]) -> Domain:
        if d is not None:
            if d[1] < d[0]:
                raise ValueError(f"Domains must have d[1] >= d[0]. ({d[0]}, {d[1]}) is invalid.")
        return super(Domain, cls).__new__(cls, d)

    def span(self) -> float:
        return self[1] - self[0]

    def center(self) -> float:
        return self[0] + self.span() / 2

    def intersection(self, d: Domain) -> Domain:
        new_min = max(self[0], d[0])
        new_max = min(self[1], d[1])
        return None if new_min > new_max else Domain((new_min, new_max))

    def union(self, d: Domain) -> Domain:
        new_min = min(self[0], d[0])
        new_max = max(self[1], d[1])
        return None if new_min > new_max else Domain((new_min, new_max))

    def contains(self, v: float) -> bool:
        if self is None:
            return False
        else:
            return not ((v < self[0]) or (v > self[1]))


class FuzzyNumber(ABC):
    def __init__(self, elsewhere: float = 0):
        """Args:
            elsewhere:  The truth for values that are otherwise undefined (the default is 0)."""
        self.e = elsewhere  # [0,1].  no checks for now.

    # abstracts
    @abstractmethod
    def _get_domain(self, allowed_domain: Domain = None) -> Union[Domain, None]:
        """the intersection of the expressed and allowed domains."""

    @abstractmethod
    def _get_numerical(self, precision: int, allowed_domain: Domain = None) -> _Numerical:
        # if precision == 0, the continuous part is excluded.
        """Obtains and returns a numerical representation of itself.

        valid_domain = intersection(self.cd, extereme_domain).  Choose sample points.  call _sample

        This is where the work is done in each subclass.  In a :class:`Literal` number, the work is to sample its
        analytical function appropriately.  In an :class:`Operator`, the work is to call for evaluations
        of its operands, operate on them, and return the numerical result.

        Arguments:
            resolution: The spacing in value between sample points of the continuous function.  This determines how
                accurately the result represents the original.
            extreme_domain:  ?

        Return:
            A numerical representation of ``self``, i.e., of the :math:`s(v)` that defines the fuzzy number.
        """

    @abstractmethod
    def t(self, v: float) -> float:
        """Given a value, return its truth"""

    # implementations: private helpers
    def _expression_as_numerical(self, resolution: float, allowed_domain: Domain = None) -> Numerical:
        """gets the numerical rep. of the entire expression restricted to allowed_domain, to the required resolution."""
        span = self._get_domain(allowed_domain).span()  # resolution must be > 0
        if span is None:
            precision = 0  # continuous domain is excluded. exceptional points might exist
        else:
            precision = ceil(span / resolution)  # number of sample points in each continuous part
        numerical = self._get_numerical(precision, allowed_domain)  # only seeks allowed domain
        # Impose an extreme domain, if required:
        if allowed_domain is not None:
            numerical.impose_domain(allowed_domain)  # discards some exceptional points.  continuous was trimmed already
        return numerical

    # implementations: public
    def crisp(self, resolution: float, allowed_domain: Domain = None, crisper: Crisper = None) -> float:
        """Returns a crisp value that is equivalent to ``self``\\ s fuzzy value.

        Arguments:
            resolution: The distance between sample values in the numerical representation.
                This controls the accuracy of the result (a smaller resolution is better).
                Also, consider that a coarse mesh in the numerical representation might miss narrow peaks.
                (Exceptional points defined explicitly are unaffected by resolution.)
            allowed_domain: bounds the domain of the result in case the answer must be limited,
                e.g., if tempo must be on [30, 200] bpm, or temperature must be on [-273.15, 100]°C.
            crisper:  The :class:`Crisper`  object that performs the defuzzification.
                If none is indicated, :attr:`default_crisper` is used.

        Return:
            The crisp equivalent of this fuzzy number, according to ``crisper``.
        """
        # Obtain a numerical representation of the fuzzy number at the appropriate resolution:
        numerical = self._expression_as_numerical(resolution, allowed_domain)
        # Defuzzify the fuzzy number to obtain its crisp value:
        if crisper is None:
            crisper = default_crisper
        v = crisper.defuzzify(numerical)
        return v

    def __float__(self) -> float:
        """Returns the crisp float value, via :meth:`.crisp`, using only default defuzzification parameters."""
        return self.crisp(default_resolution)

    def map(self, resolution: float, allowed_domain: Tuple[float, float] = None,
            range: Tuple[float, float] = (0, 1), map: str = "lin",
            interp: Interpolator = None) -> Map:
        """Creates a callable object that maps the :math:`s(v)` of ``self`` to the real numbers.

        A :class:`.Value` is a function of suitability vs. value.  Sometimes that function is itself a useful result.
        It can be used in crisp mathematical expressions via the callable :class:`.Map` object returned by
        this method.

        The range of the internal function is restricted to [0,1].  To make it more convenient, the parameters
        allow you to translate this to ``range`` via a ``map`` (linear, logarithmic, or exponential).  This should
        make the result more easily adaptable.

    Args:
        resolution: The distance between sample values in the numerical representation.
            This controls how accurately the :class:`.Map` represents the original (a smaller resolution is better).
            Explicitly defined exceptional points are unaffected by resolution.

        range:  Translates the range of the internal function to the indicated range.  See :meth:`.Truth.scale`.
        map:  And does so via linear, logarithmic, or exponential mapping.  See :meth:`.Truth.scale`.
        interp:  An :class:`crisp.Interpolator` object for interpolating between the sample points.
            If none is indicated, :attr:`.default_interpolator` is used.

    Returns:
        A callable object that can be used as a mathematical function.

    Example:
        | ``loudness = amplitude_vs_pitch.map(range=(0,96), map = "log"")``
        | ``y = loudness(pitch)``

        """
        # Obtain a numerical representation of the fuzzy number at the appropriate resolution:
        numerical = self._expression_as_numerical(resolution, allowed_domain)
        return Map(numerical, range, map, interp)


class Literal(FuzzyNumber):  # user only implements _sample
    """An abstract base for fuzzy numbers defined by a mathematical function given as a Python method.

    The fuzzy number may be continuous (the default) or discretized.  This is useful if the suitabilities may be
    easily defined by a mathematical function, but only valid for discrete values.
    There are three ways of doing this:

        * For an arbitrary collection of values, given in ``discrete``.
        * For a set of uniformly-spaced values, by setting ``uniform=True`` and, optionally, ``step`` and ``origin``.
        * Both of the above together.

    Subclasses must implement :meth:`.Literal._sample` to be the s(v) function that defines the fuzzy number.
    It's input and output are :class:`numpy.ndarray`\\ s, so it should be done using Numpy mathematics.

    Subclasses may well choose not to expose either or both discretization behaviors in their interface.
    They may also set their ``self.origin`` to a default, if it is not given on initialization.
    This should probably be the value where the function reaches its "peak" (``range[1]``), or some other
    critical point.  For example:

        | ``if origin is None:``
        |   ``origin = value_at_peak``

    Note:
        A Literal (or _Numerical) that represents a continuous function, can have exceptional points added by
        simple assignment, e.g.:

            | ``a_function = Triangle(1, 2, 3)``
            | ``an_exception = Exactly(8)``
            | ``a_function.xp = an_exception.xp``
            | ``print(a_function.suitability(7))``
            | ``print(a_function.suitability(8))``

        prints 0 then 1.  Note that the exceptional point needn't be in the domain of the continuous part.
    """

    def __init__(self, domain: Domain, range: Tuple[float, float] = (0, 1), elsewhere: float = 0,
                 discrete: Iterable[float] = None,
                 uniform: bool = False, step: float = 1, origin: float = None):
        """
        fold the fiddle-faddle parameters into **kwargs somehow.
        Args:
            domain:  The domain of values over which _sample defines a continuous function
                 (i.e., not including any exceptional points). The full, continuous domain will be
                used, unless it is discretized by defining ``discrete`` or by setting ``uniform=True``.
            range:  The extremes of suitability that will be reached.  Default: [0,1].
            default_suitability:  Returned for values that are otherwise undefined.  Default: 0.
            discrete: If defined, the domain is restricted to these values, given explicitly
                (and to any defined by ``uniform``).
            uniform:  If ``True``, the domain is restricted to a set of uniformly-spaced values, according
                to ``step`` and ``origin`` (and to any defined by ``discrete``).
            step:  Uniform points are spaced at these intervals over ``domain``.
            origin: Uniform points will be measured from ``origin``.  It needn't be in the domain.
                Default: the middle of the domain.  Subclasses would do well to default to the value where the
                function "peaks" (i.e., reaches ``range[1]``)."""
        super().__init__(elsewhere)
        self.d = domain  # Where _sample is defined, even if cd will be None due to discretization
        if not (Truth.is_valid(range[0]) and Truth.is_valid(range[1])):
            raise ValueError("Suitabilities like those in ``range`` must be on [0,1].")
        self.range = range
        if discrete is not None:
            discrete = np.array(discrete)
            if (discrete < domain[0]).any() or (discrete > domain[1]).any():
                raise ValueError("discrete points outside domain are redundant---they'd report the default anyway.")
        self.discrete = discrete
        self.uniform = uniform
        self.step = step
        if uniform:
            if step <= 0:
                raise ValueError("Step must be > 0.")
            if origin is None:
                self.origin = domain.center()
            else:
                self.origin = origin
        else:
            self.origin = 0

    def __str__(self):
        """"""  # skip = "" if self.xv is None else "\n"
        # str(f"domain: {self.d}, elsewhere: {self.e}, exceptional points: {skip}{self.xv}")
        # u, d, c = "", "", ""
        # if self.uniform:
        #     u = str(f"uniform {self.step} steps from {self.origin}")
        # if self.discrete is not None:
        #     d = str(f"at values: {self.discrete}")
        # if self.uniform and self.discrete is not None:
        #     c = str(f" and at ")
        # if self.uniform or self.discrete is not None:
        #     i = str(f"discretization: ")
        # else:
        #     i = str(f"no discretization. ")
        # return str(f"{s} \n range: {self.range}; {i}{d}{c}{u}")

    @abstractmethod
    def _sample(self, v: np.ndarray) -> np.ndarray:
        """This is where you implement the t(v) function that defines your fuzzy number.
        It should return a truth on [0,1] for any real number in the domain, ``self.d``."""

    def _get_domain(self, extreme_domain: Domain = None) -> Domain:
        return self.d.intersection(extreme_domain)

    def _get_numerical(self, precision: int, allowed_domain: Domain = None) -> _Numerical:
        """"""
        domain = self.d  # The domain on which _sample() is defined (and a bit beyond, we can hope).
        if allowed_domain is not None:
            domain = domain.intersection(allowed_domain)
        if (domain is None) or (precision <= 0):
            return _Numerical(None, 0, None, None, None, None, self.e)
        if len(self.discrete) or self.uniform:  # Discretiztion has been called for.
            cd, cn, cv, ct = None, 0, None, None
            xv = np.empty(0)
            if self.uniform:  # Find the uniformly-spaced sample points inside domain.
                n0 = ceil((domain[0] - self.origin) / self.step)  # Extreme sample numbers relative to origin.
                n1 = floor((domain[1] - self.origin) / self.step)
                v0 = self.origin + n0 * self.step  # extreme values
                v1 = self.origin + n1 * self.step
                xv = np.linspace(v0, v1, n1 - n0 + 1)  # calculate the pairs
            if self.discrete is not None:
                xv = np.unique(np.concatenate((xv, self.discrete)))
            xt = self._sample(xv)  # Sample the Literal at the exceptional points.
            xt = _guard(xt)
        else:  # A continuous function has been called for. Find cv cleverly.
            xv, xt = None, None
            cd = domain
            cn = 3 if (precision < 3) else precision
            # below: 2 to guard the secant & interval from exploding; 3 allows possible linear extrapolation.
            if default_sampling_method == "Chebychev":
                m = cn + 2
                cv = np.arange(1, m + 1, 1)
                cv = .5 * cd.span() * (-cos((cv - .5) * pi / m)) / cos(1.5 * pi / m) + cd.center()
            else:
                interval = cd.span() / (cn - 1)
                cv = np.linspace(cd[0] - interval, cd[1] + interval, num=cn + 2)
            ct = self._sample(cv)  # Sample the Literal on the continuous domain.
            if (len(ct) >= 4) and (not isfinite(ct[-2])):  # because Akima interp. produces two NaNs at the end.
                ct[-2] = 2 * ct[-3] - ct[-4]
            ct = np.concatenate((ct[0:1], _guard(ct[1:-1]), [ct[-1]]))  # Make the pts. on domain safe.
            # Literals aren't guaranteed at their guard points (beyond self.d).  So, if they're funny...
            # linear extrapolation to them---excursions beyond [0,1] allowed here---preserve the shape on domain.
            if not isfinite(ct[0]):
                ct[0] = 2 * ct[1] - ct[2]
            if not isfinite(ct[-1]):
                ct[-1] = 2 * ct[-2] - ct[-3]
        return _Numerical(cd, cn, cv, ct, xv, xt, self.e)

    def t(self, v: float) -> float:
        """Returns the suitability of a given value.

        See :meth:`.Value.suitability`.
        """
        t = self.e
        if self.uniform or len(self.discrete):
            if self.uniform and ((v - self.origin) / self.step).is_integer():
                t = self._sample(np.array([v]))[0]
            if len(self.discrete) and (v in self.discrete):
                t = self._sample(np.array([v]))[0]
        elif self.d.contains(v):
            t = self._sample(np.array([v]))[0]
        return _guard(t)


class _Numerical(FuzzyNumber):  # user doesn't see it!  dataclass?

    def __init__(self, cd: Domain = None, cn: int = 0, cv: np.ndarray = None, ct: np.ndarray = None,
                 xv: np.ndarray = None, xt: np.ndarray = None, e: float = 0):
        """Args:
        should be :
        cd: a valid Domain with d[1]>=d[0]
        cn>0 (really, >2)
        cv, xv each:  all elements unique
        ct, xt, e:  all elements on [0,1] except for ct[0] and ct[-1] which should be finite.
        length of cv and ct equal
        length of xv and xt equal
        I'd check, but this is private.
        """
        self.cd = cd  # domain of continuous part
        self.cn = cn  # number of samples in  continuous part
        self.cv = cv  # continuous part: value samples
        self.ct = ct  # continuous part: truth samples
        self.xv = xv  # exceptional points: value samples
        self.xt = xt  # exceptional points: truth samples
        super().__init__(e)  # elsewhere

    def __str__(self):
        skip = "" if self.xv is None else "\n"
        return str(f"domain: {self.cd}, elsewhere: {self.e}, exceptional points: {skip}{self.xv}")

    def _get_domain(self, allowed_domain: Domain = None) -> Union[Domain, None]:
        return self.cd.intersection(allowed_domain)

    def _get_numerical(self, precision: int, allowed_domain: Domain = None) -> _Numerical:
        """Ignores the precision---changing it would require resampling itself which would lower the quality."""
        numerical = _impose_domain(self, allowed_domain)
        if precision == 0:
            numerical.cd = None
            numerical.cn = 0
            numerical.cv = None
            numerical.ct = None
        return numerical  # but allowed_domain

    def _impose_domain(self, allowed_domain: Domain) -> _Numerical:
        """Remove xp outside allowed_domain.  The continuous domain should have been trimmed already

        restricting the xv makes sense.
        restricting the cv: resampling or discarding cv points would lower the quality, so, no.  only
        removing xv, xt exceptional points makes sense"""
        i = np.where((self.xv < allowed_domain[0]))
        new_xv = np.delete(self.xv, i)
        new_xt = np.delete(self.xt, i)
        i = np.where((new_xv > allowed_domain[1]))
        new_xv = np.delete(new_xv, i)
        new_xt = np.delete(new_xt, i)
        if len(new_xv) == 0:
            new_xv = new_xt = None
        return _Numerical(self.cd, self.cn, self.cv, self.ct, new_xv, new_xt, self.e)

    @staticmethod
    def _t_for_v_in_xv(v: float, xv: np.ndarray, xt: np.ndarray) -> Union[float, None]:
        """Implements the check for discrete points.

        This is called by :class:`.Literal` and :class:`.Numerical`.

        Args:
            v: A value to be checked.
            xp:  An array of (v, s) points.

            Returns:
                If ``v`` is in ``xp``, the corresponding ``s``, otherwise, ``None``.
            """
        # return xt[i] for v in xv at i
        if (xv is None) or (xt is None):
            return None
        i = np.where(xv == v)[0]
        if len(i) == 0:
            return None
        return xt[i[0]]

    def t(self, v: float, interp: Interpolator = None) -> float:
        """Given a value, return its truth"""
        t = _t_for_v_in_xv(v, self.xv, self.xt)
        if t is None:
            if self.d.contains(v):
                if interp is None:
                    interp = default_interpolator
                t = interp.interpolate(v, self.cv, self.ct)
            else:
                t = self.e
        return _guard(t)

#
# class Operator(FuzzyNumber):
#     """has operands.  _get_numerical does the work.  ..sub: unary, binary, associative"""
#
#     def __init__(self, elsewhere: float = 0,
#                  norm=None, override_operand_norms=True):
#         super().__init__(elsewhere)
#         if norm is None:
#             self.norm = default_norm
#         else:
#             self.norm = norm
#         if override_operand_norms:
#             self.operand_norm = self.norm
#         else:
#             self.operand_norm = None
#
#     @abstractmethod
#     def _interval_operator(self, a: float = 0, b: float = None) -> float:
#         # You define your crisp operator so it can do the interval math.
#         # e.g.: return a + b, _safe_div(1,a), min(a,b) if left
#         pass
#
#     def _right_interval_operator(self, a: float = 0, b: float = None) -> float:
#         # override only if they are different., max(a,b) if right
#         return _interval_operator(a, b)
#
#     @abstractmethod
#     def _get_domain(self, allowed_domain: Domain = None) -> Union[Domain, None]:
#         # unary:
#         a = self.a._get_domain()
#         r0 = self._interval_operator(a[0])
#         r1 = self._interval_operator(a[1])
#         return Domain(tuple(sorted((r0, r1)))).intersection(allowed_domain)
#
#         # binary:
#         a = self.a._get_domain()
#         b = self.b._get_domain()
#         r0 = self._interval_operator(a[0], b[0])
#         r1 = self._right_interval_operator(a[1], b[1])
#         return Domain(tuple(sorted((r0, r1)))).intersection(allowed_domain)
#
#         # associative:
#         a = self.c[0]._get_domain()
#         r0, r1 = a[0], a[1]
#         for item in self.c[1:]:
#             b = item._get_domain()
#             r0 = self._interval_operator(r0, b[0])
#             r1 = self._right_interval_operator(r1, b[1])
#         return Domain(tuple(sorted((r0, r1)))).intersection(allowed_domain)
#
#         # d = get domains of its operands, do interval math on them
#         # # eg. (3,5) + (5,12) = (8, 17) int (0,10) = (8,10)  return d.intersection(allowed_domain)
#         return None
#
#     @abstractmethod
#     def _get_numerical(self, precision: int, allowed_domain: Domain = None, norm=None) -> _Numerical:
#         # do interval math to see what operand subdomains to ask for.
#         # # eg. (3,5) + (5,12) = (8, 17) intersection (0,10) = (8,10)
#         # # _get_numerical of a on (3,5) _get_numerical of b on (5,7)
#         # a = self.a._get_numerical(precision, allowed_domain, , self.operand_norm)
#         # c = self.b._get_numerical(precision, allowed_domain, , self.operand_norm)
#         # # sample matrix on (8,10)
#         # i.e., only sample the parts of the literal that will matter.  If extreme_ domain becomes None, don't bother.
#         # build matrix cart. prod aof a and b, sample it n times.  return!
#         # return _Numerical(cd, cn, cv, ct, xv, xt, e)
#         pass
#
#     def t(self, v: float) -> float:
#         """Given a value, return its truth"""
#
#
# class UnaryOperator(Operator):
#     """has operands.  _get_numerical does the work.  ..sub: unary, binary, associative"""
#
#     def __init__(self, a: FuzzyNumber,
#                  elsewhere: float = 0, norm=None, override_operand_norms=True):
#         super().__init__(elsewhere, norm, override_operand_norms)
#         self.a = a
#
#     @abstractmethod
#     def _interval_operator(self, a: float) -> float:
#         # You define your crisp operator so it can do the interval math.
#         # e.g.: return -a, _safe_div(1,a), abs(a)-> if left
#         pass
#
#     def _get_domain(self, allowed_domain: Domain = None) -> Union[Domain, None]:
#         a = self.a._get_domain()
#         r0 = self._interval_operator(a[0])
#         r1 = self._interval_operator(a[1])
#         return Domain(tuple(sorted((r0, r1)))).intersection(allowed_domain)
#
#     @abstractmethod
#     def _get_numerical(self, precision: int, allowed_domain: Domain = None, norm=None) -> _Numerical:
#         # do interval math to see what operand subdomains to ask for.
#         # # eg. (3,5) + (5,12) = (8, 17) intersection (0,10) = (8,10)
#         target_domain = self._get_domain(allowed_domain)
#         if target_domain is None:
#             return None
#
#         # # _get_numerical of a on (3,5) _get_numerical of b on (5,7)
#         # a = self.a._get_numerical(precision, allowed_domain, , self.operand_norm)
#         # c = self.b._get_numerical(precision, allowed_domain, , self.operand_norm)
#         # # sample matrix on (8,10)
#         # i.e., only sample the parts of the literal that will matter.  If extreme_ domain becomes None, don't bother.
#         # build matrix cart. prod aof a and b, sample it n times.  return!
#         # return _Numerical(cd, cn, cv, ct, xv, xt, e)
#         pass
