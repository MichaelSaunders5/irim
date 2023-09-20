"""re-write of Value with fixed precision"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import ceil, floor, pi, isfinite  # , log, sqrt
from typing import Union, Tuple  # , ClassVar,

import matplotlib.pyplot as plt
import numpy as np
from screeninfo import get_monitors

from fuzzy.crisp import Crisper, MedMax, Interpolator, Map
from fuzzy.truth import Truth, TruthOperand, default_threshold

monitor_w, monitor_h = 1920, 1080  # When the module loads, find the monitor dimensions, for use by :meth:`.display`.
for m in get_monitors():
    if m.is_primary:
        if (monitor_w is not None) and (monitor_w > 0):
            monitor_w, monitor_h = m.width, m.height

default_resolution: float = .01  # Needed when calling float(FuzzyNumber).
"""The minimum difference in value that is considered by:

* The :meth:`.FuzzyNumer.crisp` method (as the default) and by ``float()``;
* All comparisons, e.g., :meth:`.__gt__` (as the default) and by their operators, e.g., ``>``.
* The constructor of :class:`._Numerical` (as the default), including 
  when :meth:`.FuzzyNumber._get_numerical` is called.
  
Who can say what is best for all units and situations?  No one!  But, in the above situations, 
there is no place for a parameter."""
default_interpolator: Interpolator = Interpolator(kind="linear")
"""The interpolator that is used when:

* Constructing a :class:`literal.CPoints` fuzzy number (to interpolate the sample points between the knots).
* Calculating the truth in the continuous domain of a :class:`._Numerical` 
  (to interpolate between the sample points)."""
default_crisper: Crisper = MedMax()
"""The :class:`.Crisper` (defuzzifier) that is used by the methods :meth:`.FuzzyNumber.crisp` (by default) 
and ``float()``."""
default_sampling_method: str = "Chebyshev"
"""How :class:`.Literal`\\ s are sampled.  
The options are Chebyshev (producing near-minimax approximations); or "uniform"."""


def _guard(t: TruthOperand) -> Union[float, np.ndarray]:
    """A private helper function to deal with exceptional truths {-inf, nan, +inf}
    as {0, :attr:`Truth.default_threshold`, 1}.

    This is used internally in :class:`._Numerical` and :class:`Literal`, from which all user-defined fuzzy numbers
    will probably descend, so it is unlikely that you will need to use it directly.

    Args:
        s: A presumed truth (or array of them), which should be on [0,1], but might be exceptional.

    Returns:
        The best equivalent, restricted to [0,1], returned as an array or float depending on the argument type."""
    r = np.nan_to_num(t, nan=default_threshold, posinf=1, neginf=0)
    r = np.clip(r, 0, 1)
    if isinstance(t, np.ndarray):
        return r
    else:
        return float(r)


def _t_for_v_in_xv(v: float, xv: np.ndarray, xt: np.ndarray) -> Union[float, None]:
    """Implements the check for discrete points.

    This is called by :meth:`._Numerical.t`.

    Args:
        v: A value to be checked.
        xp:  An array of (v, s) points.

        Returns:
            If ``v`` is in ``xp``, the corresponding ``t``, otherwise, ``None``.
        """
    if (xv is None) or (xt is None):
        return None
    i = np.where(xv == v)[0]
    if len(i) == 0:
        return None
    return xt[i][0]


def _check_points(points: np.ndarray) -> (np.ndarray, np.ndarray):
    """Checks a set of (v,t) points for validity and returns them as separate v,t arrays."""
    xv = np.atleast_2d(points)[:, 0]
    xt = np.atleast_2d(points)[:, 1]
    if (np.max(xt) > 1) or (np.min(xt) < 0):
        too_high = xv[np.where(xt > 1)]
        too_low = xv[np.where(xt < 0)]
        raise ValueError(f"Truths must be on [0,1].  "
                         f"Yours are too high at v=={too_high}; and too low at v=={too_low}.")
    all_values, counts = np.unique(xv, return_counts=True)
    dup_values = all_values[np.where(counts > 1)]
    if len(dup_values) > 0:
        raise ValueError(f"You cannot define two truths for the same value.  "
                         f"You have more than one truth at v={dup_values}.")
    return xv, xt


class Domain(Tuple):
    def __new__(cls, d: Tuple[float, float]) -> Domain:
        if d is not None:
            if d[1] < d[0]:
                raise ValueError(f"Domains must have d[1] >= d[0]. ({d[0]}, {d[1]}) is invalid.")
        return super(Domain, cls).__new__(cls, d)

    def __str__(self):
        return str(f"({self[0]:,.4g}, {self[1]:,.4g})")

    def span(self) -> float:
        """Returns the extent of the domain."""
        return self[1] - self[0]

    def center(self) -> float:
        """Returns the central coördinate of the domain."""
        return self[0] + self.span() / 2

    def intersection(self, d: Domain) -> Union[Domain, None]:
        """Returns the domain that is ``self`` ∩ ``d``."""
        if d is None:
            return None
        new_min = max(self[0], d[0])
        new_max = min(self[1], d[1])
        return None if new_min > new_max else Domain((new_min, new_max))

    def union(self, d: Domain) -> Domain:
        """Returns the domain that is ``self`` ∪ ``d``."""
        if d is None:
            return self
        new_min = min(self[0], d[0])
        new_max = max(self[1], d[1])
        return Domain((new_min, new_max))

    def contains(self, v: float) -> bool:
        """True iff ``v`` is on ``self``."""
        if self is None:
            return False
        else:
            return not ((v < self[0]) or (v > self[1]))

    @staticmethod
    def sort(*d) -> Union[Domain, None]:
        if d is None:
            return None
        if d[1] < d[0]:
            return Domain((d[1], d[0]))
        else:
            return Domain((d[0], d[1]))


class FuzzyNumber(ABC):
    """A fuzzy real number.

    The representation is a function, :math:`t(v)`, of "suitability" or "truth" (on [0,1]) vs. value
    (on the real numbers).  Generally, subclasses may explicitly define:

        * Exceptional points, (v, t) pairs;
        * A continuous function, t(v), over a single contiguous domain, defined either

            * numerically (by a :class:`._Numerical` object), or
            * programmatically (by a Python/Numpy/SciPy method).

    But, all :class:`.FuzzyNumber`\\ s have an attribute, ``e`` for "elsewhere", describing the truth for every real
    number where it is not explicitly defined.  So, all :class:`.FuzzyNumber`\\ s are defined for all ``float`` inputs.

    It implements:

        * :meth:`.display`, which shows a picture of the function;
        * :meth:`.map`, which turns the function into a callable object;
        * :meth:`.FuzzyNumber.crisp`, which finds the best real number equivalent for it (by defuzzification); and
        * ``float()``, which does the same using only default parameters.

    Its subclasses implement:

        * :meth:`.t`, which returns the truth of a given value; and
        * :meth:`._get_numerical`, which returns a numerical representation
          of :math:`t(v)`, a :class:`._Numerical`; and
        * :meth:`._get_domain`, which returns the domain over which
          the continuous part of :math:`t(v)` is explicitly defined."""

    def __init__(self, elsewhere: float = 0):
        """Args:
            elsewhere:  The truth for values that are otherwise undefined (the default is 0)."""
        self.e = elsewhere  # Must be on [0,1].  no checks for now.

    # abstracts
    @abstractmethod
    def _get_domain(self) -> Union[Domain, None]:
        """The intersection of the expressed and allowed domains.

        No need for this to impose any allowed_domain!!!!!!!?
        going up the tree to find the total domain: higher d = d_op(operand d).
        going down the tree to say only sample restricted d:
        lower allowed d = d_op_inv(allowed d).
        eg, if the operator is "+5" and operand d= 3,5, send up 3+5,5+5 = 8,10.
        if only interested in result d = 0,1, only sample (send down) 0-5,1-5 = -5,-4
        "up" = _get_domain for precision calc
        "down" = _get_numerical to ask down to sample c narrowly
        (and its return can also trim xp)



        By "expressed" I mean the single contiguous domain over which the fuzzy number is defined explicitly as a
        continuous function.  This does not include definitions of exceptional points or of the default truth
        elsewhere.  Since a fuzzy number might be an expression involving an arbitrary number of logic and math
        operators, this isn't a simple question, but it can be handled by recursive calls.

        Args:
            allowed_domain: The extreme domain on which a result is acceptable.

        Return:
            The intersection of the defined and acceptable domains.  If the answer is ``None``, you might want to
            reconsider ``allowed_domain`` (but: some exceptional points might still be in play and "elsewhere" is
            always in play.)"""

    @abstractmethod
    def _get_numerical(self, precision: int, allowed_domain: Domain = None) -> _Numerical:
        """Obtains and returns a numerical representation of itself.

        This is where the work is done in each subclass.  In a :class:`Literal` number, the work is to sample its
        analytical function appropriately.  In an :class:`Operator`, the work is to call for evaluations
        of its operands, operate on them, and return the numerical result.

        Arguments:
            precision: The number of sample points used to cover the defined domain of the continuous function.
                This determines how accurately the result represents the original.  It will have been calculated
                from the caller's desired ``resolution``, the minimum significant difference in value.
                (In practice, two guard points will always be added outside the domain.)
            allowed_domain: The extreme domain on which a result is acceptable.

        Note:
            :class:`.Operator`\\ s hold their operands, which are fuzzy numbers---either :class:`.Literal`\\ s or more
            Operators---so calling this method on an expression built up in this way can result in many recursive
            calls.  By restricting the requested domain with ``allowed_domain`` at each step, calculations avoid
            considering inconsequential values, and concentrate all of their precision (sample points) only
            on stretches of domain that will have a bearing on the final result.

        Note:
            In cases where ``precision == 0``, the continuous part is excluded from calculations.

        Return:
            A numerical representation of ``self``, i.e., of the :math:`t(v)` that defines the fuzzy number.
        """

    @abstractmethod
    def t(self, v: float) -> float:
        """Given a value, return its truth.

        It should refer to, in order of priority:  the exceptional points, the continuous function,
        and the default truth "elsewhere".  The main case for this method's use is to check on how good
        the result of :meth:`.FuzzyNumber.crisp` is.

        Args:
            v: any real number, a proposed value.

        Returns:
            The suitability of the proposed value, a measure of truth in the range [0,1],
            the degree to which it is a member of the result."""

    # implementations: private helpers
    def _expression_as_numerical(self, resolution: float, allowed_domain: Tuple[float, float] = None) -> _Numerical:
        """A helper that gets the numerical representation of the entire expression, restricted to ``allowed_domain``,
        at the required ``resolution``.  It is used by :meth:`.FuzzyNumber.crisp`, :meth:`.map`. and :meth:`.display`.
        Its main task is to calculate the required precision from the given resolution and to make sure the resulting
        domain is properly restricted.  ``resolution`` must be > 0."""
        domain = self._get_domain()  # What the raw, unrestricted result would be.
        if allowed_domain is not None:
            domain = Domain(allowed_domain).intersection(domain)  # What the restricted result will be...
        span = None if domain is None else domain.span()  # ...for the precision calculation:
        if span is None:
            precision = 0  # continuous domain is excluded. exceptional points might exist
        elif span == 0:
            precision = 1  # continuous domain is a point.  Trouble?
        else:
            precision = ceil(span / resolution)  # number of sample points in each continuous part
            precision = precision + 1 if (precision % 2) == 0 else precision  # Insist on odd precisions?
        numerical = self._get_numerical(precision, allowed_domain)  # only seeks allowed domain
        if allowed_domain is not None:  # Impose an extreme domain, if required:
            numerical = _Numerical._impose_domain(numerical, allowed_domain)  # noqa Discard some exceptional points
            # ---has to be done separately because continuous was restricted (above) to its intersection with allowed.
            # hmmm.  a parallel xp_domain could avoid keep track of it and avoid unnecessary calculation.
            # It might be more efficient, but it wouldn't affect precision.
        return numerical

    # implementations: public
    def crisp(self, resolution: float, allowed_domain: Domain = None, crisper: Crisper = None) -> float:
        """Returns a crisp value that is equivalent to ``self``\\ s fuzzy value.

        Arguments:
            resolution: The smallest significant difference between values in the numerical representation.
                This controls the accuracy of the result (a smaller resolution is better).
                Also, consider that a coarse mesh in the numerical representation might miss narrow peaks.
                (Exceptional points defined explicitly are unaffected by resolution.)
            allowed_domain: bounds the domain of the result in case the answer must be limited,
                e.g., if tempo must be on [30, 200] bpm, or temperature must be on [-273.15, 100]°C.
            crisper:  The :class:`.Crisper`  object that performs the defuzzification.
                If none is indicated, :attr:`.default_crisper` is used.

        Return:
            The crisp equivalent of this fuzzy number, according to ``crisper``, considering only ``allowed_domain``.
        """
        numerical = self._expression_as_numerical(resolution, allowed_domain)
        # Defuzzify the fuzzy number to obtain its crisp value:
        if crisper is None:
            crisper = default_crisper
        v = crisper.defuzzify(numerical)
        return v

    def __float__(self) -> float:
        """Returns the crisp float value, via :meth:`.FuzzyNumber.crisp`, using only default parameters."""
        return self.crisp(default_resolution)

    def map(self, resolution: float, allowed_domain: Tuple[float, float] = None,
            range: Tuple[float, float] = (0, 1), map: str = "lin",
            interp: Interpolator = None) -> Map:
        """Creates a callable object that maps the :math:`t(v)` of ``self`` to the real numbers.

        A :class:`.FuzzyNumber` is a function of truth vs. value.  Sometimes that function is itself a useful result.
        It can be used in crisp mathematical expressions via the callable :class:`.Map` object returned by
        this method.

        The range of the internal function is restricted to [0,1].  To make it more convenient, the parameters
        allow you to translate this to ``range`` via a ``map`` (linear, logarithmic, or exponential).  This should
        make the result more easily adaptable.

    Args:
        resolution: The distance between sample values in the numerical representation.
            This controls how accurately the :class:`.Map` represents the original (a smaller resolution is better).
            Explicitly defined exceptional points are unaffected by resolution.
        allowed_domain:  Restricts the defined domain of the result to no more than this parameter, i.e., discarding
            any exceptional points or continuous domain outside it---these will return the default "elsewhere" truth.
        range:  Translates the range of the internal function to the indicated range.  See :meth:`.Truth.scale`.
        map:  And does so via linear, logarithmic, or exponential mapping.  See :meth:`.Truth.scale`.
        interp:  An :class:`.crisp.Interpolator` object for interpolating between the sample points.
            If none is indicated, :attr:`.default_interpolator` is used.

    Returns:
        A callable object that can be used as a mathematical function.

    Example:
        | ``loudness = amplitude_vs_pitch.map(range=(0,96), map = "log"")``
        | ``y = loudness(pitch)``
        """
        numerical = self._expression_as_numerical(resolution, allowed_domain)
        return Map(numerical, range, map, interp)

    def display(self, resolution: float = None, domain: Domain = None, display_percentage: float = 25) -> None:
        # TODO: Needs work
        """Displays the t(v) that is the :class:`.FuzzyNumber`.

        Args:
            resolution:  The minimum significant difference in value for the fuzzy number.
                Default: the equivalent of one pixel.
            domain:  The domain of values to display. Default: the defined continuous domain.
            display_percentage:  The percentage of your monitor area to be taken up by the plot itself.
        """
        display_factor = (display_percentage / 100) ** .5
        px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
        plt.subplots(figsize=(monitor_w * display_factor * px, monitor_h * display_factor * px))
        if resolution is None:
            defined_c_domain = self._get_domain()
            if defined_c_domain is not None and domain is not None:
                defined_c_domain = defined_c_domain.intersection(domain)
            if defined_c_domain is not None:
                resolution = defined_c_domain.span() / (monitor_w * display_factor)
        fn = self._expression_as_numerical(resolution, domain)
        if domain is None:  # set the domain to show all exceptional points.
            c_domain = None if fn.cv is None else fn.cd
            x_domain = None if fn.xv is None else Domain((np.amin(fn.xv), np.amax(fn.xv)))
            c_domain = x_domain if c_domain is None else c_domain
            x_domain = c_domain if x_domain is None else x_domain
            domain = Domain((0, 0)) if ((c_domain is None) and (x_domain is None)) else c_domain.union(x_domain)
        # plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        plt.xlim(*domain)  # domain to plot
        plt.ylim(-.01, 1.01)
        plt.grid()
        v = fn.cv
        s = fn.ct
        if v is not None:
            plt.plot(v, s, color="red")
        v = np.array([*domain])
        s = np.array([fn.e, fn.e])
        plt.plot(v, s, color="yellow")
        if fn.xv is not None:
            xpv = fn.xv
            xps = fn.xt
            plt.plot(xpv, xps, "o", markersize=3, color="blue")
        # plt.title("Line graph")   # There's no way to get the name of the variable.
        plt.show()


class Literal(FuzzyNumber):  # user only implements _sample
    """An abstract base for fuzzy numbers defined by a mathematical function given as a Python method.

    The fuzzy number may be continuous (the default) or discretized.  This is useful if the truth may be
    easily defined by a mathematical function, but only valid for discrete values.
    There are three ways of doing this:

        * For an arbitrary collection of values, given by ``discrete=``.
        * For a set of uniformly-spaced values, by setting ``uniform=True`` and,
          optionally, ``step=`` and ``origin=``.
        * Both of the above together.

    Subclasses must implement :meth:`.Literal._sample` to be the t(v) function that defines the fuzzy number.
    Its input and output are :class:`numpy.ndarray`\\ s, so it should be done using vectorized Numpy mathematics
    (i.e., with `*ufunc*\\ s <https://numpy.org/doc/stable/reference/ufuncs.html#available-ufuncs>`_).

    Subclasses may set their ``self.origin`` to a default, if it is ``None`` (not given on initialization).
    This should probably be the value where the function reaches its "peak" (``range[1]``), or some other
    critical point.

    Note:
        A Literal (or _Numerical) that represents a continuous function, can have exceptional points added by
        simple assignment, e.g.:

            | ``a_function = Triangle(1, 2, 3)``
            | ``a_point = Exactly(8)``
            | ``a_function.xv, a_function.xt = a_point.xv, a_point.xt``
            | ``print(a_function.t(7))``
            | ``print(a_function.t(8))``

        prints 0 then 1.  Note that the exceptional point needn't be in the domain of the continuous part.
    """

    # domain: Domain, range: Tuple[float, float] = (0, 1), elsewhere: float = 0
    # expected_range: Tuple[float, float] = (0, 1), intended_range: Tuple[float, float] = (0, 1),
    #                  map: str = "lin", clip: bool = False,
    #                  elsewhere: float = 0
    def __init__(self, **kwargs):
        """
        Caution:
            Authors of subclasses, take note.
            Two of the keyword arguments, ``domain`` and ``origin`` are set to ``None`` by default.  This will cause
            errors unless the subclass replaces them with a default.  The superclass (here) can't
            because it doesn't know the intentions of the subclass.  The best practices are as follows:

                * If ``domain == None``:  set it to the domain where the function varies interestingly,
                  e.g., over [.001, .999].
                * If ``origin == None``:  set it to the value where the function peaks, inflects, or has some other
                  critical feature.

        Keyword Arguments:

            domain= (Tuple[float, float]):
                The domain of values over which _sample defines a continuous function
                (i.e., not including any exceptional points). The full, continuous domain will be
                used, unless it is discretized by defining ``discrete`` or by setting ``uniform=True``.
                Values not on ``domain`` or defined as exceptional points have a truth of ``elsewhere``.
                N.B.: if a continuous function is intended and users do not set the domain, subclasses should
                choose a default---probably the domain where the truth has most of its variation (say, 99.9%).
            range= (Tuple[float, float]):
                The extremes of truth that will be reached, scaled from ``expected_range``.  Default: [0,1].
                If only exceptional points are intended, ``domain`` *should be* ``None``.  Default: ``None``.
            elsewhere= (float):
                Returned for values that are otherwise undefined.  Default: 0.

            discrete= (Iterable[float]):
                If defined, the domain is restricted to these values, given explicitly
                (and to any defined by ``uniform``).
            uniform= (bool):
                If ``True``, the domain is restricted to a set of uniformly-spaced values, according
                to ``step`` and ``origin`` (and to any defined by ``discrete``).
            step= (float):
                Uniform points are spaced at these intervals over ``domain``.  Default: 1.
            origin= (float):
                Uniform points will be measured from ``origin``.  It needn't be in the domain.
                Default:  ``None``---the subclass must choose a default,
                and would do well to use the value where the
                function "peaks" (i.e., reaches ``range[1]``).

            points= (Iterable[Tuple[float, float]]):
                  A container of (v, t) pairs to be added to the continuous definition.  Default: ``None``.

                """
        # domain-related:
        super().__init__(kwargs.get('elsewhere', 0))
        domain = kwargs.get('domain', None)  # Where _sample is defined, even if cd will be None due to discretization
        self.d = None if domain is None else Domain(domain) if isinstance(domain, tuple) else domain
        # If None, subclasses must supply a default
        self.range = kwargs.get('range', (0, 1))
        if not (Truth.is_valid(self.range[0]) and Truth.is_valid(self.range[1])):
            raise ValueError("Truths like those in ``range`` must be on [0,1].")
        # discretization-related:
        self.discrete = kwargs.get('discrete', None)
        self.uniform = kwargs.get('uniform', False)
        self.step = kwargs.get('step', 1)
        self.origin = kwargs.get('origin', None)
        if self.discrete is not None:
            self.discrete = np.array(self.discrete)
        if self.uniform:
            if self.step <= 0:
                raise ValueError("Step must be > 0.")
        # exceptional points
        points = kwargs.get('points', None)  # TODO: Am I going to make this keyword-settable???
        if (points is not None) and (len(points) != 0):
            xv, xt = _check_points(points)
            self.xv, self.xt = xv, xt
        else:
            self.xv, self.xt = None, None  # They can hold exceptional points if they are assigned.

    def __str__(self):
        s = str(f"domain: {self.d}, range: ({self.range[0]:.3g}, {self.range[1]:.3g}), "
                f"elsewhere: {self.e:.4g}")
        if self.uniform or self.discrete is not None:
            i = str(f"\n discretization: ")
        else:
            i = str(f"no discretization. ")
        u, d, c = "", "", ""
        if self.discrete is not None:
            d = str(f"at values: {self.discrete}")
        if self.uniform and self.discrete is not None:
            c = str(f" and at ")
        if self.uniform:
            u = str(f"uniform {self.step:g} steps from {self.origin:g}")
        if self.xv is None:
            x = str(f"no exceptional points.")
        else:
            x = str(f"exceptional points: \n {np.dstack((self.xv, self.xt))[0]}")
        return str(f"{s}; {i}{d}{c}{u}; {x}")

    @abstractmethod
    def _sample(self, v: np.ndarray) -> np.ndarray:
        """This is where you implement the t(v) function that defines your fuzzy number.
        It should return a truth on [0,1] for any real number in the domain, ``self.d``,
        using vectorized Numpy functions.

        You can call ``super()._sample(v) for values outside ``self.d``---or simply return ``self.e`` for them."""

    def _get_domain(self, extreme_domain: Domain = None) -> Domain:
        if extreme_domain is None:
            return self.d
        else:
            return self.d.intersection(extreme_domain)

    def _get_numerical(self, precision: int, allowed_domain: Domain = None) -> _Numerical:
        """"""
        domain = self.d  # The domain on which _sample() is defined (and a bit beyond, we can hope).
        if allowed_domain is not None:
            domain = domain.intersection(allowed_domain)
        if (domain is None) or (precision <= 0):
            num = _Numerical(None, 0, None, None, self.xv, self.xt, self.e)
            if allowed_domain is not None:
                num = _Numerical._impose_domain(num, allowed_domain)  # noqa
            return num
        xv, xt = self.xv, self.xt
        discrete = (self.discrete is not None) and (len(self.discrete) > 0)
        if discrete or self.uniform:  # Discretization has been called for.
            cd, cn, cv, ct = None, 0, None, None
            xv = np.empty(0)
            if self.uniform:  # Find the uniformly-spaced sample points inside domain.
                n0 = ceil((domain[0] - self.origin) / self.step)  # Extreme sample numbers relative to origin.
                n1 = floor((domain[1] - self.origin) / self.step)
                v0 = self.origin + n0 * self.step  # extreme values
                v1 = self.origin + n1 * self.step
                xv = np.linspace(v0, v1, n1 - n0 + 1)  # Calculate the values of the points.
            if discrete:
                xv = np.unique(np.concatenate((xv, self.discrete)))
            xv = np.setdiff1d(xv, self.xv, assume_unique=True)  # Remove from xv any elements that are in self.xv
            xt = self._sample(xv)  # Sample the Literal at the exceptional points.
            xt = _guard(xt)
            if self.xv is not None:
                xv = np.concatenate((xv, self.xv), axis=None)
                xt = np.concatenate((xt, self.xt), axis=None)
        else:  # A continuous function has been called for. Find cv cleverly.
            cd = domain
            cn = 3 if (precision < 3) else precision  # I.e., if it's 1 or 2, make it 3.
            # below: 2 to guard the secant & interval from exploding; 3 allows possible linear extrapolation.
            if default_sampling_method == "Chebyshev":  # Find the values to sample at, Chebyshev or uniform.
                m = cn + 2
                cv = np.arange(1, m + 1, 1)
                cv = .5 * cd.span() * (-np.cos((cv - .5) * pi / m)) / np.cos(1.5 * pi / m) + cd.center()
            else:
                interval = cd.span() / (cn - 1)
                cv = np.linspace(cd[0] - interval, cd[1] + interval, num=(cn + 2))
            ct = self._sample(cv)  # Sample the Literal on the continuous domain.
            if (len(ct) >= 4) and (not isfinite(ct[-2])):  # because Akima interp. produces two NaNs at the end.
                ct[-2] = 2 * ct[-3] - ct[-4]
            ct = np.concatenate((ct[0:1], _guard(ct[1:-1]), [ct[-1]]))  # Make the pts. on domain safe.
            # Literals aren't guaranteed at their guard points (beyond self.d).  So, if they're funny...
            # linear extrapolate to them---excursions beyond [0,1] allowed here---to preserve the *shape* on domain:
            if not isfinite(ct[0]):
                ct[0] = 2 * ct[1] - ct[2]
            if not isfinite(ct[-1]):
                ct[-1] = 2 * ct[-2] - ct[-3]
        return _Numerical(cd, cn, cv, ct, xv, xt, self.e)

    def t(self, v: float) -> float:
        """Returns the truth of a given value."""
        t = self.e
        xt = _t_for_v_in_xv(v, self.xv, self.xt)
        if xt is not None:
            return xt
        if self.uniform or (self.discrete is not None):
            if self.uniform and ((v - self.origin) / self.step).is_integer():
                t = self._sample(np.array([v]))[0]
            if (self.discrete is not None) and (v in self.discrete):
                t = self._sample(np.array([v]))[0]
        elif (self.d is not None) and self.d.contains(v):
            t = self._sample(np.array([v]))[0]
        return _guard(t)


class _Numerical(FuzzyNumber):  # user doesn't see it!  dataclass?

    def __init__(self, cd: Domain = None, cn: int = 0, cv: np.ndarray = None, ct: np.ndarray = None,
                 xv: np.ndarray = None, xt: np.ndarray = None, e: float = 0):
        """Args:
            cd: The domain over which the continuous numerical function is defined.  Valid if d[1]>=d[0].
            cn: The number of samples on  ``cd`` (not counting guard points). If cn>=0, no numerical function
                will be constructed. (And if cn=={1,2}, Literal will make it 3.)
            cv, ct: The continuous function's sample points: values, truths.  (Their len() must be equal.)
            xv, xt: The exceptional points: values, truths.   (Their len() must be equal.)
            e: The truth for all values not in ``cd`` or ``xv``---elsewhere.
            cv, xv: For each:  all elements must be unique.
            ct, xt, e:  For each:  all elements For each:  on [0,1] except for ct[0] and ct[-1] which should be finite.

            I'd check all of these, but this is private, so I'm trusting myself.  """
        self.cd = cd  # domain of continuous part
        if cn <= 0:
            cn, self.cd = 0, None
        self.cn = cn  # number of samples in  continuous part
        self.cv = cv  # continuous part: value samples
        self.ct = ct  # continuous part: truth samples
        self.xv = xv  # exceptional points: value samples
        self.xt = xt  # exceptional points: truth samples
        super().__init__(e)  # elsewhere

    def __str__(self):
        if self.cd is None:
            c = "no continuous part; "
        else:
            c = str(f"continuous part: {self.cn} points on domain {self.cd}, elsewhere: {self.e:.3g}; ")
        if self.xv is None:
            x = "no exceptional points."
        else:
            x = str(f"exceptional points: \n {np.dstack((self.xv, self.xt))[0]}")
        return str(f"_Numerical: {c} {x}.")

    def _get_domain(self, allowed_domain: Domain = None) -> Union[Domain, None]:
        if allowed_domain is None:
            return self.cd
        else:
            return self.cd.intersection(allowed_domain)

    def _get_numerical(self, precision: int, allowed_domain: Domain = None) -> _Numerical:
        """A :class:`._Numerical` will not resample itself, because that would lower its quality; therefore,
        it ignores the given precision, unless a precision of 0 insists on eliminating the continuous function.

        This method is designed called only on Literals and operands within Operators.
        Some subclasses of :class:`._Numerical` may be used as literals that define sets of points without
        a continuous function---e.g., :class:`.DPoints` and :class:`.Exactly`.  In these cases, this method's only
        task is to impose the allowed_domain by possibly discarding some exceptional points.  I don't foresee any
        situation where the parent _Numerical would be asked to return itself---no harm if it does, though.  """
        numerical = self if allowed_domain is None else _Numerical._impose_domain(self, allowed_domain)
        if precision == 0:
            numerical.cd = None
            numerical.cn = 0
            numerical.cv = None
            numerical.ct = None
        return numerical

    @staticmethod
    def _impose_domain(num: _Numerical, allowed_domain: Domain) -> _Numerical:
        """Returns a version of itself without exceptional points outside ``allowed_domain``.

        The continuous domain should have been restricted already by :meth:`._get_numerical` calls
        on operands---ultimately, these calls reach literals that will sample only the domain necessary to describe
        the allowed_domain of the caller.  In other words, discarding (xv, xt) points makes sense, but resampling
        or discarding (cv, ct) points would lower the quality, and needn't be done anyway.  So, here we only
        touch (xv, xt)."""
        new_xv, new_xt = num.xv, num.xt
        if num.xv is not None:
            i = np.where((num.xv < allowed_domain[0]))
            new_xv = np.delete(num.xv, i)
            new_xt = np.delete(num.xt, i)
            if new_xv is not None:
                i = np.where((new_xv > allowed_domain[1]))
                new_xv = np.delete(new_xv, i)
                new_xt = np.delete(new_xt, i)
            if len(new_xv) == 0:
                new_xv = new_xt = None
        return _Numerical(num.cd, num.cn, num.cv, num.ct, new_xv, new_xt, num.e)

    def _sample(self, v: Union[np.ndarray, float], interp: Interpolator = None) -> Union[np.ndarray, float]:
        """Given a value, return its truth"""
        f = isinstance(v, float)
        if self.cd is None:
            t = np.full_like(v, self.e)
        else:
            if interp is None:
                interp = default_interpolator
            t = interp.interpolate(v, self.cv, self.ct)
            outside = np.where((v < self.cv[0]) | (v > self.cv[-1]))
            t[outside] = self.e
            t = _guard(t)
        return float(t) if f else t

    def t(self, v: Union[np.ndarray, float], interp: Interpolator = None) -> Union[np.ndarray, float]:
        """Given a value, return its truth"""
        c_e = self._sample(v, interp)
        if self.xv is not None:
            x = np.where(np.in1d(v, self.xv, assume_unique=True))   # where v is in xv
            y = np.where(np.in1d(self.xv, v, assume_unique=True))   # where xv is in v
            c_e[x] = self.xt[y]
        return c_e    # t = _t_for_v_in_xv(v, self.xv, self.xt)

