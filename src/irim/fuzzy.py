"""
Contains the code for fuzzy logic and math. 
"""

import math
from abc import ABC, abstractmethod
from typing import Union

import numpy as np


# I had thought to include a type for fuzzy units ("fits"), reals guaranteed to be on [0,1], but
# that would seem to violate the spirit of weak typing, and might slow things down with fuzzy checks all the time.

# here are the functions for fuzzy logic: clip, and the Norm class, which provides not/and/or and and/or integrals.:


class Norm(ABC):
    """Norms define and provide the fuzzy logic operators."""

    @classmethod
    def define(cls, **kwargs):
        """A factory method to create Norms: It parses the kwargs and returns a subclass of Norm.
        If kwargs ==
            None:       pp
            n=a norm key: a simple norm of that kind
            n1, n2==norm keys, weight==[0,100]: a compound norm
            strictness==[-100,100]: a strictness norm
        the norm keys are---simple from least to most strict:
            'lx': Lax(),
            'mm': MinMax(),
            'hh': Hamacher(),
            'pp': ProductProbabilisticSum,
            'ee': Einstein(),
            'nn': Nilpotent(),
            'lb': LukasiewiczBoundedSum,
            'dd': Drastic(),
        and my one parameterized norm so far:
            'hhp': ParameterizedHamacher

        """
        if kwargs is None:
            n = ProductProbabilisticSum()
        elif "norm" in kwargs:
            n = cls.simple_factory(kwargs.get("norm"), kwargs.get("p"))
        elif "strictness" in kwargs:
            n = StrictnessNorm(kwargs.get("strictness"))
        elif "n1" in kwargs and "n2" in kwargs and "weight" in kwargs:
            n1 = cls.simple_factory(kwargs.get("n1"))
            n2 = cls.simple_factory(kwargs.get("n2"))
            w = kwargs.get("weight")
            n = CompoundNorm(n1, n2, w)
        else:
            n = ProductProbabilisticSum()
        return n

    @classmethod
    def simple_factory(cls, name: str, *args):
        if name == "lx":
            return Lax()
        elif name == "mm":
            return MinMax()
        elif name == "hh":
            return Hamacher()
        elif name == "pp":
            return ProductProbabilisticSum()
        elif name == "ee":
            return Einstein()
        elif name == "nn":
            return Nilpotent()
        elif name == "lb":
            return LukasiewiczBoundedSum()
        elif name == "dd":
            return Drastic()
        elif name == "hhp":
            return ParameterizedHamacher(float(args[0]))

    @classmethod
    def _operate(*args):
        """This parses the args of commutative binary operators (which may be fits or numpy arrays)
        and performs the operation on them appropriately, returning an array (if there were any)
        or a fit.  The fits and arrays operate with each other separately, then the final fit and
        array together.

        Args:
            args[1] is any commutative binary operator (a method defined by a subclass of Norm).
            The remaining args are a list of floats
            or numpy.ndarrays (which must all be of the same length)
            all of these valued on [0,1]."""
        operator = args[1]
        operands = args[2]
        fits = [f for f in operands if isinstance(f, float)]
        arrays = [a for a in operands if isinstance(a, np.ndarray)]
        n_fits = len(fits)
        n_arrays = len(arrays)
        if n_fits > 1:
            for i in range(1, n_fits):
                fits[0] = operator(fits[0], fits[i])
        if n_arrays > 1:
            for i in range(1, n_arrays):
                for j in range(len(arrays[0])):
                    arrays[0][j] = operator(arrays[0][j], arrays[i][j])
            if n_fits > 1:
                for j in range(len(arrays[0])):
                    arrays[0][j] = operator(arrays[0][j], fits[0])
        if n_arrays > 1:
            return arrays[0]
        else:
            return fits[0]

    def clip_(self, s: float) -> float:
        """Clips a number to the range [0,1], ensuring that it's a fuzzy unit."""
        c = max(min(s, 1), 0)
        if c != s:
            # raise Exception("Fuzzy units (fits) must be on [0,1].")  # TODO: type of exception?
            print("Fuzzy units (fits) must be on [0,1].")  # TODO: Should it stop execution as an exception?
        return c

    def not_(self, s: float) -> float:
        """The standard fuzzy negation."""
        # I don't anticipate using others:
        # this implies a one-to-one relation between t-norms and co-norms through DeMorgan's law.
        return 1 - s

    def and_(self, *args):
        """Fuzzy logic AND on any combination of floats and equal-length numpy.arrays valued on [0,1]."""
        return self._operate(self._and, args)

    def or_(self, *args):
        """Fuzzy logic OR on any combination of floats and equal-length numpy.arrays valued on [0,1]."""
        return self._operate(self._or, args)

    @abstractmethod
    def _and(self, a: float, b: float) -> float:
        """Private definition of fuzzy logic AND that subclasses must implement."""

    @abstractmethod
    def _or(self, a: float, b: float) -> float:
        """Private definition of fuzzy logic OR that subclasses must implement."""

    # Here I'll define And and Or integrals (like and-ing or or-ing every point of a function together).
    # I only need the or_integral for fuzzy arithmetic, but I'm defining the and_integral for completeness.
    # many of these implementations will require:
    #     s = np.trapz(z) / line_length                          # definite line integral
    #     p = math.exp(np.trapz(np.log(z))) / line_length     # definite geometric (product) line integral
    # ---the definite (Riemann, geometric) integrals over some line on a function
    # (fuzzy _and_ (t-norm), in practice) of the Cartesian product of two fuzzy values.
    # They must always be divided by their line_length so they have the same metric (this is not true
    # for min/max operators, because extrema aren't diminished by rarity).  For the same reason,
    # the units of line_length should always be the sample interval on either edge of the Cartesian product.

    @abstractmethod
    def and_integral(self, z: np.ndarray, line_length: float) -> float:
        """Subclasses must define the fuzzy logic and integral here.
        Args:
            z = an array of suitabilites (on [0,1]) vs. uniformly-spaced values
            line_length = arclength of the line over which the definite integral is to be taken,
                in units of sample intervals of the Cartesian product."""

    @abstractmethod
    def or_integral(self, z: np.ndarray, line_length: float) -> float:
        """Subclasses must define the fuzzy logic or integral here.
        Args:
            z = an array of suitabilites (on [0,1]) vs. uniformly-spaced values
            line_length = arclength of the line over which the definite integral is to be taken,
                in units of sample intervals of the Cartesian product."""


class SimpleNorm(Norm):
    """These are created without arguments."""


# Here I'll implement the SimpleNorms from least to most strict (strong and, weak or to the reverse):
#   I would provide more t-norms (Schweizer, Frank, Yager, Aczél–Alsina, Dombi, Sugeno–Weber, etc.),
#   but I don't know how to interpret them continuously (which is necessary for the fuzzy arithmetic)---
#   I only know how to do Riemann and geometric (product) integrals.  That should be plenty!


class Lax(SimpleNorm):
    """(lx) Defines the lax t-norm/co-norm pair
        (my own invention, the opposite of drastic, fodder for CompoundNorms)."""

    def _and(self, a: float, b: float) -> float:
        if a == 0:
            r = b
        elif b == 0:
            r = a
        else:
            r = 1
        return r

    def _or(self, a: float, b: float) -> float:
        if a == 1:
            r = b
        elif b == 1:
            r = a
        else:
            r = 0
        return r

    def and_integral(self, z: np.ndarray, line_length: float) -> float:
        return np.amax(z) if np.amin(z) == 0 else 1

    def or_integral(self, z: np.ndarray, line_length: float) -> float:
        return np.amin(z) if np.amax(z) == 1 else 0


class MinMax(SimpleNorm):
    """(mm) Defines the Gödel-Zadeh (minimum / maximum) t-norm/co-norm pair."""

    def _and(self, a: float, b: float) -> float:
        return min(a, b)

    def _or(self, a: float, b: float) -> float:
        return max(a, b)

    def and_integral(self, z: np.ndarray, line_length: float) -> float:
        return np.amin(z)

    def or_integral(self, z: np.ndarray, line_length: float) -> float:
        return np.amax(z)


class Hamacher(SimpleNorm):
    """(hh) Defines the Hamacher product / sum t-norm/co-norm pair."""

    def _and(self, a: float, b: float) -> float:
        return 0 if a == b == 0 else a * b / (a + b - a * b)  # Could get +inf near a==b==0?

    def _or(self, a: float, b: float) -> float:
        return 1 if a == b == 1 else (a + b + - 2 * a * b) / (1 - a * b)

    def and_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z) / line_length
        p = math.exp(np.trapz(np.log(z))) / line_length
        return 0 if s == p else (p / (s - p))

    def or_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z) / line_length
        p = math.exp(np.trapz(np.log(z))) / line_length
        return s / (1 + p)


class ProductProbabilisticSum(SimpleNorm):
    """(pp) Defines the Goguen (product / probabilistic sum) t-norm/co-norm pair."""

    def _and(self, a: float, b: float) -> float:
        return a * b

    def _or(self, a: float, b: float) -> float:
        return a + b - a * b

    def and_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z)  # definite integral
        p = math.exp(np.trapz(np.log(z)))  # definite geometric (product) integral
        return p / line_length

    def or_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z)
        p = math.exp(np.trapz(np.log(z)))
        return (s - p) / line_length


class Einstein(SimpleNorm):
    """(ee) Defines the Einstein product / sum t-norm/co-norm pair."""

    def _and(self, a: float, b: float) -> float:
        return a * b / (a * b - a - b + 2)

    def _or(self, a: float, b: float) -> float:
        return (a + b) / (1 + a * b)

    def and_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z) / line_length
        p = math.exp(np.trapz(np.log(z))) / line_length
        return s / (p - s + 2)

    def or_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z) / line_length
        p = math.exp(np.trapz(np.log(z))) / line_length
        return s / (1 + p)


class Nilpotent(SimpleNorm):
    """(nn) Defines the Kleene-Dienes (nilpotent minimum / maximum) t-norm/co-norm pair."""

    def _and(self, a: float, b: float) -> float:
        return min(a, b) if a + b > 1 else 0

    def _or(self, a: float, b: float) -> float:
        return max(a, b) if a + b < 1 else 1

    def and_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z) / line_length
        return np.amin(z) if s > 1 else 0

    def or_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z) / line_length
        return np.amax(z) if s < 1 else 1


class LukasiewiczBoundedSum(SimpleNorm):
    """(lb) Defines the Łukasiewicz / bounded sum t-norm/co-norm pair."""

    def _and(self, a: float, b: float) -> float:
        return max(0.0, a + b - 1)

    def _or(self, a: float, b: float) -> float:
        return min(a + b, 1)

    def and_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z) / line_length
        return max(0, s - 1)

    def or_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z) / line_length
        return min(s, 1)


class Drastic(SimpleNorm):
    """(dd) Defines the drastic t-norm/co-norm pair."""

    def _and(self, a: float, b: float) -> float:
        if a == 1:
            r = b
        elif b == 1:
            r = a
        else:
            r = 0
        return r

    def _or(self, a: float, b: float) -> float:
        if a == 0:
            r = b
        elif b == 0:
            r = a
        else:
            r = 1
        return r

    def and_integral(self, z: np.ndarray, line_length: float) -> float:
        return np.amin(z) if np.amax(z) == 1 else 0

    def or_integral(self, z: np.ndarray, line_length: float) -> float:
        return np.amax(z) if np.amin(z) == 0 else 1


class ParameterizedNorm(SimpleNorm):
    """The __init__ method of its subclasses should take user parameter(s) on [0,100] or [-100,100]
    and map them onto whatever their t-norms/co-norms require."""


class ParameterizedHamacher(ParameterizedNorm):
    """(hhp) Defines the parameterized version of the Hamacher product / Einstein sum t-norm/co-norm pair.

    The user parameter is expected to be on [0,100] and must be >=0 (it will be clipped if it is not).
        p ==   0 == Hamacher / Einsten sum (he),
        p ==  50 == product / bounded sum (pp),
        p == 100 == drastic (dd).
    """

    def __init__(self, user_parameter=0.0):
        user_parameter = max(user_parameter, 0)
        self._p = 0 if user_parameter < 0 else 10 ** (.06 * user_parameter - 3)  # .18, 9
        # TODO: get this mapping right

    def _and(self, a: float, b: float) -> float:
        return 0 if self._p == a == b == 0 else a * b / (self._p + (1 - self._p) * (a + b - a * b))

    def _or(self, a: float, b: float) -> float:
        return (a + b + (self._p - 2) * a * b) / (1 + (self._p - 1) * a * b)

    def and_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z) / line_length
        p = math.exp(np.trapz(np.log(z))) / line_length
        return 0 if self._p == s == 0 else p / (self._p + (1 - self._p) * (s - p))

    def or_integral(self, z: np.ndarray, line_length: float) -> float:
        s = np.trapz(z) / line_length
        p = math.exp(np.trapz(np.log(z))) / line_length
        return (s - self._p * p) / (1 + (1 - self._p) * p)


class CompoundNorm(Norm):
    """Returns a linear combination of the results of two other norms according to a weight on [0,100].

    Args:
        n1, n2: two SimpleNorms or their names (2--3 letter code)
        w: a parameter on [0,100].  If the strictness of n2>n1, the w may be taken as a strictness parameter."""

    def __init__(self, n1: Union[SimpleNorm, str], n2: Union[SimpleNorm, str], w: float):
        self._n1 = Norm.simple_factory(n1) if isinstance(n1, str) else n1
        self._n2 = Norm.simple_factory(n2) if isinstance(n2, str) else n2
        self._w = w

    def combination(self, r1: float, r2: float) -> float:
        return ((100 - self._w) * r1 + self._w * r2) / 100

    def _and(self, a: float, b: float) -> float:
        return self.combination(self._n1.and_(a, b), self._n2.and_(a, b))

    def _or(self, a: float, b: float) -> float:
        return self.combination(self._n1.or_(a, b), self._n2.or_(a, b))

    def and_integral(self, z: np.ndarray, line_length: float) -> float:
        return self.combination(self._n1.and_integral(z, line_length), self._n2.and_integral(z, line_length))

    def or_integral(self, z: np.ndarray, line_length: float) -> float:
        return self.combination(self._n1.or_integral(z, line_length), self._n2.or_integral(z, line_length))


class StrictnessNorm(Norm):
    """Provides a norm of a given strictness, on a scale from [-100,100].

        The provided norm pairs are, in increasing strictness, in proportion to volume of the unit cube:
        -100    lx  lax
        -75.69  mm  minimum / maximum (Gödel, Zadeh)
        -49.78  hh  Hamacher product / sum
        -22.42  pp  product / probabilistic sum (Goguen)
          5.00  ee  Einstein product / sum
         33.20  nn  nilpotent minimum / maximum (Kleene-Dienes)
         63.63  lb  Łukasiewicz  / bounded sum
        100     dd  drastic t-norm / co-norm"""

    def __new__(cls, strictness: float):
        strictness = max(min(strictness, 100), -100)
        name = ["lx","mm","hh","pp","ee","nn","lb","dd"]
        x = [-100.0, -75.0, -35.0, -5, 10.00, 50.0, 75.0, 100.0]
        y = np.arange(0, 8)
        w = np.interp(strictness, x, y)
        n = CompoundNorm(name[math.floor(w)], name[math.ceil(w)], 100*(w%1))
        return n


# here are the classes for fuzzy value and arithmetic:
# Value --- base class;  Numerical --- "working" class of evaluation and defuzzification;
# Triangle, Trapezoid, Bell, Cauchy, Gauss, Points --- "atoms" defining a value;
# ValueNot, ValueAnd, ValueOr --- logic on values;
# Sum, Difference, Product, Quotient, Focus, Abs, Inverse --- arithmetic on values.

# class Value(ABC):
#     """Represents a generally fuzzy real number (as a function of suitability (on [0,1]) vs. value).
#     It may be obtained (defuzzified) as a crisp value, along with that value's suitability.
#     """
#     continuous_domain = None
#
#     @property
#     def continuous_domain(self):
#         return self._continuous_domain
#
#     @continuous_domain.setter
#     def continuous_domain(self, x):
#         self._continuous_domain = x
#
#     #
#     # def __init__(continuous_domain):
#     #     self.continuous_domain = continuous_domain
#
#     @abstractmethod
#     def evaluate(self, resolution: float):  # -> Numerical
#         """Obtains and returns a numerical representation of itself.
#         This is where the work is done in each subclass."""
#         return Numerical(self.continuous_domain, resolution)
#
#     def get(self, extreme_domain: (float, float), resolution: float, defuzzification_parameters) -> (float, float):
#         """Returns a crisp value that is equivalent to its (generally) fuzzy value,
#         along with that crisp value's suitability (a measure, on [0,1], of how good its answer is).
#
#         Returns: (v, s)
#             v: The crisp equivalent of this fuzzy number, a float.
#             s: The suitability (appropriateness, quality, truth, certainty) of v as an equivalent, a float on [0,1].
#
#         Arguments:
#             extreme_domain: bounds the result in case the answer must be limited,
#                 e.g., if tempo must be on [30, 200] bpm, or a parameter must be on [0,100].
#             resolution: the maximum distance between values that will be considered in the numerical representation.
#                 This controls the accuracy of the result (a smaller resolution is better).
#                 Also, consider that a coarse mesh in the numerical representation might miss narrow peaks.
#             defuzzification_parameters:  I don't know what they are yet.
#         """
#         # Evaluate, impose an extreme domain, defuzzify.  This is where it's implemented for every subclass.
#         numerical = self.evaluate(resolution)
#         # impose an extreme domain
#         numerical.impose_domian(extreme_domain)
#         # defuzzify.
#         v = s = 0  # dummy
#         return v, s
#
#
# class Numerical(Value):
#     """The numerical representation of a (generally) fuzzy value.
#
#     To represent the suitabilities of all real numbers, it uses:
#         * One Numpy array s(v) for one continuous domain [a, b].
#         * A set of exceptional points {(v_i, s_i)} that always override the array.
#         * An out-of-bounds suitability, s_0, that is assumed for any value otherwise undefined.
#
#         Is this where the defuzzify method is implemented?  Probably."""
#
#     def __init__(domain: (float, float), resolution: float):
#         """Initialization prepares the continuous domain with sample points that are
#         integer multiples of the resolution (so that all arrays in the calculation will line up),
#         covering the stated domain plus guard points on either end (for future interpolation).
#         So,  conveniently, subclasses perform a function on this array to sample themselves.
#
#         The set of exceptional points (the discrete domain) is an empty 2D array of value, suitability pairs.
#         Otherwise undefined points in the domain default to a suitability of 0.
#
#         Args:
#             domain: values over which the continuous domain will be defined.
#             resolution: the separation between sample points (a smaller resolution is better).
#         """
#         self.continuous_domain = domain
#         v_0 = math.floor(domain[0] / resolution) - 1
#         v_n = math.ceil(domain[1] / resolution) + 1
#         number_of_samples = v_n - v_0 + 1
#         v_0, v_n = v_0 * resolution, v_n * resolution
#         # sample points on the continuous domain, to be filled with s(v) by subclasses:
#         self.continuous_v = np.linspace(v_0, v_n, number_of_samples)
#         self.continuous_s = np.linspace(v_0, v_n, number_of_samples)
#         # the discrete domain, to be filled as v,s by subclasses:
#         self.exceptional_points = np.empty((2, 0))
#         # the suitability elsewhere, outside the defined domains
#         self.out_of_bounds = 0
#
#     def suitability(self, value: float):
#         """Returns the suitability of a given value, as defined by this fuzzy value.
#
#         The exceptional points of the discrete domain override the definition of the continuous domain,
#         which is generally found by interpolation.  Points outside these domains return a default value."""
#         discrete = np.where(value == self.exceptional_points[0], self.exceptional_points[1])
#         if discrete is not None:
#             return discrete
#         else:
#             if value < self.continuous_domain[0] or value > self.continuous_domain[1]:
#                 return self.out_of_bounds
#             else:
#                 return np.interp(value, self.continuous_v, self.continuous_s)
#
#     def evaluate(self, resolution: float):  # -> Numerical
#         """It returns itself because it is the evaluation.
#
#         In any other subclass of Value, this is where the work would be done."""
#         return self
#
#     def impose_domian(self, imposed_domain: (float, float)):
#         """Discard any defined suitabilites <a | >b."""
#         self.exceptional_points = np.where(self.exceptional_points[0] > imposed_domain[0] and \
#                                            self.exceptional_points[0] < imposed_domain[1], self.exceptional_points)
#         self.continuous_s = np.where(self.continuous_v > imposed_domain[0] and \
#                                      self.continuous_v < imposed_domain[1], self.continuous_s)
#         self.continuous_v = np.where(self.continuous_v > imposed_domain[0] and \
#                                      self.continuous_v < imposed_domain[1], self.continuous_v)
#
#     def defuzzify(self) -> (float, float):
#         v = s = 0  # dummy  I don't know all the methods yet, but I prefer median of global maxima.
#         return v, s
#
#
# # Triangle, Trapezoid, Bell, Cauchy, Gauss, Points --- "atoms" defining a value;
# # ValueNot, ValueAnd, ValueOr --- logic on values;
# # Sum, Difference, Product, Quotient, Focus, Abs, Inverse --- arithmetic on values.
#
# class Triangle(Value):
#     """Describes a fuzzy number as a trianglular function with a peak (maximum s) and extreme limits (s==0)"""
#
#     def __init__(peak, domain):
#         """Args:
#             peak:
#                 float: (most suitable) value , assumes s=1
#                 (float, float): (value, suitability)
#             domain:
#                 float:  HWHM
#                 (float, float): extreme domain where s>0"""
#
#         if isinstance(peak, (float, int)):  # assume its suitability = 1
#             self.peak = (peak, 1.0)
#         else:
#             self.peak = peak
#         if isinstance(domain, (float, int)):  # assume it is the HWFM about the peak
#             self.continuous_domain = (self.peak[0] - domain, self.peak[0] + domain)
#         else:
#             self.continuous_domain = domain  # shouldn't I check these things?
#
#     def evaluate(self, resolution: float):  # rethink this
#         n = Numerical(self.continuous_domain, resolution)
#         a_left = self.peak[1] / (self.peak[0] - self.continuous_domain[0])
#         a_right = self.peak[1] / (self.peak[0] - self.continuous_domain[1])
#         d = n.continuous_v  # - self.peak[0]
#
#         s = n.continuous_s
#         s = np.piecewise(d, [d < self.continuous_domain[0],
#                              (d > self.continuous_domain[0]) and (d < self.peak[0]),
#                              (d >= self.peak[0]) and (d < self.continuous_domain[1]),
#                              d > self.continuous_domain[1]],
#                          [lambda d: 0, lambda d: 1, lambda d: 2, lambda d: 3])
#         print(d)
#         # n.continuous_s =[lambda d: 0, lambda d: a_left*d, lambda d: 1-a_right*d, lambda d: 0]
#         print(s)
#         return n


# Here is where I am testing or playing around or something.

# a = .1
# b = .1
# n = Norm.define()
# print(f"__:  not: {n.not_(a)}, and: {n.and_(a, b)},  or: {n.or_(a, b)}")


# lx, mm, hh,  pp,ee,lb<->>nn,dd
# l = Norm.define(norm="dd")
# h = Norm.define(norm="lb")
#
# plt.style.use('_mpl-gallery-nogrid')
#
# # make data  strictness
# X, Y = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
# Z = np.empty((256, 256))
# for x in range(0, 256):
#     for y in range(0, 256):
#         Z[x][y] = h.and_(X[x, y], Y[x, y]) - l.and_(X[x, y], Y[x, y])
# levels = np.linspace(Z.min(), Z.max(), 20)
#
# # plot
# # plot
# fig, ax = plt.subplots()
#
# ax.contourf(X, Y, Z, levels=levels, cmap=plt.cm.viridis)
# # Plot the surface
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot_surface(X, Y, Z, vmin=Z.min() * 2, cmap=cm.PuOr)
#
# plt.show()


# This animates the norms (by strictness) for inspection.

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set the figure size
plt.rcParams["figure.figsize"] = [5.00, 5.00]
plt.rcParams["figure.autolayout"] = True

# data for the contour plot
data = np.empty((64, 64, 200))


for t in range(0, 100):
    n = Norm.define(strictness=t*2-100)
    for x in range(0, 64):
        for y in range(0, 64):
            data[x, y, t ] = n.and_(x / 63, y / 63)

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Method to change the contour data points
def animate(i):
    ax.clear()
    ax.contourf(data[:, :, i], 64, cmap='plasma')


ani = animation.FuncAnimation(fig, animate, 100, interval=10, blit=False)

plt.show()

# This plots the norms for examination.

import matplotlib.pyplot as plt
nlist = ["lx","mm","hh","pp","ee","nn","lb","dd"]
norm = []*8
data = []*8
ax = [plt.Figure()]*8


fig, ((ax[0], ax[1], ax[2], ax[3]), (ax[4], ax[5], ax[6], ax[7])) = plt.subplots(nrows=2, ncols=4, sharey="all")

for i in range(0,8):
    ax[i].set_aspect('equal', adjustable='box')

sa, sb = 10,75

for i in range(0, 8):
    st = sa + i*(sb-sa)/7
    # norm.append(Norm.define(norm=nlist[i]))
    n = Norm.define(strictness=st)
    data.append(np.empty((32, 32)))
    plt.axis('square')
    for x in range(0, 32):
        a = x / 31
        for y in range(0, 32):
            b = y / 31
            data[i][x, y] = n.or_(a, b)
    ax[i].contourf(data[i], 16, cmap='plasma')

plt.show()

# total = []
# sum, sumlast = 0, 0
# for i in range(-100,100):
#     n = Norm.define(strictness=i)
#     total.append(sum)
#     print(f"strictness: {i}, total: {sum}: increase: {sum-sumlast}")
#     sumlast = sum
#     sum = 0
#     for x in range(0, 32):
#         a = x / 31
#         for y in range(0, 32):
#             b = y / 31
#             sum += n.or_(a, b)



# sum, sumlast = 0, 0
# for i in range(0,8):
#     n = Norm.define(norm=nlist[i])
#     print(f"norm: {nlist[i]}, total: {sum}, increase: {150*(sum-sumlast)/160}")
#     sumlast = sum
#     sum = 0
#     for x in range(0, 32):
#         a = x / 31
#         for y in range(0, 32):
#             b = y / 31
#             sum += n.or_(a, b)
#
# print(f"norm: {nlist[i]}, total: {sum}, increase: {150*(sum-sumlast)/179}")

# here is a change.

