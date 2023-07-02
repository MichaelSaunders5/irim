"""
Contains the code for fuzzy logic and math. 
"""

import math
from abc import ABC, abstractmethod

import numpy
import numpy as np


# I had thought to include a type for fuzzy units ("fits"), reals guaranteed to be on [0,1], but
# that would seem to violate the spirit of weak typing, and might slow things down with fuzzy checks all the time.

# here are the functions for fuzzy logic: clip, and the Norm class, which provides not/and/or and and/or integrals.:

def clip(s: float) -> float:
    """Clips a number to the range [0,1], ensuring that it's a fuzzy unit."""
    c = max(min(s, 1), 0)
    if c != s:
        # raise Exception("Fuzzy units (fits) must be on [0,1].")  # TODO: type of exception? Does it stop execution?
        print("Fuzzy units (fits) must be on [0,1].")  # TODO: type of exception? Should it stop execution?
    return c


class NormBase(ABC):
    """Norms, created by this abstract factory, define and provide the fuzzy logic operators."""

    def not_(self, s: float) -> float:
        """The standard fuzzy negation."""
        # I don't anticipate using others:
        # this implies a one-to-one relation between t-norms and co-norms through DeMorgan's law.
        return 1 - s

    @abstractmethod
    def and_(self, a: float, b: float) -> float:
        """Subclasses must define the fuzzy logic and here."""

    @abstractmethod
    def or_(self, a: float, b: float) -> float:
        """Subclasses must define the fuzzy logic or here."""

    # Here I'll define And and Or integrals (like and-ing or or-ing every point of a function together).
    # I only need the or_integral for fuzzy arithmetic, but I'm defining the and_integral for completeness.
    # many of these implementations will require:
    #     s = numpy.trapz(z) / line_length                          # definite line integral
    #     p = math.exp(numpy.trapz(numpy.log(z))) / line_length     # definite geometric (product) line integral
    # ---the definite (Riemann, geometric) integrals over some line on a function
    # (fuzzy _and_ (t-norm), in practice) of the Cartesian product of two fuzzy values.
    # They must always be divided by their line_length so they have the same metric (this is not true
    # for min/max operators, because extrema aren't diminished by rarity).  For the same reason,
    # the units of line_length should always be the sample interval on either edge of the Cartesian product.

    @abstractmethod
    def and_integral(self, z: numpy.ndarray, line_length: float) -> float:
        """Subclasses must define the fuzzy logic and integral here.
        Args:
            z = an array of suitabilites (on [0,1]) vs. uniformly-spaced values
            line_length = arclength of the line over which the definite integral is to be taken,
                in units of sample intervals of the Cartesian product."""

    @abstractmethod
    def or_integral(self, z: numpy.ndarray, line_length: float) -> float:
        """Subclasses must define the fuzzy logic or integral here.
        Args:
            z = an array of suitabilites (on [0,1]) vs. uniformly-spaced values
            line_length = arclength of the line over which the definite integral is to be taken,
                in units of sample intervals of the Cartesian product."""


class Norm:
    """A factory to create NormBases (Norms)"""

    def __new__(cls, **kwargs):
        """Parse the kwargs and return a subclass of NormBase.
        If kwargs ==
            None:       pp
            n=a norm key: a simple norm of that kind
            n1, n2==norm keys, weight==[0,100]: a compound norm
            strictness==[-100,100]: a strictness norm
        the norm keys are---simple from least to most strict:
            'lx': Lax(),
            'mm': MinMax(),
            'he': HamacherEinstein(),
            'pp': ProductProbabilisticSum,
            'lb': LukasiewiczBoundedSum,
            'nn': Nilpotent(),
            'dd': Drastic(),
        and my one parameterized norm so far:
            'hep': ParameterizedHamacherEinstein

        """
        if kwargs is None:
            n = ProductProbabilisticSum()
        elif "norm" in kwargs:
            n = NORMS.get(kwargs.get("norm"))
            if  "p" in kwargs and kwargs.get("n")[2]=="p":
                n = n(kwargs.get("p"))
        elif "strictness" in kwargs:
            n = StrictnessNorm(kwargs.get("strictness"))
        elif "n1" in kwargs and "n2" in kwargs and "weight" in kwargs:
            n1 = NORMS.get(kwargs.get("n1"))
            n2 = NORMS.get(kwargs.get("n2"))
            w = kwargs.get("weight")
            n = CompoundNorm(n1,n2,w)
        else:
            n = ProductProbabilisticSum()
        print(n)
        return n


        # TODO: create versions that add this if kwargs["clipping"] right?:
        # s = clip(s) if self.clipping else s
        # if self.clipping:
        #     a = clip(a)
        #     b = clip(b)
        # numpy.ndarry:  normalize to [0,1] or! numpy.clip(v, 0, 1)




class SimpleNorm(NormBase):
    """These are created without arguments."""


# Here I'll implement the SimpleNorms from least to most strict (strong and, weak or to the reverse):
#   I would provide more t-norms (Schweizer, Frank, Yager, Aczél–Alsina, Dombi, Sugeno–Weber, etc.),
#   but I don't know how to interpret them continuously (which is necessary for the fuzzy arithmetic)---
#   I only know how to do Riemann and geometric (product) integrals.  That should be plenty!


class Lax(SimpleNorm):
    """(lx) Defines the lax t-norm/co-norm pair
        (my own invention, the opposite of drastic, fodder for CompoundNorms)."""

    def and_(self, a: float, b: float) -> float:
        return 0 if a == b == 0 else 1

    def or_(self, a: float, b: float) -> float:
        return 1 if a == b == 1 else 0

    def and_integral(self, z: numpy.ndarray, line_length: float) -> float:
        return 0 if numpy.amax(z) == 0 else 1

    def or_integral(self, z: numpy.ndarray, line_length: float) -> float:
        return 1 if numpy.amin(z) == 1 else 0


class MinMax(SimpleNorm):
    """(mm) Defines the Gödel-Zadeh (minimum / maximum) t-norm/co-norm pair."""

    def and_(self, a: float, b: float) -> float:
        return min(a, b)

    def or_(self, a: float, b: float) -> float:
        return max(a, b)

    def and_integral(self, z: numpy.ndarray, line_length: float) -> float:
        return numpy.amin(z)

    def or_integral(self, z: numpy.ndarray, line_length: float) -> float:
        return numpy.amax(z)


class HamacherEinstein(SimpleNorm):
    """(he) Defines the Hamacher product / Einstein sum t-norm/co-norm pair."""

    def and_(self, a: float, b: float) -> float:
        return 0 if a == b == 0 else a * b / (a + b - a * b)  # Could get +inf near a==b==0?

    def or_(self, a: float, b: float) -> float:
        return (a + b) / (1 + a * b)

    def and_integral(self, z: numpy.ndarray, line_length: float) -> float:
        s = numpy.trapz(z) / line_length
        p = math.exp(numpy.trapz(numpy.log(z))) / line_length
        return 0 if s == p else (p / (s - p))

    def or_integral(self, z: numpy.ndarray, line_length: float) -> float:
        s = numpy.trapz(z) / line_length
        p = math.exp(numpy.trapz(numpy.log(z))) / line_length
        return s / (1 + p)


class ProductProbabilisticSum(SimpleNorm):
    """(pp) Defines the Goguen (product / probabilistic sum) t-norm/co-norm pair."""

    def and_(self, a: float, b: float) -> float:
        return a * b

    def or_(self, a: float, b: float) -> float:
        return a + b - a * b

    def and_integral(self, z: numpy.ndarray, line_length: float) -> float:
        s = numpy.trapz(z)  # definite integral
        p = math.exp(numpy.trapz(numpy.log(z)))  # definite geometric (product) integral
        return p / line_length

    def or_integral(self, z: numpy.ndarray, line_length: float) -> float:
        s = numpy.trapz(z)
        p = math.exp(numpy.trapz(numpy.log(z)))
        return (s - p) / line_length


class LukasiewiczBoundedSum(SimpleNorm):
    """(lb) Defines the Łukasiewicz / bounded sum t-norm/co-norm pair."""

    def and_(self, a: float, b: float) -> float:
        return max(0.0, a + b - 1)

    def or_(self, a: float, b: float) -> float:
        return min(a + b, 1)

    def and_integral(self, z: numpy.ndarray, line_length: float) -> float:
        s = numpy.trapz(z) / line_length
        return max(0, s - 1)

    def or_integral(self, z: numpy.ndarray, line_length: float) -> float:
        s = numpy.trapz(z) / line_length
        return min(s, 1)


class Nilpotent(SimpleNorm):
    """(nn) Defines the Kleene-Dienes (nilpotent minimum / maximum) t-norm/co-norm pair."""

    def and_(self, a: float, b: float) -> float:
        return min(a, b) if a + b > 1 else 0

    def or_(self, a: float, b: float) -> float:
        return max(a, b) if a + b < 1 else 1

    def and_integral(self, z: numpy.ndarray, line_length: float) -> float:
        s = numpy.trapz(z) / line_length
        return numpy.amin(z) if s > 1 else 0

    def or_integral(self, z: numpy.ndarray, line_length: float) -> float:
        s = numpy.trapz(z) / line_length
        return numpy.amax(z) if s < 1 else 1


class Drastic(SimpleNorm):
    """(dd) Defines the drastic t-norm/co-norm pair."""

    def and_(self, a: float, b: float) -> float:
        if a == 1:
            r = b
        elif b == 1:
            r = a
        else:
            r = 0
        return r

    def or_(self, a: float, b: float) -> float:
        if a == 0:
            r = b
        elif b == 0:
            r = a
        else:
            r = 1
        return r

    def and_integral(self, z: numpy.ndarray, line_length: float) -> float:
        return numpy.amin(z) if numpy.amax(z) == 1 else 0

    def or_integral(self, z: numpy.ndarray, line_length: float) -> float:
        return numpy.amax(z) if numpy.amin(z) == 0 else 1


class ParameterizedNorm(SimpleNorm):
    """The __init__ method of its subclasses should take user parameter(s) on [0,100] or [-100,100]
    and map them onto whatever their t-norms/co-norms require."""


class ParameterizedHamacherEinstein(ParameterizedNorm):
    """(hep) Defines the parameterized version of the Hamacher product / Einstein sum t-norm/co-norm pair.

    The user parameter is expected to be on [0,100] and must be >=0 (it will be clipped if it is not).
        p ==   0 == Hamacher / Einsten sum (he),
        p ==  50 == product / bounded sum (pp),
        p == 100 == drastic (dd).
    """

    def __init__(self, user_parameter=0):
        user_parameter = max(user_parameter, 0)
        self._p = 0 if user_parameter < 0 else 10 ** (.18 * user_parameter - 9)

    def and_(self, a: float, b: float) -> float:
        return 0 if self._p == a == b == 0 else a * b / (self._p + (1 - self._p) * (a + b - a * b))

    def or_(self, a: float, b: float) -> float:
        return (a + b - self._p * a * b) / (1 + (1 - self._p) * a * b)

    def and_integral(self, z: numpy.ndarray, line_length: float) -> float:
        s = numpy.trapz(z) / line_length
        p = math.exp(numpy.trapz(numpy.log(z))) / line_length
        return 0 if self._p == s == 0 else p / (self._p + (1 - self._p) * (s - p))

    def or_integral(self, z: numpy.ndarray, line_length: float) -> float:
        s = numpy.trapz(z) / line_length
        p = math.exp(numpy.trapz(numpy.log(z))) / line_length
        return (s - self._p * p) / (1 + (1 - self._p) * p)


class CompoundNorm(NormBase):
    """Returns a linear combination of the results of two other norms according to a weight on [0,100].

    Args:
        n1, n2: two SimpleNorms
        w: a parameter on [0,100].  If the strictness of n2>n1, the w may be taken as a strictness parameter."""

    def __init__(self, n1: SimpleNorm, n2: SimpleNorm, w: float):
        self._n1 = n1
        self._n2 = n2
        self._w = w

    def combination(self, r1: float, r2: float) -> float:
        return ((100 - self._w) * r1 + self._w * r2) / 100

    def and_(self, a: float, b: float) -> float:
        return self.combination(self._n1.and_(a, b), self._n2.and_(a, b))

    def or_(self, a: float, b: float) -> float:
        return self.combination(self._n1.or_(a, b), self._n2.or_(a, b))

    def and_integral(self, z: numpy.ndarray, line_length: float) -> float:
        return self.combination(self._n1.and_integral(z, line_length), self._n2.and_integral(z, line_length))

    def or_integral(self, z: numpy.ndarray, line_length: float) -> float:
        return self.combination(self._n1.or_integral(z, line_length), self._n2.or_integral(z, line_length))


class StrictnessNorm(NormBase):
    """Provides a norm of a given strictness, on a scale from [-100,100].

        The provided norm pairs are, in increasing strictness:
        -100 lx  lax
        -90  mm  minimum / maximum (Gödel, Zadeh)
        -50  he  Hamacher product / Einstein sum
          0  pp  product / probabilistic sum (Goguen) (This is always the default!)
         50  lb  Łukasiewicz  / bounded sum
         90  nn  nilpotent minimum / maximum (Kleene-Dienes)
        100  dd  drastic t-norm / co-norm"""

    def __new__(cls, strictness: float):
        strictness = max(min(strictness, 100), -100)
        if strictness < -90:
            n = CompoundNorm(Lax(), MinMax(), (strictness + 100) * 10)
        elif strictness < -50:
            n = CompoundNorm(MinMax(), HamacherEinstein(), (strictness + 90) * 2.5)
        elif strictness < 0:
            n = CompoundNorm(HamacherEinstein(), ProductProbabilisticSum(), (strictness + 50) * 2)
        elif strictness < 50:
            n = CompoundNorm(ProductProbabilisticSum(), LukasiewiczBoundedSum(), strictness * 2)
        elif strictness < 90:
            n = CompoundNorm(LukasiewiczBoundedSum(), Nilpotent(), (strictness - 50) * 2.5)
        else:
            n = CompoundNorm(Nilpotent(), Drastic(), (strictness - 90) * 10)
        return n


NORMS = {'lx': Lax(),
         'mm': MinMax(),
         'he': HamacherEinstein(),
         'pp': ProductProbabilisticSum,
         'lb': LukasiewiczBoundedSum,
         'nn': Nilpotent(),
         'dd': Drastic(),
         'hep': ParameterizedHamacherEinstein()}


# here are the classes for fuzzy value and arithmetic:
# Value --- base class;  Numerical --- "working" class of evaluation and defuzzification;
# Triangle, Trapezoid, Bell, Cauchy, Gauss, Points --- "atoms" defining a value;
# ValueNot, ValueAnd, ValueOr --- logic on values;
# Sum, Difference, Product, Quotient, Focus, Abs, Inverse --- arithmetic on values.

class Value(ABC):
    """Represents a generally fuzzy real number (as a function of suitability (on [0,1]) vs. value).
    It may be obtained (defuzzified) as a crisp value, along with that value's suitability.
    """
    continuous_domain = None

    @property
    def continuous_domain(self):
        return self._continuous_domain

    @continuous_domain.setter
    def continuous_domain(self, x):
        self._continuous_domain = x

    #
    # def __init__(self, continuous_domain):
    #     self.continuous_domain = continuous_domain

    @abstractmethod
    def evaluate(self, resolution: float):  # -> Numerical
        """Obtains and returns a numerical representation of itself.
        This is where the work is done in each subclass."""
        return Numerical(self.continuous_domain, resolution)

    def get(self, extreme_domain: (float, float), resolution: float, defuzzification_parameters) -> (float, float):
        """Returns a crisp value that is equivalent to its (generally) fuzzy value,
        along with that crisp value's suitability (a measure, on [0,1], of how good its answer is).

        Returns: (v, s)
            v: The crisp equivalent of this fuzzy number, a float.
            s: The suitability (appropriateness, quality, truth, certainty) of v as an equivalent, a float on [0,1].

        Arguments:
            extreme_domain: bounds the result in case the answer must be limited,
                e.g., if tempo must be on [30, 200] bpm, or a parameter must be on [0,100].
            resolution: the maximum distance between values that will be considered in the numerical representation.
                This controls the accuracy of the result (a smaller resolution is better).
                Also, consider that a coarse mesh in the numerical representation might miss narrow peaks.
            defuzzification_parameters:  I don't know what they are yet.
        """
        # Evaluate, impose an extreme domain, defuzzify.  This is where it's implemented for every subclass.
        numerical = self.evaluate(resolution)
        # impose an extreme domain
        numerical.impose_domian(extreme_domain)
        # defuzzify.
        v = s = 0  # dummy
        return v, s


class Numerical(Value):
    """The numerical representation of a (generally) fuzzy value.

    To represent the suitabilities of all real numbers, it uses:
        * One Numpy array s(v) for one continuous domain [a, b].
        * A set of exceptional points {(v_i, s_i)} that always override the array.
        * An out-of-bounds suitability, s_0, that is assumed for any value otherwise undefined.

        Is this where the defuzzify method is implemented?  Probably."""

    def __init__(self, domain: (float, float), resolution: float):
        """Initialization prepares the continuous domain with sample points that are
        integer multiples of the resolution (so that all arrays in the calculation will line up),
        covering the stated domain plus guard points on either end (for future interpolation).
        So,  conveniently, subclasses perform a function on this array to sample themselves.

        The set of exceptional points (the discrete domain) is an empty 2D array of value, suitability pairs.
        Otherwise undefined points in the domain default to a suitability of 0.

        Args:
            domain: values over which the continuous domain will be defined.
            resolution: the separation between sample points (a smaller resolution is better).
        """
        self.continuous_domain = domain
        v_0 = math.floor(domain[0] / resolution) - 1
        v_n = math.ceil(domain[1] / resolution) + 1
        number_of_samples = v_n - v_0 + 1
        v_0, v_n = v_0 * resolution, v_n * resolution
        # sample points on the continuous domain, to be filled with s(v) by subclasses:
        self.continuous_v = np.linspace(v_0, v_n, number_of_samples)
        self.continuous_s = np.linspace(v_0, v_n, number_of_samples)
        # the discrete domain, to be filled as v,s by subclasses:
        self.exceptional_points = np.empty((2, 0))
        # the suitability elsewhere, outside the defined domains
        self.out_of_bounds = 0

    def suitability(self, value: float):
        """Returns the suitability of a given value, as defined by this fuzzy value.

        The exceptional points of the discrete domain override the definition of the continuous domain,
        which is generally found by interpolation.  Points outside these domains return a default value."""
        discrete = np.where(value == self.exceptional_points[0], self.exceptional_points[1])
        if discrete is not None:
            return discrete
        else:
            if value < self.continuous_domain[0] or value > self.continuous_domain[1]:
                return self.out_of_bounds
            else:
                return np.interp(value, self.continuous_v, self.continuous_s)

    def evaluate(self, resolution: float):  # -> Numerical
        """It returns itself because it is the evaluation.

        In any other subclass of Value, this is where the work would be done."""
        return self

    def impose_domian(self, imposed_domain: (float, float)):
        """Discard any defined suitabilites <a | >b."""
        self.exceptional_points = np.where(self.exceptional_points[0] > imposed_domain[0] and \
                                           self.exceptional_points[0] < imposed_domain[1], self.exceptional_points)
        self.continuous_s = np.where(self.continuous_v > imposed_domain[0] and \
                                     self.continuous_v < imposed_domain[1], self.continuous_s)
        self.continuous_v = np.where(self.continuous_v > imposed_domain[0] and \
                                     self.continuous_v < imposed_domain[1], self.continuous_v)

    def defuzzify(self) -> (float, float):
        v = s = 0  # dummy  I don't know all the methods yet, but I prefer median of global maxima.
        return v, s


# Triangle, Trapezoid, Bell, Cauchy, Gauss, Points --- "atoms" defining a value;
# ValueNot, ValueAnd, ValueOr --- logic on values;
# Sum, Difference, Product, Quotient, Focus, Abs, Inverse --- arithmetic on values.

class Triangle(Value):
    """Describes a fuzzy number as a trianglular function with a peak (maximum s) and extreme limits (s==0)"""

    def __init__(self, peak, domain):
        """Args:
            peak:
                float: (most suitable) value , assumes s=1
                (float, float): (value, suitability)
            domain:
                float:  HWHM
                (float, float): extreme domain where s>0"""

        if isinstance(peak, (float, int)):  # assume its suitability = 1
            self.peak = (peak, 1.0)
        else:
            self.peak = peak
        if isinstance(domain, (float, int)):  # assume it is the HWFM about the peak
            self.continuous_domain = (self.peak[0] - domain, self.peak[0] + domain)
        else:
            self.continuous_domain = domain  # shouldn't I check these things?

    def evaluate(self, resolution: float):  # rethink this
        n = Numerical(self.continuous_domain, resolution)
        a_left = self.peak[1] / (self.peak[0] - self.continuous_domain[0])
        a_right = self.peak[1] / (self.peak[0] - self.continuous_domain[1])
        d = n.continuous_v  # - self.peak[0]

        s = n.continuous_s
        s = np.piecewise(d, [d < self.continuous_domain[0],
                             (d > self.continuous_domain[0]) and (d < self.peak[0]),
                             (d >= self.peak[0]) and (d < self.continuous_domain[1]),
                             d > self.continuous_domain[1]],
                         [lambda d: 0, lambda d: 1, lambda d: 2, lambda d: 3])
        print(d)
        # n.continuous_s =[lambda d: 0, lambda d: a_left*d, lambda d: 1-a_right*d, lambda d: 0]
        print(s)
        return n


# Here is where I am testing or playing around or something.

n = Norm(norm="pp")
print(type(n))
print(n.not_(.1))
print(n.and_(.5,.5))
print(n.or_(.5,.5))
