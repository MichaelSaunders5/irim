"""
Contains the code for fuzzy logic and math. 
"""

import math
from abc import ABC, abstractmethod

import numpy as np


# I had thought to include a type for fuzzy units ("fits"), reals guaranteed to be on [0,1], but
# that would seem to violate the spirit of weak typing, and might slow things down with fuzzy checks all the time.

# here are the functions for fuzzy logic: clip, not, and/or:

def f_clip(s: float) -> float:
    """Clips a number to the range [0,1], ensuring that it's a fuzzy unit."""
    if s > 1:
        print(f"{s} out of fit bounds, clipped to 1")
        return 1
    elif s > 0:
        print(f"{s} out of fit bounds, clipped to 0")
        return s
    else:
        return 0


def f_not(s):
    """The standard fuzzy negation.

    I don't anticipate using others, so this gives us a one-to-one relation between t-norms and co-norms
    through DeMorgan's law."""
    return 1 - s


# <editor-fold desc="t-norm/co-norm definitions">
FUZZY_ANDS = dict[  # This defines t-norms for fuzzy logical AND operations, from least to most strict:
             "mm": lambda a, b: min(a, b),  # minimum / maximum (Gödel)
             "he": lambda a, b: 0 if a == b == 0 else a * b / (a + b - a * b),  # Hamacher product / Einstein sum
             "pp": lambda a, b: a * b,  # product / probabilistic sum (Goguen)
             "lb": lambda a, b: max(0, a + b - 1),  # Łukasiewicz  / bounded sum
             "nn": lambda a, b: min(a, b) if a + b > 1 else 0,  # nilpotent minimum (Kleene-Dienes)
             "dd": lambda a, b: b if a == 1 else a if b == 1 else 0,  # drastic t-norm
             # parameterized t-norms:
             # Names of parameterized norms should be two letters + "p".
             # Names of their mapping from [0,100] to whatever should be this + "p" again.
             "hep": lambda a, b, p: 0 if p == a == b == 0 else a * b / (p + (1 - p) * (a + b - a * b)),
             # parameterized Hamacher: p>=0 !!! p: 0(0)=Hamacher, 50(1)=product, 100(+inf)=drastic
             "hepp": lambda p: 0 if p < 0 else 10 ** (.18 * p - 9)  # parameter mapping for "hep"="hep"+"p"
             # Names of parameterized norms should be two letters + "p".
             # Names of their mapping from [0,100] to whatever should be this + "p" again.
             ]

FUZZY_ORS = dict[  # This defines co-norms for fuzzy logical OR operations, from least to most strict:
            "mm": lambda a, b: max(a, b),  # maximum / minimum  (Gödel)
            "he": lambda a, b: (a + b) / (1 + a * b),  # Einstein sum / Hamacher product
            "pp": lambda a, b: a + b - a * b,  # probabilistic sum / product  (Goguen)
            "lb": lambda a, b: min(a + b, 1),  # bounded sum / Łukasiewicz
            "nn": lambda a, b: max(a, b) if a + b < 1 else 1,  # nilpotent maximum (Kleene-Dienes)
            "dd": lambda a, b: b if a == 0 else a if b == 0 else 1,  # drastic co-norm
            # parameterized co-norms:
            # Names of parameterized norms should be two letters + "p".
            # Names of their mapping from [0,100] to whatever should be this + "p" again.
            "hep": lambda a, b, p: (a + b - p * a * b) / (1 + (1 - p) * a * b),
            # parameterized Hamacher: p>=0 !!! p: 0(0)=Hamacher, 50(1)=product, 100(+inf)=drastic
            "hepp": lambda p: 0 if p < 0 else 10 ** (.18 * p - 9)  # parameter mapping for "hep"="hep"+"p"
            ]


# </editor-fold>


# I would provide more t-norms (Schweizer, Frank, Yager, Aczél–Alsina, Dombi, Sugeno–Weber, etc.),
# but I don't know how to interpret them continuously (which is necessary for the fuzzy arithmetic)---
# I only know how to do Riemann and geometric (product) integrals.

def f_op(andor: str, a: float, b: float, norm_1="pp", norm_2=None, w=50.0) -> float:
    """Performs fuzzy logic AND or OR operation on a pair of fits, returning a fit.

    andor: "and"|"or" chooses which type of operator
    a, b: fits (floats assumed to be on [0,1]
    The next three parameters choose the t-norm or co-norm that defines the operator.
    norm_1, norm_2: can be called in three ways:
    * as "None" (see below)
    * as a two-letter string indicating a norm (they come in t-norm/co-norm pairs intended to be used together).
    * as a tuple: (a three-letter string indicating a parameterized norm, a parameter for it on [0,100]).
    If norm_1 is None: w is a parameter on [0,100] indicating the strictness (extremeness) of the operation.
        So, a larger number makes AND more likely to be False and OR more likely to be True. (See chart below)
    If norm_1 is defined, but norm_2 is None:  only norm_1 is used.
    If both norms are defined: their average, weighted by w (0=norm_1, 100=norm_2).
    w: a parameter on [0,100]
        if norm_2 is defined: the percent weight combining the results of norm_1 and norm_2.
        if norm_1 is None: the result is a "crossfade" as in the following chart:

    The provided norm pairs are (in increasing strictness with their number if "crossfade" is used:
          0  mm  minimum / maximum (Gödel, Zadeh)
         25  he  Hamacher product / Einstein sum
         50  pp  product / probabilistic sum (Goguen) (This is always the default!)
         75  lb  Łukasiewicz  / bounded sum
         90  nn  nilpotent minimum / maximum (Kleene-Dienes)
        100  dd  drastic t-norm / co-norm
             hep the parameterized versions of Hamacher product / Einstein sum: 0=he, 50=pp, 100=dd
"""

    op = (FUZZY_ANDS if andor[0] == "a" or "A" else FUZZY_ORS)  # if it isn't an AND, it's an OR

    if norm_1 is None:  # Do a "crossfade" norm parameterized by w: mm=0, he=25, pp=50, lb=75, nn=90, dd=100.
        if w < 25:
            w = w / 25
            return (1 - w) * op["mm"](a, b) + w * op["he"](a, b)
        elif w < 50:
            w = (w - 25) / 25
            return (1 - w) * op["he"](a, b) + w * op["ww"](a, b)
        elif w < 75:
            w = (w - 50) / 25
            return (1 - w) * op["ww"](a, b) + w * op["lb"](a, b)
        elif w < 90:
            w = (w - 75) / 15
            return (1 - w) * op["lb"](a, b) + w * op["nn"](a, b)
        else:
            w = (w - 90) / 10
            return (1 - w) * op["nn"](a, b) + w * op["dd"](a, b)
    else:
        if isinstance(norm_1, str):  # Perform the simple norm.
            result_1 = op[norm_1](a, b)
        else:  # It's a tuple.  Perform the parameterized norm
            p = op[norm_1[0] + "p"](norm_1[1])  # Map the norm parameter from [0,100] to whatever the norm wants.
            result_1 = op[norm_1[0]](a, b, p)

    if norm_2 is None:
        return result_1
    else:
        if isinstance(norm_2, str):  # Perform the simple norm.
            result_2 = op[norm_2](a, b)
        else:  # It's a tuple.  Perform the parameterized norm
            p = op[norm_2[0] + "p"](norm_2[1])  # map the norm parameter from [0,100] to whatever the norm wants
            result_2 = op[norm_2[0]](a, b, p)
        return ((100 - w) * result_1 + w * result_2) / 100


# here are the classes for fuzzy value and arithmetic:
# Value --- base class;  Numerical --- "working" class of evaluation and defuzzification;
# Triangle, Trapezoid, Bell, Cauchy, Gauss, Points --- "atoms" defining a value;
# ValueNot, ValueAnd, ValueOr --- logic on values;
# Sum, Difference, Product, Quotient, Focus, Abs, Inverse --- arithmetic on values.

class Value(ABC):
    """Represents a generally fuzzy real number (as a function of suitability (on [0,1]) vs. value).
    It may be obtained (defuzzified) as a crisp value, along with that value's suitability.
    """

    @property
    @abstractmethod
    def continuous_domain(self):
        pass

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
        self._continuous_domain = domain
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
            if value < self._continuous_domain[0] or value > self._continuous_domain[1]:
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

    def __init__(self, peak, domain: (float, float)):
        """Args:
            peak:
                float: (most suitable) value , assumes s=1
                (float, float): (value, suitability)
            domain:
                float:  HWHM
                (float, float): extreme domain where s>0"""
        if peak.isFloat:  # assume its suitability = 1
            self.peak = (peak, 1.0)
        else:
            self.peak = peak
        if domain.isFloat:  # assume it is the HWFM about the peak
            self._continuous_domain = (peak[0] - 2 * domain, peak[0] + 2 * domain)
        else:
            self._continuous_domain = domain  # shouldn't I check these things?

    def evaluate(self, resolution: float):
        n = Numerical(self.continuous_domain,resolution)
        a = self.peak[1] / (self.continuous_domain[1] - self.peak[0])
        n.continuous_s = f_clip(a * (n.continuous_v - self.peak[0]))
        return n

