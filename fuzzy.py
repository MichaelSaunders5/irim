"""
Contains the code for fuzzy logic and math. 
"""
# from abc import ABC, abstractmethod


# I had thought to include a type for fuzzy units ("fits"), reals guaranteed to be on [0,1], but
# that would seem to violate the spirit of weak typing, and might slow things down with fuzzy checks all the time.

# here are the functions for fuzzy logic: clip, not, and/or:

def f_clip(s: float) -> float:
    """Clips a number to the range [0,1], ensuring that it's a fuzzy unit."""
    if s > 1:
        return 1
    elif s > 0:
        return s
    else:
        return 0


def f_not(s):
    """The standard fuzzy negation.

    I don't anticipate using others, so this gives us a one-to-one relation between t-norms and co-norms
    through DeMorgan's law."""
    return 1 - s


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

class Value:  # (ABC)

    # @abstractmethod
    def evaluate(self):  # -> Numerical
        # evaluates (obtains a numerical representation).
        return Numerical()

    def get(self) -> (float, float):
        # evaluate, impose an extreme domain, defuzzifies.
        v = s = 0  # dummy
        return v, s


class Numerical(Value):
    # continuous domain
    # discrete domain
    out_of_bounds_value = 0

    def defuzzify(self) -> (float, float):
        v = s = 0  # dummy
        return v, s

# Triangle, Trapezoid, Bell, Cauchy, Gauss, Points --- "atoms" defining a value;
# ValueNot, ValueAnd, ValueOr --- logic on values;
# Sum, Difference, Product, Quotient, Focus, Abs, Inverse --- arithmetic on values.
