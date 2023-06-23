"""
Contains the code for fuzzy logic and math. 
"""


# I had thought to include a type for fuzzy units ("fits"), reals guaranteed to be on [0,1], but
# that would seem to violate the spirit of weak typing, and might slow things down with fuzzy checks all the time.

def fit_clip(s: float) -> float:
    """Clips a number to the range [0,1], ensuring that it's a fuzzy unit."""
    if s > 1:
        return 1
    elif s > 0:
        return s
    else:
        return 0


def fuzzy_not(s):
    """The standard fuzzy negation.

    I don't anticipate using others, so this gives us a one-to-one relation between t-norms and co-norms
    through DeMorgan's law."""
    return 1 - s


fuzzy_ands = dict[  # This defines t-norms for fuzzy logical AND operations, from least to most strict:
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

fuzzy_ors = dict[  # This defines co-norms for fuzzy logical OR operations, from least to most strict:
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

def fuzzy_op(andor: str, a: float, b: float, norm_1="pp", norm_2=None, w=50.0) -> float:
    op = (fuzzy_ands if andor[0] == "a" or "A" else fuzzy_ors)  # if it isn't an AND, it's an OR
    if norm_1.isinstance(float):  # assume a parameterized crossfade: mm=0, he=25, pp=50, lb=75, nn=90, dd=100
        p = norm_1
        if p < 25:
            p = p / 25
            return (1 - p) * op["mm"](a, b) + p * op["he"](a, b)
        elif p < 50:
            p = (p - 25) / 25
            return (1 - p) * op["he"](a, b) + p * op["pp"](a, b)
        elif p < 75:
            p = (p - 50) / 25
            return (1 - p) * op["pp"](a, b) + p * op["lb"](a, b)
        elif p < 90:
            p = (p - 75) / 15
            return (1 - p) * op["lb"](a, b) + p * op["nn"](a, b)
        else:
            p = (p - 90) / 10
            return (1 - p) * op["nn"](a, b) + p * op["dd"](a, b)
    elif not norm_2:  # then it's a single norm
        if norm_1.isinstance(tuple):
            norm_name = str(norm_1[0])
            p = op[norm_name + "p"](float(norm_1[1]))
            return op[norm_name](a, b, p)
        else:
            return op[norm_1](a, b)
    else:  # then it's two norms and a weight

# args[0].isinstance(str):  # assume one norm or a linear combination of two
# norm_1=str(args[0])
# if args.len()==1:
#     return op[norm_1](a, b)
# elif args.len()==2:
#     if args[1].isinstance(str):
#         return (op[norm_1](a, b) + op[str(args[1])](a, b))/2
#     else:
#         return op[norm_1](a, b,)
