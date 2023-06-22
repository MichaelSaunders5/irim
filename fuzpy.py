'''
Contains the code for fuzzy logic and math.
'''

# I had thought to include a type for fuzzy units ("fits"), reals gauranteed to be on [0,1], but
# that would seem to violate the spirit of weak typing, and might slow things down with fuzzy checks all the time.
def fuzzy_not(s):
    '''The standard fuzzy negation.

    I don't anticipate using others, so this gives us a one-to-one relation between t-norms and co-norms
    through DeMorgan's law.'''
    return 1 - s


fuzzy_ands = dict[  # This defines t-norms for fuzzy logical AND operations.
             "mm": lambda a, b: min(a, b),  # minimum / maximum (Gödel)
             "pp": lambda a, b: a * b,  # product / probabilistic sum (Goguen)
             "lb": lambda a, b: max(0, a + b - 1),  # Łukasiewicz  / bounded sum
             "dd": lambda a, b: b if a == 1 else a if b == 1 else 0,  # drastic t-norm
             "nn": lambda a, b: min(a, b) if a + b > 1 else 0,  # nilpotent minimum (Kleene-Dienes)
             "he": lambda a, b: 0 if a == b == 0 else a * b / (a + b - a * b),  # Hamacher product / Einstein sum
             "hep": lambda a, b, p: 0 if p == a == b == 0 else a * b / (p + (1 - p) * (a + b - a * b))
             # parameterized Hamacher: p=0: =Hamacher, p=1: =product, p>=0 !!!
             ]

fuzzy_ors = dict[  # This defines co-norms for fuzzy logical OR operations.
            "mm": lambda a, b: max(a, b),  # maximum / minimum  (Gödel)
            "pp": lambda a, b: a + b - a * b,  # probabilistic sum / product  (Goguen)
            "lb": lambda a, b: min(a + b, 1),  # bounded sum / Łukasiewicz
            "dd": lambda a, b: b if a == 0 else a if b == 0 else 1,  # drastic co-norm
            "nn": lambda a, b: max(a, b) if a + b < 1 else 1,  # nilpotent maximum (Kleene-Dienes)
            "he": lambda a, b: (a + b) / (1 + a * b),  # Einstein sum / Hamacher product
            "hep": lambda a, b, p: (a + b - p * a * b) / (1 + (1 - p) * a * b)
            # parameterized Hamacher
            ]


# I would provide more t-norms (Schweizer, Frank, Yager, Aczél–Alsina, Dombi, Sugeno–Weber, etc.),
# but I don't know how to interpret them continuously (which is necessary for the fuzzy arithmetic)---
# I only know how to do Riemann and geometric (product) integrals.

def fuzzy_and(a, b, *args):
    pass
