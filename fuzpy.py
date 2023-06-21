fuzzy_not = lambda s: 1 - s

fuzzy_ands = dict[
             "mm": lambda a, b: min(a, b),  # minimum / maximum (Gödel)
             "pp": lambda a, b: a * b,  # product / probabilistic sum (Goguen)
             "lb": lambda a, b: max(0, a + b - 1),  # Łukasiewicz  / bounded sum
             "dd": lambda a, b: b if a == 1 else a if b == 1 else 0,  # drastic t-norm / co-norm
             "nn": lambda a, b: min(a, b) if a + b > 1 else 0,  # nilpotent minimum / maximum (Kleene-Dienes)
             "he": lambda a, b: 0 if a == b == 0 else a * b / (a + b - a * b)  # Hamacher product / Einstein sum
             ]


def fuzzy_and(a, b, tnorm):
    pass
