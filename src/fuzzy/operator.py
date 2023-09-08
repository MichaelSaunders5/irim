from fuzzy.norm import *
from fuzzy.value import *

Operand = Union[Value, Truth, float, int, bool]  # but not np.ndarray, because it has no domain

class Truthy(Numerical):
    """A wrapper for Truths (or equivalent numbers) we want to treat as :class:`.Value`\\ s."""
    def __init__(self, suitability):
        super().__init__(default_suitability=suitability)
    def __str__(self):
        return str(f"truthy {self.ds}")

class Operator(Value):
    """A base class: A :class:`.Value` with a :class:`.Norm`.  To be added by subclasses: operands."""
    def __init__(self, domain: Tuple[float, float], default_suitability: float = 0, norm: Norm = None):
        """"""
        super().__init__(domain, default_suitability)  # bogus because only norm matters.
        self.n = default_norm if norm is None else norm
    @abstractmethod
    def evaluate(self, resolution: float) -> Numerical:
        pass
    @abstractmethod
    def suitability(self, v: float) -> float:
        pass
    @staticmethod
    def _prepare_operand(operand: Operand):
        """If it's not a Value, make it a Truthy: a Value with .ds == it."""
        if not isinstance(operand, Value):
            if isinstance(operand, Truth):
                operand = operand.to_float()
            operand = Truthy(operand)   # If it's a Truthy, we'll make its .ds operate with all other attributes.
        return operand

class UnaryOperator(Operator):
    """A base class: An :class:`.Operator` has one :class:`.Operand` operand."""
    def __init__(self, a: Operand, n: Norm = None):
        """"""
        a = Operator._prepare_operand(a)
        super().__init__(a.d, a.ds, n)    # bogus because only self.a matters and nothing happens until evaluation.
        self.a = a

    def unary_operate(self, resolution: float, operator: Norm.Operator) -> Numerical:
        """"""
        na = self.a.evaluate(resolution)    # a Numerical version of the operand
        if na.xp is not None:
            xpv = np.atleast_2d(na.xp)[:, 0]
            xps = np.atleast_2d(na.xp)[:, 1]
            xps = operator(xps)
            na.xp = np.column_stack((xpv, xps))
        na.s = operator(na.s)
        na.ds = operator(na.ds)
        return na

    @abstractmethod
    def suitability(self, v: float) -> float:
        pass


class Not(UnaryOperator):
    """"""
    def __str__(self):
        a = self.a.__str__()
        return str(f"NOT({a})")
    def evaluate(self, resolution: float) -> Numerical:
        """"""
        return super().unary_operate(resolution, self.n.not_)   # this would be clearer pulled down here.

    def suitability(self, v: float) -> float:
        """"""
        return self.n.not_(self.a.suitability(v))

# TODO:  test Negative and code the rest
class Negative(UnaryOperator):
    """"""
    def __str__(self):
        a = self.a.__str__()
        return str(f"NEG({a})")
    def evaluate(self, resolution: float) -> Numerical:
        """"""
        na = self.a.evaluate(resolution)
        na.d = (-na.d[1], -na.d[0])
        if na.xp is not None:
            xpv = np.atleast_2d(na.xp)[:, 0]
            xps = np.atleast_2d(na.xp)[:, 1]
            xpv = -1 * xpv
            na.xp = np.column_stack((xpv, xps))
        na.v = -1 * na.v
        return na

    def suitability(self, v: float) -> float:
        """"""
        return -self.a.suitability(v)


class Invert(UnaryOperator):
    """"""
    def __str__(self):
        a = self.a.__str__()
        return str(f"INV({a})")
    def evaluate(self, resolution: float) -> Numerical:
        """"""
        na = self.a.evaluate(resolution)
        na.d = (-na.d[1], -na.d[0])     # This will take resampling... but how far?  ...
        if na.xp is not None:
            xpv = np.atleast_2d(na.xp)[:, 0]
            xps = np.atleast_2d(na.xp)[:, 1]
            xpv = -1 * xpv
            na.xp = np.column_stack((xpv, xps))
        na.v = -1 * na.v
        return na

    def suitability(self, v: float) -> float:
        """"""
        return 1 / self.a.suitability(v)


class Abs(UnaryOperator):
    """"""
    def __str__(self):
        a = self.a.__str__()
        return str(f"ABS({a})")
    def evaluate(self, resolution: float) -> Numerical:
        """"""
        na = self.a.evaluate(resolution)
        na.d = (-na.d[1], -na.d[0])     # This will take resampling and ORing... but how far?  ...
        if na.xp is not None:
            xpv = np.atleast_2d(na.xp)[:, 0]
            xps = np.atleast_2d(na.xp)[:, 1]
            xpv = -1 * xpv
            na.xp = np.column_stack((xpv, xps))
        na.v = -1 * na.v
        return na

    def suitability(self, v: float) -> float:
        """"""
        return abs(self.a.suitability(v))
