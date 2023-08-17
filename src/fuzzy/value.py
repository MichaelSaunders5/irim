"""Value
The :class:`.Value` class applies the idea of fuzzy truth.py to the representation of numbers.  A :class:`Value`
object is a function of truth.py vs. value.  We may think of the function as describing the suitability of each
possible value for some purpose.  E.g., we might describe room temperature as a symmetrical triangular function
on (68, 76)°F or (20, 24)°C.  In this way, we might state our opinions as "preference curves".  We might also
represent empirical knowledge as fuzzy numbers.  E.g., sensory dissonance vs. interval is a highly structured
function with many peaks and valleys across a wide domain, but all of this information can be encapsulated
in a single :class:`Value` object.

and...

"""

# Here are the classes for fuzzy value and arithmetic:
# Value --- base class;  Numerical --- "working" class of evaluation and defuzzification;
# Triangle, Trapezoid, Bell, Cauchy, Gauss, Points --- "atoms" defining a value;
# ValueNot, ValueAnd, ValueOr --- logic on values;
# Sum, Difference, Prod, Quotient, Focus, Abs, Inverse --- arithmetic on values.

# class Value(ABC):
#     """Represents a generally fuzzy real number (as a function of suitability (on [0,1]) vs. value).
#     It may be obtained (defuzzified) as a crisp value, along with that value's suitability.
#     """
#     _default_norm = Norm.define()
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
#             s: The suitability (appropriateness, quality, truth.py, certainty) of v as an equivalent, a float on [0,1].
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
#         Otherwise, undefined points in the domain default to a suitability of 0.
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
# # Sum, Difference, Prod, Quotient, Focus, Abs, Inverse --- arithmetic on values.
#
# class Triangle(Value):
#     """Describes a fuzzy number as a trianglular function with a peak (maximum s) and extreme limits (s==0)"""
#
#     def __init__(peak, domain):
#         """Parameters:
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
