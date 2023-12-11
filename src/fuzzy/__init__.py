"""provides all the facilities for working with fuzzy logic and arithmetic.

.. epigraph::

   | “If a man will begin with certainties, he shall end in doubts;
   | but if he will be content to begin with doubts, he shall end in certainties.”

   -- Francis Bacon, *Of the Proficience and Advancement of Learning, Divine and Human*

#######################################
Introduction:  What Is Fuzzy Technique?
#######################################

The fuzzy technique automates the practical wisdom we call "skill".  It is for expressing expert knowledge and for
making reasoning with it computable.  It automates the type of judgement and behaviour at which humans are particularly
adept.  It is good at dealing with imprecise or incomplete information, conflicting rules, opinions, systems
of heuristics, vague notions, subtle influences, nuance, personal preference, ambiguity, combined ideas, style,
perceptual data, subjective experience, whims, and fancies.  All of this is in addition to a perfectly fine ability
to work with hard facts and precision---the fuzzy is a superset which includes the crisp.

Crisp logic is concerned with a domain of truth that can take only two values: {0,1}, absolute falsehood or
absolute truth.  Fuzzy logic extends this to include the infinite degrees of truth in-between---its domain
is [0,1].  Crisp logic defines operators with truth tables:  a number of operands (usually one or two) each have
crisp truths (0 or 1), and every possible combination of these is assigned a crisp result, also on the {0,1} domain.
Fuzzy logic defines operators as functions mapping one or more combined domains of [0,1] onto a range of [0,1].
The crisp truth tables still hold at the corners of this space, but in-between there is room for mystery and adventure.

To extend the idea to arithmetic, consider our familiar, "crisp" numbers.  Each is really a statement saying:
"of all the real numbers, this one, and only this one, is presently the case (it is completely true)
and all others are not (they are completely false).  A single fuzzy number, on the other hand, indicates the degree of
relative truth or falsehood for every member of the real numbers---it is a function of truth vs. value.  I like
to think of this as the measure of the suitability of a proposed value for a given purpose---some values may be
absolutely suitable or unsuitable (1, or 0), and some may be somewhere in-between.  With fuzzy numbers thus
represented by arbitrary functions, arithmetic operators on them become something rather complicated,
but the :mod:`fuzzy` package encapsulates all the difficulty into code hidden behind familiar symbols.

You may express your problems in the familiar crisp form; solve them with the usual techniques of logic, mathematics,
and so on; and also express their solutions in the familiar form:  with crisp variables and operators.
When the :mod:`fuzzy` package is used, the very same expressions become "overloaded" with fuzzy meaning, and
complex, organic, artistically-inflected behaviours emerge.  And so, fuzzy techniques fit easily into algorithms
almost anywhere.  Beyond this, they allow problems and solutions to be formulated in a more daring way:  experts
may express their knowledge of a domain in heuristics that the technique makes computable.

Most implementations of fuzzy techniques focus on logic more than math, typically representing quantity crudely,
as trapezoidal numbers operated on with interval arithmetic.  This is adequate for simple control systems.
The present :mod:`fuzzy` package, however, represents numbers as arbitrary functions and provides rigorously-derived
operators that preserve their detail.  This allows individual numbers to convey the subtlety of expression
necessary for both scientific precision and artistic richness.

The package is housed in six modules: :mod:`.norm`, :mod:`.truth`, :mod:`.number`,  :mod:`.literal`,  :mod:`.operator`,
and :mod:`.crisp`.  If you just want fuzzy logic, import :mod:`truth`.  If you want fuzzy math, import :mod:`literal`
and :mod:`operator`.  It's unlikely that you'll ever need to change their defaults or to use most of them directly.
Almost all fuzzy reasoning can be performed with :class:`.Truth` and :class:`.Literal` objects (analogous to ``bool``
and ``float`` respectively) using common logic and math operators overloaded in :class:`.Truth`
and :class:`.Operator`---so your fuzzy work will look like ordinary logical and arithmetical expressions.

I will explain how to use the package, and then, how the more interesting features work.

**********************
How to Use the Package
**********************

The fuzzy technique is this:

    #. Take some values.
    #. Put them in logical/mathematical expressions.
    #. Get the result of the calculation in a crisp form.
    #. Assess its quality.
    #. Act accordingly.

Routines like this can fit into many places within conventional algorithms.  Often, familiar tasks can be performed with
greater elegance and refinement.  Better yet:  a richer variety of tasks becomes practical.  The following sections
will deal with each step of the technique.

Fuzzy Values
============

We normally use two kinds of values in reasoning:  truth itself, and number (let's restrict ourselves to the real
numbers).  Python represents these as ``bool`` and ``float`` objects, respectively.  We will represent them with
:class:`.Truth` and :class:`.Literal` objects.  We will need to be able to create "literal" objects of these types
to state facts and opinions in fuzzy form.

Ordinary crisp logic and ``bool`` describe truth with two possible values, the Boolean domain: {0,1}, for ``False``
and ``True``.  Fuzzy logic and the :mod:`.fuzzy` package add to this the infinite possibilities of partial truths
in-between, the fuzzy domain: [0,1].  A :class:`.Truth` object is initialized simply, with a statement
like ``sincerity = Truth(.8)``.  Between the extremes of "yes" and "no" you can define "maybe" by setting the
rare parameter :attr:`fuzzy.truth.default_threshold`.  I don't recommend
changing the obvious default, .5, which allots equal space to the agreeable and the reluctant.

Numbers are a much richer field for the imagination.  Familiar ``float`` objects represent the real numbers
with enough accuracy and range that you can usually think of them as the same thing.  Both describe a
number as being absolutely true for one value and absolutely false for all other values.  The :mod:`.fuzzy` package
represents a real number as a fuzzy truth assigned to every possible real value, i.e., as a function, :math:`t(v)`,
of truth vs. value over all available floats---a function with range [0,1] and domain practically infinite.

For example, we might describe room temperature as a symmetrical triangular function
on (68, 76)°F or (20, 24)°C.  The certainty, strength, or likelihood of an input and our opinions about the
desirability or "goodness" of an output can encoded in a single number.  Notice how the fuzzy
number includes information about the ideal (the maximum), our tolerance for its variation (the width), and the limits
of what we will accept (the *support*, or non-zero domain).  An arbitrary function can contain an enormous amount
of detail.


Built-in Literal Numbers
------------------------

Using arbitrary functions to represent fuzzy numbers makes the package extremely powerful, but arbitrary functions
allow so much freedom that it is difficult to speak of them in directly.  As users, we need simple and intuitive ways
for stating "literal" fuzzy numbers.  The :mod:`.literal` module provides this with a variety of classes for defining
fuzzy numbers parametrically:

    * :class:`.Triangle`
    * :class:`.Trapezoid`
    * :class:`.Cauchy`
    * :class:`.Gauss`
    * :class:`.Bell`
    * :class:`.Inequality`
    * :class:`.Sigmoid`
    * :class:`.DPoints`
    * :class:`.CPoints`
    * :class:`.Exactly`
    * :class:`.Truthy`

They are subclasses of the abstract class :class:`.Literal`, which provides the apparatus for declaring the
fuzzy number conveniently and precisely:  as a function of one real variable ("value") implemented as a Python
method (:meth:`.Literal._sample`).  They include the option to restrict the domain further to a set of
uniformly-spaced, discrete values, and to a set of explicit, discrete values.  This is useful if the solutions one
seeks can only take certain values or must be integer multiples of some unit, e.g., if you need to reason about
feeding your elephants, you may leave the oats continuous, but please, make sure your elephants are discrete.

The classes :class:`.Triangle` and :class:`.Trapezoid` are the piecewise linear functions one usually encounters in
simpler fuzzy logic implementations.  (However, I like how they have constant variability over their linear segments,
and so have relatively large variabilities near their extrema---this keeps things exciting.)

The classes :class:`.Cauchy`, :class:`.Gauss`, and :class:`.Bell` all represent bell-shaped functions defined by their
peak and width, a very natural way for talking vaguely about a number.  The classes :class:`.Inequality`
and :class:`.Sigmoid` do the same for talking vaguely about an inequality---a shelf function with a fuzzy middle.

The classes :class:`.DPoints` and :class:`.CPoints` allow one to describe a fuzzy number as a set of (value, truth)
pairs.  In the former, this is simply a discrete set.  In the latter, it implies a continuous function interpolated
between the points (the choice of interpolator is another rare parameter, which may be left to the module
attribute, :attr:`fuzzy.number.default_interpolator`).  This is an excellent way to turn empirical data or incomplete
information into a fuzzy number.

The class :class:`.Exactly` is for talking about a crisp number in the fuzzy world---it defines a fuzzy number with
suitability 1 at the crisp value and 0 elsewhere.  Its counterpart, :class:`.Truthy`, is for talking about a fuzzy
truth in the world of fuzzy numbers---it defines a fuzzy number that is everywhere the same truth.  When writing
fuzzy expressions, it is usually safe to mix fuzzy numbers with :class:`.Truth` objects and ``floats``---these will
be automatically promoted to :class:`.Truthy` and :class:`.Exactly` objects on calculation.


Making Your Own Literals
------------------------

The above are fine for describing numbers in the ordinary sense---and simple numbers are often needed---but it is
most practical and interesting to use fuzzy numbers that represent real things---physical phenomena, mental experiences,
and relations among them.  This is done by using arbitrary, parameterized functions as fuzzy numbers.  The only
requirement is that their range be restricted to [0,1].  For example: Is the sun shining?  It depends on the time of
day, and, for a significant part of the day, a fuzzy answer is appropriate.

Another example:  sensory dissonance is a very predictable experience.  Usually, we are interested in its
amount vs. the pitch interval between two tones.  That spiky, highly structured curve, scaled to [0,1], becomes a very
useful fuzzy number if you want to tune a musical instrument.

Both examples may depend on many factors---custom-made
fuzzy numbers may well depend on all sorts of other objects as parameters---but they may be boiled down to useful
functions of one variable and used as fuzzy numbers in fuzzy reasoning.

No doubt you can think of many examples in whatever fields you have mastered.  To bring your knowledge into
the :mod:`fuzzy` package, you simply subclass of one of the existing classes.  There are three good candidates:

        * :class:`.Literal`, for functions defined by a Python method, on a continuous or discretized domain.
        * :class:`.CPoints`, for functions defined by passing through a few critical points.
        * :class:`.DPoints`, for arbitrary discrete points.

How do you choose?  If your fuzzy number could be easily described by a function, :math:`t(v)`, written in the form of
a Python method, use :class:`.Literal`.  If you know about certain important points, or curve features, or empirical
data, but not a detailed mathematical function, use :class:`.CPoints`.  If it's discrete, and you know the points that
matter or can determine them algorithmically, use :class:`.DPoints`.

First, you must implement your class's ``__init__`` method.  BinAdd to it the parameters that shape your function.
If you're subclassing :class:`.DPoints` or :class:`.CPoints`, you'll set
your points in it as well.  Finally, remember to call the superclass's constructor in your
own: ``super().__init__(...)``.

If you're subclassing :class:`.Literal`, you'll need to do one more thing:  implement the :meth:`._sample` method.
This is where the action is.  It is a mathematical function, based on your initialized parameters, and
defined over a continuous domain---given a value, it returns the truth of that value.  Its input and output,
though, are not single floats, but `Numpy <https://numpy.org/doc/stable/index.html>`_ arrays of floats, so the
mathematical expressions you use should be compatible with Numpy.  This will not change the usual syntax very much.
Of course, unlike an ordinary mathematical function, you have the full resources of Python and Numpy (and SciPy and
whatever else you like) to perform algorithms.

If you need to map real-world data to the fuzzy domain, you can use the helper :meth:`.literal._scale`.
Don't worry too much about singular points (like ``±inf`` or ``nan``), or otherwise defining suitabilities
outside [0,1]---this is guarded against internally.

There you go.  You've taken a concept from the real world---containing your human knowledge, experience, and insight,
I hope---and made it a computable object, available for reasoning in fuzzy algorithms.


Fuzzy Expressions
=================

You have defined or measured the truth of some propositions and the values of some numbers.  Now you can reason and
calculate with them by writing fuzzy expressions.  These are just like ordinary mathematical expressions that use
literal numbers, variables, and operators, but here the operators are methods of the :class:`.Operator` class.
Most of these are also available as the usual symbols (``&``, ``|``, ``+``, ``*``, and so on), overloaded to alias the
operator methods.  So, most fuzzy expressions you may write will look exactly the same as crisp expressions.  The
:mod:`fuzzy` package though, adds more functionality:  a few new operators and the ability to perform logical
operations on numbers.  First I'll describe the operators that work with :class:`.Truth` objects, then those that
work with :class:`.FuzzyNumber` objects, including all kinds of :class:`.Literal`\\ s.  Then, I'll describe the
qualifiers, operators unique to fuzzy truths and numbers that alter their strength.

Logical Operators
-----------------

Logical operators may be familiar to you as the logical connectives of propositional calculus or as the "logic gates"
of digital electronics.   In crisp logic, operators are defined by truth tables listing all possible inputs and their
corresponding outputs.  In fuzzy logic, these become continuous mathematical functions of the inputs, but at the
extremes of {0,1}, the truth tables still hold.  The functions that define the operators are called "norms" and will
be discussed below.  Here, I'll describe the available operators with the descriptive, crisp truth tables.

As for unary logical operators (those taking only one operand), two are possible and one is interesting:
**not**, which gives the opposite of the operand. (The uninteresting operator simply returns the operand).
**Not** can be accessed in three ways:

    * ``Truth.not_(a)``, where ``a`` can be a :class:`.Truth` object or a literal ``float``, ``int``, or ``bool``.
    * ``a.not_()``, where ``a`` must be a :class:`.Truth` object.
    * ``~a``, where ``a`` must be a :class:`.Truth` object.

The underscore is to difference it from a keyword.  The truth table for r = ~a is:

+----+---+---+
| a  | 0 | 1 |
+====+===+===+
| r  | 1 | 0 |
+----+---+---+

You may be wondering what function makes it fuzzy.  It is :math:`r = 1 - a`.  This is defined in :class:`.Norm`, and
discussed there.

There are sixteen logical binary operators.  Six of these are not useful:  the same as a, the same as b, the opposite
of these, always false (denial), and always true (insistence).  The remaining ten are:

+---+-------------+----------+--------------------+-------------------------------------------------------+
|   | truth table |          |                    |                                                       |
+---+-------------+----------+--------------------+-------------------------------------------------------+
| a | 0 0 1 1     |          |                    |                                                       |
+---+-------------+----------+--------------------+-------------------------------------------------------+
| b | 0 1 0 1     | method   | symbolic call      | names                                                 |
+===+=============+==========+====================+=======================================================+
| r | 0 0 0 1     | ``and_`` | a ``&`` b          | ∧, and, conjunction, intersection                     |
+---+-------------+----------+--------------------+-------------------------------------------------------+
|   | 0 1 1 1     | ``or_``  | a ``|`` b          | ∨, or, disjunction, union                             |
+---+-------------+----------+--------------------+-------------------------------------------------------+
|   | 1 1 0 1     | ``imp``  | a ``>>`` b         | →, implies, material implication, if-then             |
+---+-------------+----------+--------------------+-------------------------------------------------------+
|   | 1 0 1 1     | ``con``  | a ``<<`` b         | ←, implied by, converse implication                   |
+---+-------------+----------+--------------------+-------------------------------------------------------+
|   | 1 0 0 1     | ``iff``  | ``~`` (a ``@`` b)  | ↔, if and only if, xnor, equivalence                  |
+---+-------------+----------+--------------------+-------------------------------------------------------+
|   | 0 1 1 0     | ``xor``  | a ``@`` b          | ⨁, xor, exclusive or, non-equivalence                 |
+---+-------------+----------+--------------------+-------------------------------------------------------+
|   | 1 1 1 0     | ``nand`` | ``~`` (a ``&`` b)  | ↑, nand, alternative denial                           |
+---+-------------+----------+--------------------+-------------------------------------------------------+
|   | 1 0 0 0     | ``nor``  | ``~`` (a ``|`` b)  | ↓, nor, joint denial                                  |
+---+-------------+----------+--------------------+-------------------------------------------------------+
|   | 0 0 1 0     | ``nimp`` | ``~`` (a ``>>`` b) | :math:`\\nrightarrow`, material non-implication        |
+---+-------------+----------+--------------------+-------------------------------------------------------+
|   | 0 1 0 0     | ``ncon`` | ``~`` (a ``>>`` b) | :math:`\\nleftarrow`, converse non-implication         |
+---+-------------+----------+--------------------+-------------------------------------------------------+

They have interesting relationships, both to each other and to natural language.  These are discussed in their
documentation in :class:`.Truth`: :meth:`.Truth.and_`, :meth:`.Truth.or_`, :meth:`.Truth.imp`, :meth:`.Truth.con`,
:meth:`.Truth.iff`, :meth:`.Truth.xor`, :meth:`.Truth.nand`, :meth:`.Truth.nor`, :meth:`.Truth.nimp`,
:meth:`.Truth.ncon`.  They may be called in the following ways, using **and** as an example:

    * ``Truth.and_(a, b)``, where ``a``, ``b`` can be any combination of :class:`.Truth` objects
      or literals of ``float``, ``int``, or ``bool``.
    * ``a.and_(b)``, where ``a`` must be a :class:`.Truth` object, but ``b`` can be a :class:`.Truth` object
      or a literal ``float``, ``int``, or ``bool``.
    * ``a & b``, where at least one of the operands must be a :class:`.Truth` object, but the others can be
      a :class:`.Truth` object or a literal ``float``, ``int``, or ``bool``.

Two of them, :meth:`.Truth.and_` and :meth:`.Truth.or_`, are more than binary---they are associative, meaning they
can be performed on any number of operands at once.  This means that either style of method call may include any
number of operands, or even an :class:`.Iterable` of operands such as a list or Numpy array.


.. figure:: ../../images/logic_heatmaps_prod.png
   :alt: the logic operators, product norm

   The logic operators.  The horizontal axis is a; the vertical axis is b. Lighter colors denote greater truth.
   The default norm (product) was used.


Arithmetic Operators
--------------------

The same patterns of method calls hold for arithmetic operators, in the unary, binary, and associative forms described
above, except that, in place of "Truth" you must use "Operator", e.g.:  ``Operator.not_(a)``, and the types of
operands can include :class:`.FuzzyNumber`\\ s (:class:`.Literal`\\ s, :class:`.Operator`\\ s, and fuzzy expressions
in general).  As mentioned before, objects of type :class:`.Truth`, ``float``, ``int``, and ``bool`` are interpreted
appropriately by the operators, so you can mix them freely in your expressions.  It is probably easiest to use
symbolic notation, but the text forms allow for the setting of rare parameters like norm.  Such settings automatically
propagate down to all operands belonging to an operator and to *its* operators, and so on, unless they too have
explicit settings.

The unary operators are:

    * :meth:`.Operator.not_`, for logical negation, as above, ¬a (``~a``).
    * :meth:`.neg`, for arithmetic negation, :math:`-a` (``-a``).
    * :meth:`.reciprocal`, for :math:`1/a`.
    * :meth:`.abs`, for absolute value, :math:`|a|` (``+a``).

The binary operators are:

    * The eight logical operators described
      above--- :meth:`.Operator.imp` (``a >> b``), :meth:`.Operator.con` (``a << b``),
      :meth:`.Operator.iff` (``~(a @ b)``), :meth:`.Operator.xor` (``a @ b``),
      :meth:`.Operator.nand` (``~(a & b)``), :meth:`.Operator.nor` (``~(a | b)``),
      :meth:`.Operator.nimp` (``~(a >> b)``), :meth:`.Operator.ncon` (``~(a << b)``).
    * :meth:`.sub`, for subtraction (``a - b``).
    * :meth:`.div`, for division (``a / b``).

The associative operators are:

    * The two logical operators described above:
      :meth:`.Operator.and_` (``a & b``), :meth:`.Operator.or_` (``a | b``)
    * :meth:`.add`, for addition (``a + b``).
    * :meth:`.mul`, for multiplication (``a * b``).

What does it mean for logic to operate on numbers?  It means that the logical operator is applied "element-wise", to
every value.  In binary operations, it is applied to every pair of like value defined by the operands (imagine
plotting the functions on the same graph and operating on every pair of points cut by a vertical line).
So, **anding** two numbers finds their compromise; **oring** them finds their combination; and so on,
for all the logical operators.  This is an extremely useful way of reasoning with numbers that is not easily
approximated by crisp methods.

.. figure:: ../../images/logic_on_numbers.png
   :alt: logic applied to fuzzy numbers

   Logic applied to fuzzy numbers.  The blue and yellow bell curves are fuzzy number operands.  The red curve is
   the result of the logical operator indicated.  The instance calls and equivalent overloaded symbols are shown for
   reference.  The default norm (product) was used.

Qualifiers
----------

Both :class:`.Truth`\\ s and :class:`.FuzzyNumber`\\ s can be qualified by methods that alter their strength in an
expression.  Given an operand and a parameter on [-100, 100], they return a weaker or stronger version of the operand
that will have more or less influence in its expression.  The range of the parameter is conventional---it may be
exceeded, but the results obtained at the extremes are probably as much as you would ever want to use.  In-between,
the effect of changing the parameter is meant to be perceptually linear.

:class:`.Truth` has only one type of qualifier: :meth:`.weight` (``a // w``, where ``w`` is the parameter).
Weight makes a truth more or less extreme.
A positive parameter pulls truths away from the :attr:`truth.default_threshold` towards either 0 or 1.
A negative parameter pulls them away from the extremes and towards the threshold.
This effect is really a partial "crisping" of the truth, as we shall see below.

:class:`.FuzzyNumber` has three types of qualifier:

* :meth:`.Operator.weight` (``a // w``), which is identical to the above,
  but applied to every truth in :math:`t(v)`;
* :meth:`.normalize`, which simply increases the range of truths to fill [0,1]; and,
* :meth:`.focus` (``a ^ f``, where ``f`` is the parameter).

Focus makes the peaks sharper or broader (and the valleys correspondingly broader or sharper).
A positive parameter pulls all truths but the highest down towards 0, making the peaks narrower---the number becomes
more certain and less amenable to compromise.
A negative parameter pulls them in the other direction, making the peaks broader---the number becomes vaguer and
less emphatic.

Imagine turning a number on its side and looking down on it.  If the truth corresponds to brightness, positive focus
makes the number sharper, clearer; negative focus makes it blurrier, less distinct.  As with weight, this is a
partial crisping of the number.

[pictures demonstrating the qualifiers]

Crisp Results
=============

.. epigraph::

    | “Did you ever have to finally decide?
    | And say yes to one and let the other one ride?
    | It's not often easy, and not often kind.
    | Did you ever have to make up your mind?”

    -- John Sebastian, *Did You Ever Have to Make Up Your Mind?*

What a joy it would be to remain in the fuzzy world and never decide, but sometimes we cannot resist the lure of
``bool`` and ``float`` objects, so we must derive them from our :class:`.Truth`\\ s and :class:`.FuzzyNumber`\\ s.
This process is usually called "defuzzification", an impossibly awkward word I've replaced with the more compact
and euphonious "crisp".  We'll bring our fuzzy results back into the crisp world by "crisping" them.

Crisping a fuzzy truth is extremely simple:  it is compared to a threshold, and if it is as true as that, we can
call it ``True``.  The :meth:`.Truth.crisp` method does this, and the built-in ``bool()`` function is overloaded to
do the same, using the :attr:`truth.default_threshold` automatically.

Crisping a fuzzy number can be complicated, and there are many ways to do it, so this is handled by a family of
classes in the :mod:`.crisp` module, which can be chosen as options in the :meth:`.FuzzyNumber.crisp` method.  The
default is a common one that I happen to think is the most reasonable:  it takes the median from among the global
maxima of the number's truths.

Finally, there is a way to bring the fuzziness itself into the crisp world.  A fuzzy number is defined by a function,
and sometimes that function can itself be a useful result.  Consider that it embodies the whole history of
deliberations that led up to it.  By using it to control a variable, we retain all of that information, transforming
it into subtle effects and nuanced behavior.   All that is necessary is to map its range from [0,1] to whatever your
real-world variable requires.  I can't anticipate all the ways of doing this, but I provide a handy
function, :meth:`.map` that maps the defining function of a :class:`.FuzzyNumber` (including its exceptional points
and "elsewhere" truth) onto whatever range you require via linear, logarithmic, or exponential mapping.  It produces
a callable object of type :class:`.Map` that you can use in crisp expressions, e.g.: ``y = my_map(x)``.

How does the fuzzy number come about?  How is it all calculated?  With difficulty, numerically, behind the scenes
where you needn't worry about it.  There are, however, two important parameters used when calling for a calculation
(i.e., when calling :meth:`.FuzzyNumber.crisp`) that you should set wisely:

    * ``resolution``: the largest discrepancy in result values that you would tolerate, an acceptable error-bound.
    * ``allowed_domain``: the range of values in which you are willing to receive a result.

Internally, the package uses a fixed precision based on these parameters and the natural domain of the expression in
question (which is ultimately some combination of the defined domains of all its literal operands).  In any case,
it only considers operand domains that will lead to results in your ``allowed_domain``, so it wastes no precision.
Therefore, setting this parameter helps to keep everything efficient.  It also guards against overflows in case
your expression is valid to infinity (e.g., the reciprocal of values approaching zero).


Suitability
===========

I've always thought of the unit of the fuzzy number function as "suitability":  among all possible values of a number,
it says how suitable each is for an intended purpose.  In this documentation, I call it a "truth" and avoided the
terms "suitability" and "fit" (for fuzzy unit, [0,1]) because I thought they might lead to confusion.
At this point though, I think it's illustrative.

Suppose you have performed some fuzzy reasoning that results in a crisp answer.  How good is it?  The fuzzy number's
truth at its crisp value is a measure of that.  Now is a good time to check it.  You can find the truth for any
value (or an Iterable of values) of a :class:`.FuzzyNumber` by using its ``.t(v)`` method ("t" for "truth").
Is it good enough?  Does it suit you?

Finally, :class:`.FuzzyNumber` has a handy :meth:`.display` method for inspecting the defining
function---visualized results can often be more valuable than single crisp answers.
It automatically sets the resolution to match your screen resolution for maximum efficiency.


Final Decisions
===============

If your results aren't true enough, or if they don't suit your purpose, you might decide to change the inputs and
try again. You might decide to zoom in with ``resolution`` and ``allowed_domain`` to get a more precise result, or
use them to examine a different region.  Your decision may bear on further steps in your algorithm.

To put it dramatically:---

    | ``give_me_a_yes_or_no_answer = my_truth.crisp()``: a bool
    | ``give_me_a_hard_cold_figure = my_value.crisp()``: a float
    | ``is_that_so = my_value.t(give_me_a_hard_cold_figure)``: a float
    | ``give_me_the_perfume_of_the_thing = my_value.map()``: Map

You might be curious about when all the calculation takes place.  Not until it is required, that is, not
until :meth:`.Operator.crisp` or :meth:`.display` are called.  Until then, an expression can sit in a
variable as a tree of relatively lightweight operators and literals.  For example:

    | ``expression = (a + b) & c ^ 50``
    | ---This can remain unevaluated while you do other things.
    | ``truth_at_v = expression.t(v)``
    | ---Obtains a ``float`` result symbolically or algorithmically and does not require numerical methods.
    | ``my_map = expression.map(...)``
    | ---A callable object to be used in crisp expressions, e.g.:
    | ``output_variable = my_map(v)``
    | ---A mapping of t(v) to some useful unit.
    | ``result_value = expression.crisp(...)``
    | ---The best crisp value, as a single ``float``.
    | ``expression.display(...)``
    | ---A picture of the fuzzy number.

Note that only the last two involve the relatively expensive construction of numerical representations.

Make all the natural-language statements you can about a system, its problems and heuristics.  They are already
very close to fuzzy statements, and are easily converted.  Your knowledge of a domain can quickly become a very
sophisticated computer program automating your own judgment.


Rare Parameters
===============

There are six rarely-needed parameters that can be given to some methods.  Usually they are not of interest, and one
allows them to be set automatically by global defaults---module attributes that act like environment variables
for the :mod:`fuzzy` package.  In method calls they are indicated by a uniform system of keyword arguments.
In fuzzy expressions, settings on an operator propagate downward to all the operands it owns, and so on, unless it
contains an operator with another explicit setting.

The same keyword arguments can be used to set the module attributes using the :func:`.fuzzy_ctrl` function.
The :func:`.fuzzy_ctrl_show` function prints them to the terminal for inspection.

The module attributes are:

    :attr:`fuzzy.norm.default_norm`
        Norms define the details of fuzzy truth tables.  They are needed for most math and logic operators.
    :attr:`fuzzy.truth.default_threshold`
        The dividing point between truth and falsehood in the fuzzy domain (default: .5). Its use
        by :meth:`.Truth.crisp`, ``bool()``, and :meth:`.weight`, as has already been described.
    :attr:`fuzzy.number.default_sampling_method`
        Sampling is necessary to make numerical representations of literals and to carry out some operations.
        The option describes the spacing of the points.  The default is ``"Chebyshev"`` (which results in  near-minimax
        polynomial approximations which are usually more accurate). The other option is ``"uniform"``.
    :attr:`fuzzy.number.default_interpolator`
        Interpolators are used in sampling and in the definition of :class:`.CPoints` literals. Ten types are provided.
        See :class:`.Interpolator`.
    :attr:`fuzzy.number.default_crisper`
        Crispers are not used in operators, but do the work of :meth:`.FuzzyNumber.crisp` and ``float()``.
        See :mod:`.crisp` for details.
    :attr:`fuzzy.number.default_resolution`
        Is used only by ``float()``.  When crisping, you really ought to give the resolution you need, because only
        you know the units of value you have in mind, and numerical methods without an error bound are meaningless.

This syntax of ``kwarg`` parameters is described in  :func:`.fuzzy_ctrl`.

.. figure:: ../../images/t-norm_gallery.png
   :alt: a gallery of t-norms

   A gallery of t-norms.  :meth:`.Truth.and_` is plotted for the eight built-in, simple norms.  Lax and drastic are
   fuzzy only on their very edges, a feature that doesn't show up here.



Norms require further comment.  First, there is the choice for **not**.  I use only the standard fuzzy negation: ¬s=1-s.
It is the only choice that is continuous, invertible (¬¬s=s), and compatible with crisp negation (¬0=1, ¬1=0).

Imagine the unit cube.  Two dimensions can represent *a* and *b*, the operands of a fuzzy binary operator.  The third
can represent *r*, the result.  So, any surface within the cube and over the *ab* plane describes a fuzzy truth table.
We require curves that are compatible with crisp truth tables, i.e., that have the value of crisp tables at the
corners, *a, b, r* = {0,1}.  In-between there is a lot of room for opinion, hence the variety of choices.

Truth tables are related, so their norm definitions must be compatible.  The **nor** and **nand** operators are both
functionally complete, so it would be sufficient to define only one of them and derive the other nine operators
of interest.  For efficiency, I describe a norm with two operators: **and** (called a "t-norm") and **or** (called
a "co-norm"); but they are not independent.  They are related by De Morgan's laws (¬(p∧q)=(¬p)∨(¬q); ¬(p∨q)=(¬p)∧(¬q))
so that the definition of any two of {¬, ∧, ∨} implies the definition of the third.  Therefore, because we have
settled on a definition for **not**, every t-norm unambiguously implies a dual co-norm.  I provide:

    * Eight standard norms:  lax, min-max (Gödel-Zadeh), Hamacher, product (Goguen), Einstein,
      nilpotent (Kleene-Dienes), Łukasiewicz, and drastic.
    * A parameterized version of Hamacher.
    * A "compound norm" mechanism to give the linear combination of any two norms.
    * The "strictness norm".

Strictness is my own characterization of a norm:  it is simply the percentage of the unit cube under the co-norm
curve.  Stricter norms tend to extreme values---**and** more false and **or** more true.


.. figure:: ../../images/strictness_norms.png
   :alt: a gallery of strictness t-norms

   Strictness.  :meth:`.Truth.and_` is plotted for the parameterized strictness norm from -81 to 73.  Strictness
   here is the fraction of the volume above the t-norm surface, mapped onto the parameter range, [-100, 100].
   Some t-norms bulge along the diagonal where a ≈ b.  Some exclude the false half of the unit square.  This is not
   captured by the strictness parameter.  The strictness norm I provide interpolates between the eight simple norms
   mentioned above.

There are more exotic norms than these in the world.  They can wait.


*********************
How the Package Works
*********************

Instead of a complete walk-through, I'll only mention some unusual points of interest.


The _Numerical Class
====================

The private class :class:`._Numerical` provides a standard form for the numerical representations.  It represents
a fuzzy number as a function (range on [0,1]) over all ``float``\\ s, by defining three kinds of domains, in order of
decreasing priority:

* A set of exceptional points, discrete (value, truth), :math:`(v,t)` pairs.  We can call the set of values the
  domain of discrete points, :math:`D_p`.
* A continuous function of truth vs. value, :math:`t(v)`, over a domain :math:`D_c`, represented by an array of values
  and an array of corresponding truths.  Its evaluation, therefore, requires interpolation,
  so there is a system of selectable :class:`.Interpolator`\\ s.
* A default truth, :math:`t_e`, (usually 0) to be reported for values not included in the above continuous
  or discrete domains.  We can call all the otherwise undefined values the "elsewhere" domain, :math:`D_e`.

The class also includes an :meth:`._impose_domain` method in case one needs to impose an extreme domain on a result,
e.g., if values outside a certain range are nonsensical.  This only discards exceptional points, not sample points
on :math:`D_c`, since that would only lower the quality, and it is not necessary, since unnecessary sample points
will not have been created, as is explained below.

In some ways, the "defined" domain (:math:`D_d  = D_p \\cup D_c`) is treated distinctly from the "undefined" domain
(:math:`D_e`), which acts as a sort of default truth to return when the number is queried for an out-of-range value.
In logical operations, it is treated as part of the number.  In arithmetic operations, it would cause misleading
results if it were allowed to operate with the :math:`D_d` of other numbers, so it is treated as a separate part of
the number.  In arithmetic operations, :math:`D_e` operate amongst themselves and not with :math:`D_d`.

I have allowed only one continuous domain per number in order to keep things simple.  I have considered multiple
continuous (but not overlapping) subdomains, and decided that this would create more complications than it would
remove.  So, a :class:`._Numerical` has the following elements:

* Zero or more exceptional points.
* Zero or one continuous domains.
* One elsewhere truth.

Our fuzzy numbers are defined as smaller domains sitting atop larger ones:  an infinite plane on the bottom;
then, a single, finite continuous domain amenable to numerical sampling; and, overlaid on top of these, a sprinkling
of exceptional points.


Fuzzy Logic on Numbers
======================

The :math:`D_c` of the operands are not necessarily identical.  I've dealt with this by partitioning their extreme
union at every operand domain boundary and sampling each resulting subdomain with the full precision of the call,
even if it only refers to the :math:`D_e` of the operands.  In such cases, I might consider, say, five points
sufficient, but I want to avoid oscillations between them.

For logical operators, every value in the combined :math:`D_p` of
the operands results in a new exceptional point with a truth that is the result of operating on :meth:`.Literal.t`
evaluations of all operands, i.e., The truth of at any exceptional point in the result is the result of operating on
the truths at that value in all the operands, whether they come from :math:`D_p`, :math:`D_c`, or :math:`D_e`.

Similarly, for every :math:`D_c` in the result, its samples are the result of operating on truths from
the :math:`D_d  = D_c \\cup D_e` of every operand (using the :class:`.Literal._sample` method);  :math:`D_p` is not
consulted because the points are *exceptions* to the continuous function.

Finally,  the truth on :math:`D_e` of the result comes from operating on the :math:`t_e` of all the operands.


Fuzzy Arithmetic
================

Traditionally, fuzzy arithmetic is defined by a few alpha cuts and some interval math.  This is awkward and misses
the great subtlety that's possible.  I have attempted a more complete and rigorous definition of the four
basic operators.

In general, since every possible value of the operands, :math:`a` and :math:`b`, have a truth, every possible pair of
values, one from :math:`a` and one from :math:`b`, have a truth that combines their individual
truths:  :math:`t_{ab} = t_a \\land t_b`.  You can visualize this as a Cartesian product, a plane indexed
by  :math:`(x,y)` coördinates, possible values of :math:`a` and :math:`b`.  For any operator, there are an infinite
number of pairs for each result, :math:`r = a * b`.  For each result, the truths of all its pairs are combined by
**oring** them all together:   :math:`t_r = \\lor t_{ab}\\,  |\\,  a * b = r`.  The fuzzy result, is then the
collection of (result value, truth) pairs.  This is the mathematics of the fuzzy operation.

The organization of the :class:`._Numerical` class complicates this picture somewhat.  It is conceived as layers
that are exceptions to those below: the continuous domain sits atop the "elsewhere" plane as an exception, and the
set of exceptional points sits atop this terrain.  We need a computation that achieves the equivalent of the above
mathematics using this data structure.  Each element of the two operands interacts to produce a new element and
these are combined in the resulting :class:`._Numerical` object.  The result of any operation with respect
to element types is:

    * Two points: another point.
    * Two continuous functions: another continuous function.
    * One point and one continuous function: another continuous function.
    * Two elsewhere truths:  another elsewhere truth.
    * An unpaired point set: another point set.
    * An unpaired continuous function: another continuous function.

All results of like type are then **ored** together to produce a complete :class:`_Numerical` object result.
I'll describe the above operations in turn.

Between Exceptional Points
--------------------------

Consider two numbers with only one point each, :math:`a: (v_1, t_1), b: (v_2, t_2)`.  Adding them would result
in one point:  :math:`(v_1 + v_2, t_1 \\land t_2)`.  The **and** operator is used to combine the truths of the two
specific instances of the operands because the result, :math:`(v_1 + v_2)`, is true only to the extent that
both :math:`v_1` **and** :math:`v_2` are true.

Consider two numbers with two points each, :math:`a: (1, t_1), (2, t_2);  b: (1, t_3), (2, t_4)`.  The combinations
of each pair from :math:`a` and :math:`b` have a result. The resulting points :math:`(2, t_1 \\land t_3)`
and :math:`(4, t_2 \\land t_4)` are not surprising, but two combinations lead to results in the same
spot:  :math:`(3, t_1 \\land t_4)` and :math:`(3, t_2 \\land t_3)`.  Either could have lead to the same result, and,
in fact, both are implied, so they are combined by **or**: :math:`(3, (t_1 \\land t_4) \\lor (t_2 \\land t_3))`.
Fuzzy truth is not probability, but it is similar in this way:  requiring both circumstances to occur
(:math:`a` **and** :math:`b`) suggests the intersection of their truths;  having several pathways to a result
(:math:`1+2=3` **or** :math:`2+1=3`) suggests their union.  It's intuitive that the former reduces the truth and the
latter increases it, just as it would with probability.

So, in general, the truths of every combination of :math:`a` **and** :math:`b` are found.
If there are multiple truths for one result, they are **ored** together.

If a point set occurs in one operand, but not the other, its truths are simply anded with the truth of the
corresponding values in the other operand, and the resulting (value, truth) pairs are the exceptional points of the
result.  (This would appear to introduce errors if the operand is a subtrahend or divisor, but as subtraction and
division will be recast as addition and multiplication, this is avoided.)


Between a Point and a Function
------------------------------

Between every point in one operand and the continuous function of the other, there is one continuous function
result.  The values of its sample points are :math:`v_p * v_i` (where ":math:`*`" signifies any binary operator) and
their truths are :math:`t_p \\land t_i`, i.e., the single point operates on every point of the function.
There may be several points in either operand, and so, several resulting continuous functions that may overlap.
These will be **ored** together with the result of the following section.


Between Two Functions
---------------------

This is simply a continuous version of the discrete case, with two sets of points.  Here now, there are an infinite
number of them, represented by sample arrays.

Consider the :math:`D_c` of our operands, :math:`a` and :math:`b`.  To form a sum, let's say, we take
every possible value in :math:`a` and :math:`b` that might come together to produce a result, :math:`r`.
Picture them as the Cartesian product of :math:`a` and :math:`b`, a rectangular plane on which each point uniquely
corresponds to one of the possible combinations.  The truth of that combination is the truth of *that* :math:`a`
**and** the truth of *that* :math:`b`.  So, over the rectangular area of the Cartesian product, we have a
surface :math:`t(a,b) = t(a) \\land t(b)`.

We are interested in specific results of the operation :math:`r = a * b`.  Each :math:`r` corresponds to a line
across the Cartesian product.  For example, if we are adding two numbers, the
result :math:`r = 4` corresponds to a line where :math:`b = 4 - a`.  Along this line there are an infinite number of
possible combinations, each with its own truth.  Every combination leads to the result, but only to the
extent that it is true.  At each point along the line, we may say "it is this one *or* some other".  And so, we
**or** them all together with the or-integral mentioned above.

Now we can calculate the truth of any result.  It is the or-integral along the line that describes the result over
the Cartesian product of the operands, using the truth of one operand **and** the other.  We must take care though,
that the or-integral proceeds through the line giving equal attention to each point.  So, the line must be described
parametrically and that parameter mapped so that its first derivative is constant---the or-integral proceeds at
an uniform pace over the line.  I.e., the sample points across the line are uniformly-spaced by
arc length;  Chebyshev points are not used in this application.

It is the business of an operator method to form the Cartesian product as a numerical matrix, fill it
with :math:`t(a,b) = t(a) \\land t(b)`, describe the correct line formula for the operator, and to sample the domain
of possible results at the appropriate points, or-integrating over the line for each :math:`r` sample.  The resulting
set of samples represents the continuous function of truth vs. value that defines the result.  The remaining
question is: at what values should :math:`r` be sampled?

My insistence on using only one :math:`D_c` introduces a slight complication to the sampling of the result.
A logical operation might create a fuzzy number that has features concentrated in small subdomains separated by vast
featureless subdomains.  If this number then enters into an arithmetic operation, a simple uniform or Chebyshev
sampling of the result will not do.  The sample points must be chosen carefully to avoid loosing detail.  First, a
sort of cross product is formed: every combination of sample points :math:`(v_a, v_b)` is considered.  These each have
a result :math:`r = v_a * v_b`.  Duplicate results are discarded and this gives us a set of sample values that ensures
sufficient detail in the result where there is structure in the operands.  However, to avoid the number of sample
points from ballooning with each operation, we must winnow this set by discarding points so that the total
number of points in the result does not greatly exceed the maximum among its operands.  So, the sample set is decimated
uniformly; areas of greater structure are still more densely populated.

Finally, all of the continuous function results (this one and the ones from the previous section)
are **ored** together to produce a resulting continuous function element.


If a continuous part occurs in one operand, but not the other, its truths are simply *anded* with the elsewhere truth
of the other operand, and this becomes the continuous part of the result.  (As with exceptional points, the recasting
of subtraction and division avoids errors where the unpaired element is a subtrahend or divisor.)

Operations on Elsewhere
-----------------------

That leaves the vast "elsewhere" territory, outside the Cartesian product. What if we admitted arithmetic operations
between this region and the defined domains, :math:`D_d`?  the or-integral along a result line across
the infinite plane may encounter some other truths over finite regions, but they are proportionally infinitesimal.
The result would be dominated by the infinite number of elsewhere points and the defined points would be insignificant.
We would end up **oring** together an infinite number of elsewhere truths, :math:`t_e`.  In most cases, all the
information in the defined region would have to be thrown away.

My solution is to only consider :math:`D_e` separately:  the two areas---defined and undefined---operate only among
their own kind and not with each other.  I justify this by saying that the large undefined area does not affect
the small defined area because the smaller is an exception to the larger.  In the same way that the exceptional points
are conceptually added after the continuous domain, and do not affect the sampling of the domain, so the
continuous domain is independent of the default truth, and sits on :math:`D_e` as an exception.

Following the definition of the or-integral to be explained below, the elsewhere truths are combined
by :math:`t_r = (t_a\\land t_b)\\lor(t_a\\land t_b)`.  This is the limit of the or-integral when we consider that
their truths over all real values are nearly constant and equal to their "elsewhere" truths.

It would be simpler to make the elsewhere truth always zero.  In practice, it usually will be.  What are the uses
of a non-zero elsewhere?  I can think of two:

* If :math:`t_e = 1`, the fuzzy number can be used as a mask that discourages some possibilities.
* If it is very small, say If :math:`t_e = .001`, it allows one to proceed even if all defined values become false.


The Operators
-------------

Addition is our model.  Multiplication is defined by the addition of logs.  Subtraction is recast as the addition of
the first operand to the negation of the second.  Division is recast as the multiplication of the first operand by the
reciprocal of the second.  Associative operations (those that can take more than two operands) are simply computed
as chains of binary operation.


The Or-integral
---------------

I need something like an integral that, instead of summing an infinite number of infinitesimals, **ors** together
an infinite number of truths of infinitesimal significance.  I don't have a rigorous definition, but I know how such
a thing should behave:

* The result cannot be less true than the truest point on the integrand, its maximum.
* The result approaches 1 as the average of the integrand approaches 1.
* The speed at which it approaches 1 must depend on the norm (faster for stricter norms).
* The average truth of the integrand cannot exceed its maxiumum.

For our practical purposes, I define the or-integral as:

.. math::

    \\lor_0^\\ell\\, t(x)\\, dx =\\, M \\lor A

where :math:`A = (1/\\ell) \\int_0^\\ell\\, t(x)\\, dx`
and :math:`M = \\max\\left(t(x)\\right)`---the average and maximum truths on :math:`[0, \\ell]`.

The result for a single value of non-zero truth is that truth.  For a constant truth, :math:`t`,
it is :math:`t \\lor t`.  For a truth function between these extremes, its result varies between the above results
depending on the strictness of the norm.  As the average approaches 1, so does the result.  Therefore, the above
assumptions about how the integral must behave are satisfied.


Problems in Real Analysis
-------------------------

The analysis of this system could use some review:

    * Is there a practical formulation of or-integrals that is more rigorous?
    * In arithmetic operations I treat the elsewhere region as not really part of the number.
      Is there a better solution?

These are questions for a real mathematician.


Fuzzy Expressions as Trees
==========================

I want fuzzy numbers to be defined powerfully, compactly, and accurately, with all the facilities of a Python function,
however, most operations need to be carried out numerically.  So, the public sees :class:`.Literal`\\ s, with simple
parameters, and can make new ones by defining a single Python method (:meth:`.Literal._sample`).

They also use operators that overload convenient symbols, so that everything looks like crisp math with logic thrown in.
These operators are methods of :class:`.Operator` that simply return instances of :class:`.Operator` subclasses,
one for each type of operator. Though I don't expect it to happen, the operators have been designed so that it should
be almost as easy for users to create new ones as it is for them to create new :class:`.Literal`\\ s.

Since an :class:`.Operator` represents a :class:`.FuzzyNumber` it *is a* :class:`.FuzzyNumber`.  It *has* within it
its operands, which are also :class:`.FuzzyNumber`\\ s---:class:`.Literal`\\ s or :class:`.Operator`\\ s---so an
expression builds up a tree, but the tree can be held in a variable just so, as objects containing objects that
ultimately lead to :class:`.Literal`\\ s---which can give precise results for :math:`t(v)`, defined symbolically
and algorithmically, not dependent on numerical approximations.  This is what
the :meth:`.FuzzyNumber.t` method does, and it can do so for arrays of values as well.  The :meth:`.FuzzyNumber.map`
method packages the same behavior into a callable object for convenient use in crisp expressions.

Internally, though, the operators do most of their work numerically, so every :class:`.FuzzyNumber`,
whether :class:`.Literal` or :class:`.Operator`, must be able, when called upon, to return a numerical representation
of itself, a :class:`._Numerical`.  This representation, detailed above, is prepared for a given precision
(number of continuous sample points) when the query is made.  I'm guessing that, in most cases, the tree of objects
will be smaller than the numerical representation of the expression.  In any case, the required precision will not
generally be known until it is time for calculation.

In :class:`.Literal`\\ s, the numerical representation is prepared by sampling the defining
function, :meth:`.Literal._sample`.  In :class:`.Operator`\\ s, it is prepared by obtaining the numerical
representation of each operand, performing the operation, and returning the result.  So, when a numerical is called for
(by :meth:`.FuzzyNumber._get_numerical`), calls go down the tree and :class:`._Numerical`\\ s bubble up to the top.

Users never have to make this call directly.  It happens internally only when it is needed---only
when :meth:`.FuzzyNumber.crisp`, ``float()``, or :meth:`.display` are called.  Calls to :meth:`.FuzzyNumber.t`
and :meth:`.FuzzyNumber.map` (and to the resulting :class:`.Map` callable object) are all performed non-numerically,
relying ultimately on defining methods of :class:`.Literal`\\ s and individual floating-point operations,
without the need for the elaborate sampling and calculation required to create a :class:`._Numerical`.  (I also
provide an analogous method, :meth:`.FuzzyNumber.numerical_map` which creates a :class:`.NumericalMap` callable object
that *does* store the expression numerically---to accommodate any odd cases where this may be more efficient in terms
of speed or memory.)


Resolution, Fixed-precision, and Domain Restriction
===================================================

How many sample points (what *precision*) should you use for your calculation?  It's rarely obvious.  What you almost
certainly know, however, is the maximum error in value that you are willing to accept for the units and calculation
in question.  You know the resolution you need.  When calling :meth:`.FuzzyNumber.crisp`, you should give this as a
parameter.  If you do not, or if you call ``float()``, the package default will be used.  It is a bad idea to rely on
this unless you are doing many calculations of the same kind and set the package default accordingly.

Internally, :meth:`.FuzzyNumber.crisp` calls :meth:`._expression_as_numerical`, which queries down the tree to find the
natural domain of the result, as defined by the domains of all the :class:`.Literal` leaves operated upon by all the
:class:`.Operator` branches, with the same cascade of calls and responses used when :meth:`.FuzzyNumber._get_numerical`
is called.  The natural domain of the expression is then divided by the resolution to determine the precision
needed.  Actual sampling is done with guard points outside the domain, one on either side, in order to ensure good
interpolation near the edges.  The sample points are chosen either uniformly or at Chebyshev collocation points.
This latter is the default, and it ensures that the interpolating polynomial will have the near-minimax
property---it minimizes the maximum error and wiggles less between the points.  It's probably the best choice in
most cases, unless your function has some very narrow spikes.

There is more to this than I have told you.  It may be that you are only interested in a subdomain of the expression's
natural domain, perhaps because results outside of it would be nonsensical or impractical for you to act upon.
By setting the ``allowed_domain`` parameter in the call, you sent into motion a series of events.  First, the
intersection of your domain and that of the expression will be found, and this is what is used to calculate the
precision.  Next, in the cascade of calls down the tree seeking a :class:`._Numerical`, each operator will be told
the domain of results that are sought,  do the reverse of the usual interval math to find the corresponding domain
of interest in its operands, and communicate that to them.  When the calls finally reach :class:`.Literal`\\ s and it
is time for them to sample themselves, they will only sample the subdomain of their defined domain that would have
a result in the ultimate domain you have requested.  Therefore, all the precision and calculation is efficiently
focused only on the regions where it will matter.

"""
__all__ = ["norm", "truth", "number", "literal", "operator", "crisp"]
