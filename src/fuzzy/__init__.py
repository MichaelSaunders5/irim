"""provides all the facilities for working with fuzzy logic and arithmetic.

.. epigraph::

   | “If a man will begin with certainties, he shall end in doubts;
   | but if he will be content to begin with doubts, he shall end in certainties.”

   -- Francis Bacon, *Of the Proficience and Advancement of Learning, Divine and Human*


Introduction
------------

Fuzzy techniques are for expressing expert knowledge and for making reasoning with it computable.  It automates
the type of judgement and behaviour that humans are particularly adept at.  It is good at dealing with imprecise
or incomplete information, conflicting rules, opinions, systems of heuristics, vague notions, subtle influences,
nuance, personal preference, ambiguity, combined ideas, style, perceptual data, subjective experience, whims,
and fancies.  All of this is in addition to a perfectly fine ability to work with hard facts and precision---the
fuzzy is a superset which includes the crisp.

Fuzzy techniques deal with truths as "fits"---just as "binary units" are called "bits" and have a domain of {0,1},
"fuzzy units" are called "fits" and have a domain of [0,1].  (I resist the urge to turn "bool" into "fool".)
So, whereas crisp logic defines operators with truth tables, fuzzy logic must define them as functions mapping
one or more domains of [0,1] onto a range of [0,1].

To extend this idea to arithmetic, consider our familiar, "crisp" numbers.  Each is really a statement saying:
"of all the real numbers, this one, and only this one, is presently the case (it is completely true)
and all others are not (they are completely false).  A single fuzzy number, on the other hand, indicates the degree of
relative truth or falsehood for every member of the real numbers.  I like to think of this as the measure of
the suitability of a proposed value for a given purpose---some values may be absolutely suitable or unsuitable
(1, or 0), and some may be somewhere in-between.  With fuzzy numbers represented as arbitrary functions,
arithmetic operators on them become something rather complicated, but the :mod:`fuzzy` package encapsulates
all the difficulty into code hidden behind familiar symbols.

Problems may be expressed in the familiar crisp form; solved with the usual techniques of logic, mathematics,
and so on; and their solutions also expressed in the familiar form:  with crisp variables and operators.
When the :mod:`fuzzy` package is used, the very same expressions become "overloaded" with fuzzy meaning, and
complex, organic, artistically-inflected behaviours emerge.  And so, fuzzy techniques fit easily into algorithms
almost anywhere.  Beyond this, they allow problems and solutions to be formulated in a more daring way:  experts
may express their knowledge of the domain in heuristics which the technique makes computable.

Most implementations of fuzzy techniques focus on logic more than math, typically representing quantity crudely
as trapezoidal numbers operated on with interval arithmetic.  This is adequate for simple control systems.
The present :mod:`fuzzy` package, however, represents numbers as arbitrary functions and provides rigorously-derived
operators that preserve their detail.  This allows individual numbers to convey the subtlety of expression
necessary for both scientific precision and artistic richness.

The package's three modules house the three main families of classes:  :class:`.Norm`, :class:`.Truth`,
and :class:`.Value`.  It's unlikely that you'll ever need to change their defaults or to use :class:`Norm`
objects directly.  Almost all fuzzy reasoning can be performed with :class:`.Truth` and :class:`.Value` objects
using common (overloaded) operators---so it will look much like ordinary logical and arithmetical expressions.



How to Use the Package
----------------------

There are two ways:

* The hard way: create a :class:`.Norm` object and use its functions as logic and arithmetic operators.
* The easy way: set the :attr:`default_norm` and :attr:`global_defuzzifier`  (if you aren't happy with the defaults)
  and use overloaded operators on :class:`.Truth` and :class:`.Value` objects.

The hard way
............

To use the operators you must first create a Norm object (see the factory method, :meth:`.Norm.define`).
This defines the t-norm/co-norm pair that defines logic and arithmetic operations.

Example:
    ``n = Norm.define()``

This object has the fuzzy operators as methods.  For logic operators, you call it to operate on *fits*
(fuzzy units: floats on [0,1]), Numpy arrays of fits, or :class:`Value` objects (which can represent fuzzy numbers).

Example:
    | ``a, b = .2, .8``
    | ``print(n.not_(a), n.and_(a, b), n.or_(a, b)``
    | yields: .8, .16, .84.

For arithmetic operators, you call it to operate on :class:`Value` objects.


The easy way
............


How the Package Works
---------------------

very well?

overviews of the three families ...
"""
__all__ = ["norm", "truth", "value", "crisp"]
# 291.4 9/3
