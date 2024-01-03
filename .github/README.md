# Meet Írim

## What is Írim?

Írim is a composer\'s workshop. The goal is to maximize the beauty,
richness, and variety of music. The method is a music theory based on
human cognition, so that it is supremely *effective*. The software that
automates that theory is Írim. For more details (and a prototype), see
the [Odracam website](https://www.odracam.us).

## How Does One Use Írim?

Music-making is too complex to be covered by a single program, or even
by a single language. Írim is a bundle of languages for making
(composing, performing, synthesizing, and producing) music. Composers
write scripts in these special-purpose languages to create musical
objects of particular types\-\--familiar things like rhythms, melodies,
harmonies, scales, tunings, forms, and scores. Often, objects of one
type are used to create another, e.g., a harmony may be composed to
complement a melody (harmonization); one melody may be composed in
response to another (variation, development, contrast), or to play
simultaneously with it (counterpoint), etc.

Finally, a composer assembles musical material into a score, and sends
it to be performed by virtual performers (who automatically add artistic
nuance) on synthesized instruments (in a [Csound](https://csound.com/)
orchestra, with possible
[VSTi](https://en.wikipedia.org/wiki/Virtual_Studio_Technology)
plugins), rendering audio files for each instrumental part. In a
production stage, the composer then adds 3D spatial effects to these,
positioning (and possibly moving) them on the soundstage in an ideal
virtual hall. The final output is an audio file prepared with
[binaural](https://en.wikipedia.org/wiki/Binaural_recording) or
[Ambisonic](https://en.wikipedia.org/wiki/Ambisonics) techniques for
headphone or multichannel reproduction. Thus, a composer\'s entire
workflow, from theoretical ideas to finished recording, can be
accommodated by Írim.

Language and text are the most powerful, compact, and efficient means
for expressing ideas and for controlling software, but music is a
complex subject, requiring many small, interoperating
languages\-\--e.g., those for describing a meter, a note, a performer\'s
personality, the constraints for a tuning, the criteria for writing a
melody, an instrumental timbre, an arpeggio figure, a conductor\'s
instructions, a strum, a musical form, and so on. A composer needs the
aid of an IDE (integrated development environment) to write scripts of
many kinds and to organize them together into projects\-\--hence, a
central GUI (graphical user interface) for using Írim. I also hope,
wherever possible, to provide graphical editors, or, at least, displays,
to aid script editing.

## Who Might Use Írim?

Generally, anyone who might want to make music, but I have a special
place in my heart for anyone with the insight, whether natural or
gleaned from experience, to see that every stage of music-making is an
engineering problem, best solved by engineering methods.

In particular, the theories behind the software have a relentless focus
on the experience and cognition of listeners, not on those of
performers. For practical reasons, performers must pay most of their
attention to aspects of music hardly noticed by listeners, and so end up
experiencing it differently than the audience and focusing on
theoretical constructs of little relevance.

Similarly, many technical people approach music with a sublime grasp of
engineering technique, but no knowledge of how or where to best apply it
to the domain area (and so sometimes build elaborate systems that give
listeners little satisfaction). Yet, they are often music-lovers who
would like to become composers.

So, many may wish to use Írim:

> -  Music professionals:
>
>     :   -   Producers who want the facility of mixing in space.
>         -   Composers with scores who want the facility of virtual
>             performers and instruments.
>         -   Composers who want a computer to aid their compositional
>             tasks.
>
> -   Souls with seeking hearts:
>
>     :   -   Musicians who sense that something greater is possible.
>         -   Engineers who want to make music.

To musical people: What you will find in Írim is not an alien world.
Your familiar musical world, in tones of light and dark grey, will be
rendered in Technicolor, revealing a thousand wonders that had been
hidden before.

To technical people: You are my people. I speak your language. Lucky for
us: composition is engineering. We are going to have a good time!

## How to Use the Code

It doesn\'t do much yet (see `the roadmap<roadmap>`{.interpreted-text
role="ref"}). When it does, I\'ll put installation instructions here.

For now, if you have a modicum of tech savy, you can try the old version
of the [rendering
orchestra](http://odracam.us/index.php/about/the-software), which can do
a lot of interesting things, and can be very useful if you already know
how to write music.

## How You Can Help

The foundation of the work is an understanding of how musical perception
works. The best information about that comes from the field of Music
Cognition and related scientific disciplines. In the many places where
this knowledge is incomplete, we must turn to a broad comparison of
existing musical traditions. I am always learning new things, but I have
been at the subject for a long time, and I understand it very well.

The software includes algorithms that make use of technologies like
fuzzy logic and math, optimization, generative grammar, Markov chains,
signal processing, and so on. I understand these fairly well, but might
benefit from help in these areas, or in similar technologies I\'m
unaware of. Note: I\'m not interested in LLM AI or Data Science for
several reasons:

> -   Most musical practice today and in history depends on musicians
>     imitating each other. This is boring, ineffective, and does not
>     explore new territory. Algorithms that imitate existing music
>     propagate the problem.
> -   Trendy techniques (LLM and its many predecessors) are \"that one
>     wierd trick\" to program complex behavior without the programmer
>     having to learn about the domain area. It\'s always shallow and
>     disappointing.

I\'m interested in technologies that allow expert knowledge about the
domain to be expressed by humans and applied by the machine. This
requires someone to have expertise, and to apply effort to express it,
but it avoids many problems, including the imitation of bad models and
the failure to create ideas that are both novel and effective.

The place where I need the most help is down in the nuts and bolts. I\'m
well acquainted with basic programming and the OO paradigm, but my
background is not in Computer Science and modern, professional software
development is new to me. I\'m feeling my way through Python, Pycharm,
CVS, Github, automated testing, and all that jazz. In many cases, I
might have a good algorithm but poor implementation\-\--I\'d love to be
told when I could do something more robustly and efficiently. I welcome
any contribution, but I\'d especially love help with the details and the
mechanics of the software engineering itself. Feel free to look over my
work and let me know what can be improved. If you are interested in the
project, feel free to contact me.

## The Roadmap {#roadmap}

This is a sketch (for now, very rough) of the work I plan to do. If you
are a musician and not an engineer, don\'t be alarmed by the
jargon\-\--this is a description of the programming job; the objects and
structures users will deal with are mostly familiar to musicians
already. Most of the code will be in Python, but the audio rendering
will be done in Csound.

> -   First, as a warm-up, I\'m building some of the mathematical
>     infrastructure I need. The largest item is
>     `fuzzy`{.interpreted-text role="mod"}, a package for fuzzy logic
>     and arithmetic. If you are familiar with the subject, you might
>     find that my formulation of it is a bit novel and, I hope,
>     interesting.
>
> -   The first major section that must be built is the code that takes
>     a text-based score and renders it as an audio file. (A
>     [prototype](http://odracam.us/index.php/about/the-software),
>     written entirely in Csound, may give you an idea of the basic
>     functionality required.) This is a problem with many parts:
>
>     > -   Code for working with musical form: a sort of structured
>     >     container for holding musical objects, to be useful later
>     >     for holding musical data and scripts during the composition
>     >     process\-\--a score is the same sort of container for
>     >     holding performance instructions.
>     > -   Code for working with the physical and musical lengths of
>     >     sections, average tempi, tempo curves, resolving conflicting
>     >     directives about these and finally drawing the time map.
>     > -   Code for working with meters: objects for structuring time
>     >     and scheduling musical parameters.
>     > -   Code for interpreting notes and things that evaluate to
>     >     notes (strums, arpeggio figures, ornaments), and for working
>     >     with chunks of these and translating them into sequences of
>     >     playable Csound statements.
>     > -   Code for working with scale, dissonance, and tuning, which
>     >     includes some interesting optimization problems, and a
>     >     little spectral analysis.
>     > -   Code for virtual performers: for generating bundles of
>     >     control signals based on individual gestures and a language
>     >     for describing fuzzy inference engines to make the decisions
>     >     about this. (And this would benefit from experienced
>     >     performers stating heuristics in that language.)
>     > -   Code for virtual conducting: for analyzing the score and
>     >     creating control signals based on the analysis.
>     > -   A Csound orchestra for rendering instrumental parts, with
>     >     instruments for different synthesis techniques. One of them
>     >     should load VSTi plugins and translate the control signals
>     >     into MIDI data\-\--probably the only use of MIDI in the
>     >     project.
>
> -   A realization of [Mannerism]{.title-ref}, a language that combines
>     generative grammar, Markov chains, and fuzzy logic to generate
>     sequences of abstract symbols. These sequences can be directly
>     translated to and from melodies, harmonies, rhythms, and chunks of
>     form. So, scripts written in this language
>     ([modes]{.title-ref}\-\--manners of composition) are the main
>     generators of musical material.
>
> -   The [HW]{.title-ref} (harmonic world) program\-\--that generates a
>     database of sonorities and measures their perceptual qualities
>     (\"harmonic flavors\") and those of their transitions. This is a
>     combinatorial problem, so keeping the size manageable will be
>     important.
>
> -   The [HC]{.title-ref} (harmony and counterpoint) program\-\--that,
>     given composer criteria, finds optimal sequences of sonorities
>     drawn from an [HW]{.title-ref} database, optionally unweaving them
>     into simultaneous melodic lines. This is an optimization problem
>     akin to the travelling salesman problem.
>
> -   Some minor programs for scale and tuning design, e.g., for the
>     artful use of dissonance as in polychrome, hypertonal
>     temperaments.
>
> -   Some minor melody and counterpoint programs, e.g., for generating
>     canons.
>
> -   Development of more programs written in [Mannerism]{.title-ref},
>     e.g., the schema for partimento improvisation, jazz, etc.
