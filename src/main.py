# Here is where I am testing or playing around or something.
# from fuzzy.operator import *

import numpy as np

from fuzzy.literal import *
from fuzzy.operator import *
# from fuzzy.truth import *
from math import log, exp



# fuzzy_ctrl(norm={'n1':"str", 'n1p':[-80], 'n2':"hhp", 'n2p':[20], 'cnp':70})
fuzzy_ctrl(norm={'n1': "mm"}, threshold=.5)       #  , r_precision=100000
fuzzy_ctrl_show()
n = Norm.define(n1="pp")
a = Truth(.6)
b = .6
w = 50
tw = Truth(.75)
ta = np.linspace(0,1,5)


print(f"t//f {a // w}")
print(f"C tf {Truth.weight(a, w)}")
print(f"tf {a.weight(w)}")

print(f"t//t {a // tw}")
print(f"C tt {Truth.weight(a, tw)}")
print(f"tt {a.weight(tw)}")


print(f"f//f {b // w}---bogus, because it's real floor division.")
print(f"C ff {Truth.weight(b, w)}")   # make this work

print(f"f//t {b // tw}")        # this thinks it's floor division, but it shouldn't be.
print(f"C ft {Truth.weight(b, tw)}")


# t = Triangle(-.9,0.1,1.1, elsewhere=0, points=[(2, .8)])
a = Triangle(0, 2, 4, elsewhere=0)
b = Triangle(-2, 0, 2, elsewhere=0)
# c = Sigmoid(">", 0, domain=(-5,5))
# print(f">0?: -6: {c.t(-6)},  -5: {c.t(-5)},  -1: {c.t(-1)},  -.5: {c.t(-.5)},  0: {c.t(0)}, ")
# print(f">0?: .5: {c.t(.5)},  -5: {c.t(1)},  -1: {c.t(1)},  5: {c.t(5)},  6: {c.t(6)}, ")


# a = Triangle(1, 2, 3)
# b = Triangle(-1, 0, 1)
c = a / b
e = c._expression_as_numerical(.01, (-8,8))
e.display(None, Domain((-8,8)))
d = b / a
f = d._expression_as_numerical(.01, (-4,4))
a.display()
f.display()

# 2 * 0: some ripples about 0, but 1 at 0.
# 0 * 2: same ripples, but dramatic divot: .93 at 0
# sampling on axis with _any_ epsilon, however close to 0 solves this, but for slopes crossing 0 there's still a thing.

# a = Trapezoid(-5, 4.9, 5.1, 6, elsewhere=0)      # , points=[(3, .2), (8, 1)]
# a = Trapezoid(-.5, .49, .51, .6, elsewhere=0)      # , points=[(3, .2), (8, 1)]
# a = Trapezoid(1, 1.49, 1.51, 2)      # , points=[(3, .2), (8, 1)]
# a  = Trapezoid(2, 4, 6, 8, elsewhere=0, points=[(3, .2), (2.5, .5)])  #, points=[(0,.5),
# (1,.5), (2,.5), (3,.5), (4,.5), (5,.5), (6,.5), (7,.5), (8,.5), (9,.5)]
# c = Trapezoid(4, 6, 8, 10, elsewhere=0)
# a = Bell(4, 3, 1)
# b = Bell(8, 3, 1)

# n=5
# xs=[-10,-.1]
# x = xs[1] - np.exp(np.linspace(log(xs[1] - xs[0] + 1), 0, n)) + 1
# x = xs[1] - np.exp(np.linspace(log1p(xs[1] - xs[0]), 0, n)) + 1
#
# print(x)

# a1 = Triangle(.1, 2, 4 , points=[.5,.3])     # , points=[.5,.3]
# a1 = Triangle(-1, 1, 3 , points=[.5,.3])     # , points=[.5,.3]
# a1 = Triangle(-5, -3, -1)     # , points=[.5,.3]
# a2 = Triangle(1, 3, 5)
# d1 = DPoints([(1,.9), (2,.2)])
# d2 = DPoints((2.5,.1))
# d3 = DPoints([(1,.3), (2,.5)])
# ex = Exactly(5)
# tr = Truthy(.5)
# b = a * a * a     # 1, .03;  3, .06|.05 = , 4, .1?
# c = a + a + a
# b = Operator.div(a, a)  #, allowed_domain=(35,65), allowed_domain=(-1,1), allowed_domain=(-.01,.01)
# c = a + b
# d = b + a

# b = Operator.add(a, a, a)
# print(c)
# x = 0
# print(f"b({x}): {b.t(x)}")
# c = b._expression_as_numerical(.1)
# print(f"c({x}): {c.t(x)}")
# c = b._expression_as_numerical(.01)
# print(f"c({x}): {c.t(x)}")
# b.display() # -1, ~.1





# development of r.xv, r.xt method:
# axv = np.array([1, 2, 3])
# bxv = np.array([1, 2, 3])
# axt = np.array([.1, .2, .3])
# bxt = np.array([.4, .5, .6])
# x, y = np.meshgrid(axv, bxv)
# print(f"x, y:  {x}, {y}\n\n")
# xabv = x + y
# x, y = np.meshgrid(axt, bxt)
# n = getattr(fuzzy.norm, "default_norm") #
# xabt = np.ndarray.flatten(n.and_(x, y))
# xv,i = np.unique(xabv, return_inverse=True)
# rlen = len(xv)        # np.max(i) + 1
# xt =  np.ndarray((rlen,))
#
# print(f"xabv = {xabv}")
# print(f"xabt = {xabt}")
# print(f"i = {i}")
# print(f"rlen = {rlen}")
# for j in range(0, rlen):
#     k = np.atleast_1d(np.where(i==j))[0]
#     print(f"j={j}, -- {k} -- truths -- {xabt[k]}")
#     xt[j] = n.or_(xabt[k])
# print(f"xv = {xv}")
# print(f"xt = {xt}")
#
#

# The following test different operator calls, especially operand promotion
# x = 4
# instance_call = a.imp(b)
# class_call = Operator.imp(a, b)
# symbol_call = Truth(.3) >> b
#
# print(f"a, b: {a.t(x)}, {b.t(x)}")
# print(f"instance, class, symbol: {instance_call.t(x)}, {class_call.t(x)}, {symbol_call.t(x)}")
# n * n == Python math.  imp is ~a|b, so .3>>.5 = .85 correct!  .5>>.3 = .65  backwards!
# x, y = 3, 5
# xir, yir = .3, .5   # number in [0,1] range
# Tx, Ty = Truth(.3), Truth(.5)
#
# print(f"\nT * T\n Tx & Ty: {~(Tx & Ty)}")                    # T * T
# print(f"\nnir * T\n xir & Ty: {~(xir & Ty)}")                     # nir * T
# print(f"\nT * nir\n Tx & yir: {~(Tx & yir)}")                     # T * nir
#
# print(f"\nn * T\n x & Ty: {~(x & Ty)}")                     # n * T
# d = ~(x & Ty)
# d.display()
# print(d.t(0))
# print(f"\nT * n\n Tx & y: {~(Tx & y)}")                     # T * n
# d = ~(Tx & y)
# d.display()
# print(d.t(0))
#
# print(f"\nT * F\n Tx & a: {~(Tx & a)}")                       # T * F
# d = ~(Tx & a)      # x & 1 = 1
# d.display()
# print(d.t(20))
#
# print(f"\nF * T\n a & Ty: {~(a & Ty)}")                       # F * T
# d = ~(a & Ty)      # 1 & .5 = .5, .5&.5= .75
# d.display()
# print(d.t(20))
#
# print(f"\nF * F\n a & b: {~(a & b)}")                                       # F * F
# d = ~(a & b)
# d.display()
# print(d.t(20))
#
# print(f"\nF * nir\n a & y: {~(a & yir)}")                                     # F * nir
# d = ~(a & yir)
# d.display()
# print(d.t(20))
#
# print(f"\nnir * F\n x & a: {~(xir & a)}")                                     # nir * F
# d = ~(xir & a)
# d.display()
# print(d.t(20))
#
# print(f"\nF * n\n a & y: {~(a & y)}")                                     # F * n
# d = ~(a & y)
# d.display()
# print(d.t(20))
#
# print(f"\nn * F\n x & a: {~(x & a)}")                                     # n * F
# d = ~(x & a)
# d.display()
# print(d.t(20))


# a.display()
# instance_call.display()
# class_call.display()
# symbol_call.display()
# t = Not(a)
# t = Negative(a)
# t = Reciprocal(a)
# t = Absolute(a)
# t = Imp(a, b)
# t = Con(a, b)
# t = Iff(a, b)
# t = Xor(a, b)
# t = Nand(a, b)
# t = Nor(a, b)
# t = Nimp(a, b)
# c = Ncon(a, b)
# t = And(a, b, c, norm={'n1':"str", 'n1p':[-80], 'n2':"hhp", 'n2p':[20], 'cnp':70})
# #, norm={'n1':"str", 'n1p':[-80], 'n2':"hhp", 'n2p':[20], 'cnp':70})
# t = Or(a, b, c)

# print(t)

# t.display()

# print(t.t(1))
# z, p, o = Truth(0), Truth(.5), Truth(1),
# print(f"a>>b: (0, {t.t(0)} = {z>>z},  ")
# print(f"a>>b: (1, {t.t(1)} = {p>>z},  ")
# print(f"a>>b: (2, {t.t(2)} = {o>>z},  ")
# print(f"a>>b: (3, {t.t(3)} = {o>>p},  ")
# print(f"a>>b: (4, {t.t(4)} = {o>>o},  ")
# print(f"a>>b: (5, {t.t(5)} = {p>>o},  ")
# print(f"a>>b: (6, {t.t(6)} = {z>>o},  ")
# print(f"a>>b: (7, {t.t(7)} = {z>>p},  ")
# print(f"a>>b: (8, {t.t(8)} = {z>>z},  ")
# print(f"a>>b: (9, {t.t(9)} = {z>>z},  ")
# print(f"Imp object:---\n  {t}")

# t = t._expression_as_numerical(.01)
# print(f"_expression_as_numerical:---\n    {t}")
# d = t._get_domain()
# print({d})
# print(f"_expression_as_numerical._get_domain:---  {d}")domain=(1.95,2.05)
# a.display()
# b.display()
# t.display()

# a_function = Triangle(1, 2, 3)
# tr = Trapezoid(1, 2, 3, 4)
# a_point = Exactly(8)
# a_function.xv, a_function.xt = a_point.xv, a_point.xt
# print(a_function.t(7))
# print(a_function.t(8))

# f = Truth(.1)
# mf = Truth(.4)
# m = Truth(.5)
# mt = Truth(.6)
# t = Truth(.9)
# print(f"10 = {f.not_()}, {t.not_()}")
# print(f"0001 = {f.and_(f)}, {f.and_(t)}, {t.and_(f)}, {t.and_(t)}")
# print(f"0111 = {f.or_(f)}, {f.or_(t)}, {t.or_(f)}, {t.or_(t)}")
# print(f"1101 = {f.imp(f)}, {f.imp(t)}, {t.imp(f)}, {t.imp(t)}")
# print(f"1011 = {f.con(f)}, {f.con(t)}, {t.con(f)}, {t.con(t)}")
# print(f"1001 = {f.iff(f)}, {f.iff(t)}, {t.iff(f)}, {t.iff(t)}")
# print(f"0110 = {f.xor(f)}, {f.xor(t)}, {t.xor(f)}, {t.xor(t)}")
# print(f"1110 = {f.nand(f)}, {f.nand(t)}, {t.nand(f)}, {t.nand(t)}")
# print(f"1000 = {f.nor(f)}, {f.nor(t)}, {t.nor(f)}, {t.nor(t)}")
# print(f"0010 = {f.nimp(f)}, {f.nimp(t)}, {t.nimp(f)}, {t.nimp(t)}")
# print(f"0100 = {f.ncon(f)}, {f.ncon(t)}, {t.ncon(f)}, {t.ncon(t)}")
#
#
# print(f"10 = {~f}, {~t}")
# print(f"0001 = {f & f}, {f & t}, {t & f}, {t & t}")
# print(f"0111 = {f | f}, {f | t}, {t | f}, {t | t}")
# print(f"1101 = {f >> f}, {f >> t}, {t >> f}, {t >> t}")
# print(f"1011 = {f << f}, {f << t}, {t << f}, {t << t}")
# print(f"1001 = {~(f @ f)}, {~(f @ t)}, {~(t @ f)}, {~(t @ t)}")
# print(f"0110 = {f @ f}, {f @ t}, {t @ f}, {t @ t}")
# print(f"1110 = {~(f & f)}, {~(f & t)}, {~(t & f)}, {~(t & t)}")
# print(f"1000 = {~(f | f)}, {~(f | t)}, {~(t | f)}, {~(t | t)}")
# print(f"0010 = {~(f >> f)}, {~(f >> t)}, {~(t >> f)}, {~(t >> t)}")
# print(f"0100 = {~(f << f)}, {~(f << t)}, {~(t << f)}, {~(t << t)}")


# #  This animates the norms (by strictness) for inspection.
#
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
#
# # Set the figure size
# plt.rcParams["figure.figsize"] = [5.00, 5.00]
# plt.rcParams["figure.autolayout"] = True
#
# # data for the contour plot
# data = np.empty((64, 64, 200))
#
#
# for t in range(0, 100):
#     n = Norm.define(strictness=t*2-100)
#     for x in range(0, 64):
#         for y in range(0, 64):
#             data[x, y, t ] = n.and_(x / 63, y / 63)
#
# # Create a figure and a set of subplots
# fig, ax = plt.subplots()
#
# # Method to change the contour data points
# def animate(i):
#     ax.clear()
#     ax.contourf(data[:, :, i], 64, cmap='plasma')
#
#
# ani = animation.FuncAnimation(fig, animate, 100, interval=10, blit=False)
#
# plt.show()

# #  This plots the norms for examination.
#
# import matplotlib.pyplot as plt
# nlist = ["lx","mm","hh","pp","ee","nn","lb","dd"]
# norm = []*8
# data = []*8
# ax = [plt.Figure()]*8
#
#
# fig, ((ax[0], ax[1], ax[2], ax[3]), (ax[4], ax[5], ax[6], ax[7])) = plt.subplots(nrows=2, ncols=4, sharey="all")
#
# for i in range(0,8):
#     ax[i].set_aspect('equal', adjustable='box')
#
# sa, sb = 10,75
#
# for i in range(0, 8):
#     st = sa + i*(sb-sa)/7
#     # norm.append(Norm.define(norm=nlist[i]))
#     n = Norm.define(strictness=st)
#     data.append(np.empty((32, 32)))
#     plt.axis('square')
#     for x in range(0, 32):
#         a = x / 31
#         for y in range(0, 32):
#             b = y / 31
#             data[i][x, y] = n.or_(a, b)
#     ax[i].contourf(data[i], 16, cmap='plasma')
#
# plt.show()


# c = Truth(.4)
# d = Truth(.5)
# e = Truth(.6)
# (.2).or_(.3)
# print(f"c and d={(c and d)}, c.and(d)={c.and_(d)}, d.and(e)={d.and_(e)}, c.and(e)={c.and_(e)}")
# print(f"c and d={d and c}, c&d={c & d}, d&e={d & e}, c&e={c & e}")
# print(f"")
#
# print(f"c.and(f)={c.and_(f)}, c.and(i)={c.and_(i)}, c.and(b)={c.and_(b)}")
# print(f"f.and(c)=err, i.and(c)=err, b.and(c)=err")
# print(f"c&f={c & f}, d&i={d & i}, c&b={c & b}")
# print(f"f&c={f & c}, i&d={i & d}, b&c={b & c}")
# tr = Truth(1)
# fa = Truth(0)
# TEST THE TRUTH TABLES
# print(f"not_=10: {fa.not_()}, {tr.not_()}")
# print(f"and_=0001: {fa.and_(fa)}, {fa.and_(tr)}, {tr.and_(fa)}, {tr.and_(tr)}")
# print(f"nand=1110: {fa.nand(fa)}, {fa.nand(tr)}, {tr.nand(fa)}, {tr.nand(tr)}")
# print(f"or_=0111: {fa.or_(fa)}, {fa.or_(tr)}, {tr.or_(fa)}, {tr.or_(tr)}")
# print(f"nor=1000: {fa.nor(fa)}, {fa.nor(tr)}, {tr.nor(fa)}, {tr.nor(tr)}")
# print(f"imp=1101: {fa.imp(fa)}, {fa.imp(tr)}, {tr.imp(fa)}, {tr.imp(tr)}")
# print(f"nimp=0010: {fa.nimp(fa)}, {fa.nimp(tr)}, {tr.nimp(fa)}, {tr.nimp(tr)}")
# print(f"con=1011: {fa.con(fa)}, {fa.con(tr)}, {tr.con(fa)}, {tr.con(tr)}")
# print(f"ncon=0100: {fa.ncon(fa)}, {fa.ncon(tr)}, {tr.ncon(fa)}, {tr.ncon(tr)}")
# print(f"iff=1001: {fa.iff(fa)}, {fa.iff(tr)}, {tr.iff(fa)}, {tr.iff(tr)}")
# print(f"xor=0110: {fa.xor(fa)}, {fa.xor(tr)}, {tr.xor(fa)}, {tr.xor(tr)}")

# TEST THE OVERLOADED OPERATORS ~ & | >> <<;  commutativity or not with float, int, bool:
# print(f"c={c}, d={d}, e={e}, f={f}, i={i}, b={b}")
# print(f"~c={~c}, ~d={~d}, ~e={~e}, ~f=ERR, ~i={~i}=UNX, ~b={~b}=UNX")
# print("")
# print(f"c&d={c&d}, c&f={c&f}, c&i={c&i}, c&b={c&b}")
# print(f"d&c={d&c}, f&c={f&c}, i&c={i&c}, b&c={b&c}")
# print(f"c|d={c|d}, c|f={c|f}, c|i={c|i}, c|b={c|b}")
# print(f"d|c={d|c}, f|c={f|c}, i|c={i|c}, b|c={b|c}")
# print(f"c>>d={c>>d}, c>>f={c>>f}, c>>i={c>>i}, c>>b={c>>b}")
# print(f"d>>c={d>>c}, f>>c={f>>c}, i>>c={i>>c}, b>>c={b>>c}")
# print(f"c<<d={c<<d}, c<<f={c<<f}, c<<i={c<<i}, c<<b={c<<b}")
# print(f"d<<c={d<<c}, f<<c={f<<c}, i<<c={i<<c}, b<<c={b<<c}")
# print("")   # truth.py tables:
# print(f"~=10: {~fa}, {~tr}")
# print(f"&=0001: {fa&(fa)}, {fa&(tr)}, {tr&(fa)}, {tr&(tr)}")
# print(f"|_=0111: {fa|(fa)}, {fa|(tr)}, {tr|(fa)}, {tr|(tr)}")
# print(f">>=1101: {fa>>(fa)}, {fa>>(tr)}, {tr>>(fa)}, {tr>>(tr)}")
# print(f"<<=1011: {fa<<(fa)}, {fa<<(tr)}, {tr<<(fa)}, {tr<<(tr)}")

#
# def display_number_logic(a: FuzzyNumber, b: FuzzyNumber) -> None:    # This generates logic_on_numbers.png
#     px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
#     plt.rcParams['figure.figsize'] = [px*800, px*800/1.618]
#     plt.rcParams["figure.autolayout"] = True
#     fig, ax = plt.subplots(nrows=4, ncols=3)
#     fig.patch.set_facecolor('#eee8d5')
#
#     v = linspace(-3, 15, 400)
#     colora="#268bd2"
#     colorb="#b58900"   # "#2aa198"   # "#859900"
#     colorc="#d30102"
#     facecolor="#fdf6e3"
#     lw=2
#
#     sa = a.t(v)
#     sb = b.t(v)
#
#     for axs in ax.flat:
#         axs.label_outer()
#         axs.set_facecolor(facecolor)
#         axs.set_xticks([], [])
#         axs.plot(v, sa, color=colora, linewidth=lw, alpha=.4)
#         axs.plot(v, sb, color=colorb, linewidth=lw, alpha=.4)
#
#
#     sc = Not(a).t(v)
#     ax[0, 0].plot(v, sc, color=colorc, linewidth=lw, alpha=1)
#     ax[0, 0].set_title('a.not(),      ~a')
#     sc = Not(b).t(v)
#     ax[1, 0].plot(v, sc, color=colorc, linewidth=lw, alpha=1)
#     ax[1, 0].set_title('b.not(),      ~b')
#     sc = Imp(a,b).t(v)
#     ax[2, 0].plot(v, sc, color=colorc, linewidth=lw, alpha=1)
#     ax[2, 0].set_title('a.imp(b),      a >> b')
#     sc = Nimp(a,b).t(v)
#     ax[3, 0].plot(v, sc, color=colorc, linewidth=lw, alpha=1)
#     ax[3, 0].set_title('a.nimp(b),      ~(a >> b)')
#
#     sc = And(a,b).t(v)
#     ax[0, 1].plot(v, sc, color=colorc, linewidth=lw, alpha=1)
#     ax[0, 1].set_title('a.and_(b),      a & b')
#     sc = Nand(a,b).t(v)
#     ax[1, 1].plot(v, sc, color=colorc, linewidth=lw, alpha=1)
#     ax[1, 1].set_title('a.nand(b),      ~(a & b)')
#     sc = Con(a,b).t(v)
#     ax[2, 1].plot(v, sc, color=colorc, linewidth=lw, alpha=1)
#     ax[2, 1].set_title('a.con(b),      a << b')
#     sc = Ncon(a,b).t(v)
#     ax[3, 1].plot(v, sc, color=colorc, linewidth=lw, alpha=1)
#     ax[3, 1].set_title('a.ncon(b),      ~(a << b)')
#
#     sc = Or(a, b).t(v)
#     ax[0, 2].plot(v, sc, color=colorc, linewidth=lw, alpha=1)
#     ax[0, 2].set_title('a.or_(b),      a | b')
#     sc = Nor(a, b).t(v)
#     ax[1, 2].plot(v, sc, color=colorc, linewidth=lw, alpha=1)
#     ax[1, 2].set_title('a.nor(b),      ~(a | b)')
#     sc = Iff(a, b).t(v)
#     ax[2, 2].plot(v, sc, color=colorc, linewidth=lw, alpha=1)
#     ax[2, 2].set_title('a.iff(b),      ~(a @ b)')
#     sc = Xor(a, b).t(v)
#     ax[3, 2].plot(v, sc, color=colorc, linewidth=lw, alpha=1)
#     ax[3, 2].set_title('a.xor(b),      a @ b')
#
#     plt.show()

# def display_logic_ops() -> None:    # This generates logic_heatmaps_prod.png
#     px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
#     plt.rcParams['figure.figsize'] = [px*800, px*800/1.24245436]
#     plt.rcParams["figure.autolayout"] = True
#     fig, ax = plt.subplots(nrows=3, ncols=4)
#     fig.patch.set_facecolor('#eee8d5')
#
#     facecolor="#fdf6e3"
#     lev=9
#     # colormap = "cividis"  # "bone""cividis""BrBG""pink""copper"  #  The following sets up solarized linear lightness
#     cdict = {'red': [[0.000, 0.0, 0.0],
#                      [0.036, 7/255, 7/255],
#                      [0.349, 88/255, 88/255],
#                      [0.410, 101/255, 101/255],
#                      [0.530, 131/255, 131/255],
#                      [0.590, 147/255, 147/255],
#                      [0.927, 238/255, 238/255],
#                      [1.0, 253/255, 253/255]],
#              'green': [[0.000, 43/255, 43/255],
#                      [0.036, 54/255, 54/255],
#                      [0.349, 110/255, 110/255],
#                      [0.410, 123/255, 123/255],
#                      [0.530, 148/255, 148/255],
#                      [0.590, 161/255, 161/255],
#                      [0.927, 232/255, 232/255],
#                      [1.0, 246/255, 246/255]],
#              'blue': [[0.000, 54/255, 54/255],
#                      [0.036, 66/255, 66/255],
#                      [0.349, 117/255, 117/255],
#                      [0.410, 131/255, 131/255],
#                      [0.530, 150/255, 150/255],
#                      [0.590, 161/255, 161/255],
#                      [0.927, 213/255, 213/255],
#                      [1.0, 227/255, 227/255]]}
#     colormap = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
#
#     a = b = linspace(0, 1, 400)
#
#     for axs in ax.flat:
#         axs.label_outer()
#         axs.set_facecolor(facecolor)
#         axs.set_xticks([0, .5, 1], [0, .5, 1])
#         axs.set_yticks([0, .5, 1], [0, .5, 1])
#         axs.set_box_aspect(1)
#     x = a.reshape(1, -1)
#     y = b.reshape(-1, 1)
#     znota = Truth.not_(x)
#     znota = np.tile(znota, (400, 1))
#     znotb = Truth.not_(y)
#     znotb = np.tile(znotb, (1, 400))
#     zand = Truth.and_(x, y)
#     zor = Truth.or_(x, y)
#     zimp = Truth.imp(x, y)
#     zcon = Truth.con(x, y)
#     znand = Truth.nand(x, y)
#     znor = Truth.nor(x, y)
#     znimp = Truth.nimp(x, y)
#     zncon = Truth.ncon(x, y)
#     ziff = Truth.iff(x, y)
#     zxor = Truth.xor(x, y)
#
#     x, y = x.flatten(), y.flatten()
#     ax[0, 0].contourf(x, y, znota, levels = lev, cmap = colormap)
#     ax[0, 0].set_title('not    ~a')
#     ax[0, 1].contourf(x, y, znotb, levels = lev, cmap = colormap)
#     ax[0, 1].set_title('not    ~b')
#     ax[0, 2].contourf(x, y, zand, levels = lev, cmap = colormap)
#     ax[0, 2].set_title('and_    a & b')
#     ax[0, 3].contourf(x, y, zor, levels = lev, cmap = colormap)
#     ax[0, 3].set_title('or_    a | b')
#
#     ax[1, 0].contourf(x, y, zimp, levels = lev, cmap = colormap)
#     ax[1, 0].set_title('imp   a >> b')
#     ax[1, 1].contourf(x, y, zcon, levels = lev, cmap = colormap)
#     ax[1, 1].set_title('con   a << b')
#     ax[1, 2].contourf(x, y,  znand, levels = lev, cmap = colormap)
#     ax[1, 2].set_title('nand   ~(a & b)')
#     ax[1, 3].contourf(x, y,  znor, levels = lev, cmap = colormap)
#     ax[1, 3].set_title('nor   ~(a | b)')
#
#     ax[2, 0].contourf(x, y, znimp, levels = lev, cmap = colormap)
#     ax[2, 0].set_title('nimp   ~(a >> b)')
#     ax[2, 1].contourf(x, y, zncon, levels = lev, cmap = colormap)
#     ax[2, 1].set_title('ncon   ~(a << b)')
#     ax[2, 2].contourf(x, y,  ziff, levels = lev, cmap = colormap)
#     ax[2, 2].set_title('iff   ~(a @ b)')
#     ax[2, 3].contourf(x, y,  zxor, levels = lev, cmap = colormap)
#     ax[2, 3].set_title('xor    a @ b')
#
#     plt.show()
#
#
#
# def display_norms() -> None:    # This generates t-norm_gallery.png
#     px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
#     plt.rcParams['figure.figsize'] = [px*800, px*800/1.618]
#     plt.rcParams["figure.autolayout"] = True
#     fig, ax = plt.subplots(nrows=2, ncols=4)
#     fig.patch.set_facecolor('#eee8d5')
#
#     facecolor="#fdf6e3"
#     lev=9
#     # colormap = "cividis"  # "bone""cividis""BrBG""pink""copper"  #  The following sets up solarized linear lightness
#     cdict = {'red': [[0.000, 0.0, 0.0],
#                      [0.036, 7/255, 7/255],
#                      [0.349, 88/255, 88/255],
#                      [0.410, 101/255, 101/255],
#                      [0.530, 131/255, 131/255],
#                      [0.590, 147/255, 147/255],
#                      [0.927, 238/255, 238/255],
#                      [1.0, 253/255, 253/255]],
#              'green': [[0.000, 43/255, 43/255],
#                      [0.036, 54/255, 54/255],
#                      [0.349, 110/255, 110/255],
#                      [0.410, 123/255, 123/255],
#                      [0.530, 148/255, 148/255],
#                      [0.590, 161/255, 161/255],
#                      [0.927, 232/255, 232/255],
#                      [1.0, 246/255, 246/255]],
#              'blue': [[0.000, 54/255, 54/255],
#                      [0.036, 66/255, 66/255],
#                      [0.349, 117/255, 117/255],
#                      [0.410, 131/255, 131/255],
#                      [0.530, 150/255, 150/255],
#                      [0.590, 161/255, 161/255],
#                      [0.927, 213/255, 213/255],
#                      [1.0, 227/255, 227/255]]}
#     colormap = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
#
#     a = b = linspace(0, 1, 400)
#
#     for axs in ax.flat:
#         axs.label_outer()
#         axs.set_facecolor(facecolor)
#         axs.set_xticks([0, .5, 1], [0, .5, 1])
#         axs.set_yticks([0, .5, 1], [0, .5, 1])
#         axs.set_box_aspect(1)
#     x = a.reshape(1, -1)
#     y = b.reshape(-1, 1)
#     # The following is for the t-norm gallery:
#     zlax = Truth.iff(x, y, norm={'n1':"lx"})
#     zmin = Truth.iff(x, y, norm={'n1':"mm"})
#     zham = Truth.iff(x, y, norm={'n1':"hh"})
#     zgog = Truth.iff(x, y, norm={'n1':"pp"})
#     zein = Truth.iff(x, y, norm={'n1':"ee"})
#     znil = Truth.iff(x, y, norm={'n1':"nn"})
#     zluk = Truth.iff(x, y, norm={'n1':"lb"})
#     zdra = Truth.iff(x, y, norm={'n1':"dd"})
#     # The following is for the strictness gallery:
#     # zlax = Truth.iff(x, y, norm={'n1':"str", 'n1p':[-81]})
#     # zmin = Truth.iff(x, y, norm={'n1':"str", 'n1p':[-59]})
#     # zham = Truth.iff(x, y, norm={'n1':"str", 'n1p':[-37]})
#     # zgog = Truth.iff(x, y, norm={'n1':"str", 'n1p':[-15]})
#     # zein = Truth.iff(x, y, norm={'n1':"str", 'n1p':[7]})
#     # znil = Truth.iff(x, y, norm={'n1':"str", 'n1p':[29]})
#     # zluk = Truth.iff(x, y, norm={'n1':"str", 'n1p':[51]})
#     # zdra = Truth.iff(x, y, norm={'n1':"str", 'n1p':[73]})
#
#     x, y = x.flatten(), y.flatten()
#     ax[0, 0].contourf(x, y, zlax, levels = lev, cmap = colormap)
#     ax[0, 0].set_title('lax')       #('-81')       #
#     ax[0, 1].contourf(x, y, zmin, levels = lev, cmap = colormap)
#     ax[0, 1].set_title('min-max (Gödel-Zadeh)')       #('-59')       #
#     ax[0, 2].contourf(x, y, zham, levels = lev, cmap = colormap)
#     ax[0, 2].set_title('Hamacher')       #('-37')       #
#     ax[0, 3].contourf(x, y, zgog, levels = lev, cmap = colormap)
#     ax[0, 3].set_title('product (Goguen)')       #('-15')       #
#
#     ax[1, 0].contourf(x, y, zein, levels = lev, cmap = colormap)
#     ax[1, 0].set_title('Einstein')       #('7')       #
#     ax[1, 1].contourf(x, y, znil, levels = lev, cmap = colormap)
#     ax[1, 1].set_title('nilpotent (Kleene-Dienes)')       #('29')       #
#     ax[1, 2].contourf(x, y,  zluk, levels = lev, cmap = colormap)
#     ax[1, 2].set_title('Łukasiewicz')       #('51')       #
#     ax[1, 3].contourf(x, y,  zdra, levels = lev, cmap = colormap)
#     ax[1, 3].set_title('drastic')       #('73')       #
#
#     plt.show()
#
#
#
# display_norms()

# This block makes the logic_weight figure.
# import matplotlib.pyplot as plt
# import matplotlib.collections as mcol
# from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
# from matplotlib.lines import Line2D
# px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
# plt.rcParams['figure.figsize'] = [px*800, px*800/1.618]
# plt.rcParams["figure.autolayout"] = True
# fig, ax = plt.subplots()
# fig.patch.set_facecolor("#eee8d5")
# ax.set_facecolor("#fdf6e3")
# xpoints = np.linspace(0, 1, num=1000)
# yn100 = Truth.weight(xpoints, -90)
# yn66 = Truth.weight(xpoints, -60)
# yn33 = Truth.weight(xpoints, -30)
# yz0 = Truth.weight(xpoints, 0)
# yp33 = Truth.weight(xpoints, 30)
# yp66 = Truth.weight(xpoints, 60)
# yp100 = Truth.weight(xpoints, 90)
# lw = 2
# l1 = ax.plot(xpoints, yn100, color='#6c71c4', linewidth=lw)
# l2 = ax.plot(xpoints, yn66, color='#268bd2', linewidth=lw)
# l3 = ax.plot(xpoints, yn33, color='#2aa198', linewidth=lw)
# l4 = ax.plot(xpoints, yz0, color='#859900', linewidth=lw)
# l5 = ax.plot(xpoints, yp33, color='#b58900', linewidth=lw)
# l6 = ax.plot(xpoints, yp66, color='#cb4b16', linewidth=lw)
# l7 = ax.plot(xpoints, yp100, color='#dc322f', linewidth=lw)
# ax.legend(('w=-90','w=-60','w=-30','w=0','w=30','w=60','w=90'), loc='lower right')
# xdots = np.array([0, .5, 1])
# ydots = np.array([0, .5, 1])
# plt.plot(xdots, ydots, 'o', color='#002b3688')
# plt.show()