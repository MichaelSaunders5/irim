# Here is where I am testing or playing around or something.
import numpy as np
import matplotlib.pyplot as plt
from fuzzy.value import *

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

# TODO: test dp, cp, ;cont + xp?, trap cauchy gauss bell exactly. width/focus, logic, arith, outputs, crisp etc.
# t = Triangle(0,5,10)
# t = DPoints(((0,1),(2.5,.5),(5, 0),(7.5,.5),(10,0)))
# Truth.default_threshold = .1
# t = CPoints(((0,1),(2.5,.5),(5, 1),(7.5,.75),(10,0)), None)
# t = Trapezoid(1, 3, 5, 9, (.4,.6))
# t = Cauchy(5,2, (0,10))
# t = Gauss(6,.1, 18)
t = Bell(1,8,5)

# "linear" or "bary" or "krogh" or "pchip" or "akima":‘not-a-knot’, ‘periodic’, ‘clamped’, ‘natural’
print(f"0: {t.suitability(0)}, 2.5: {t.suitability(2.5)}, 5: {t.suitability(5)}, 10: {t.suitability(10)}")
fn = t.evaluate(.01)
# # Here's a way to view fuzzy numbers (1D functions):
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.xlim(-1,11)     # domain to plot
plt.ylim(-.1, 1.1)
plt.grid()

v = fn.v       # [:, 0]
s = fn.s       # [:, 1]
plt.plot(v, s, color="red")

if fn.xp is not None:
    xpv = fn.xp[:, 0]
    xps = fn.xp[:, 1]
    plt.plot(xpv, xps, "o", markersize=5, color="blue")
print(f"xp: {fn.xp}")


plt.title("Line graph")


plt.show()


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

# x = Truth(.2)
# print(f"x={x}, 100*x={100*x}, x.not_()={x.not_()}, not x={not x}, ~x={~x}")
# y = Truth(.7)
# print(f"y={y}, 100*y={100*y}, y.not_()={y.not_()}, not y={not y}, ~y={~y}")
f = .499
# print(f"f={f}, 100*f={100*f}") # , f.not_()={f.not_()}, ~f={~f}
i = 0
# print(f"i={i}, 100*i={100*i}, ~i={~i}, not i={not i}")
b = True
# print(f"b={b}, 100*b={100*b}, ~b={~b}, not b={not b}")

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

# Truth.default_threshold = .8
# Truth.setGlobalThreshold(.8)

# print(f"true: {Truth(0).isValid()}{Truth(.5).isValid()}{Truth(1).isValid()}")
# print(f"false{Truth(-1).isValid()}{Truth(1.1).isValid()}{Truth(2).isValid()}")
# print("w=0")
# print(
#     f"0, {Truth(0).weight(0)};  .25, {Truth(.25).weight(0)};  .5, {Truth(.5).weight(0)};  1, {Truth(1).weight(0)}.")
# print("w=100")
# print(f".01, {Truth(.01).weight(100)};  .49, {Truth(.49).weight(100)};  ")
# print(f".51, {Truth(.51).weight(100)};  .99, {Truth(.99).weight(100)}.")
# print("w=-100")
# print(f"0, {Truth(0).weight(-100)};  .01, {Truth(.01).weight(-100)};  .5, {Truth(.5).weight(-100)};  ")
# print(f".99, {Truth(.99).weight(-100)};  1, {Truth(1).weight(-100)}.")
#
# print("w=50")
# print(f".25, {Truth(.25).weight(50)};  .75, {Truth(.75).weight(50)};  ")
#
# tarray = np.linspace(0, 1, 11)
#
# print(Truth.weight(tarray, 50))
