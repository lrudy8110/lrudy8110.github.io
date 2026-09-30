"""
SYDE572 A1 Part 1: cleaned-up overview plot and non-polynomial demo
(Newton-Raphson and Golden Section compared on e^x, ln(x), sqrt(x)).
Run from inside the 1/ folder: python3 part1_extras.py
"""
import numpy as np
import matplotlib.pyplot as plt
from point_to_curve import find_distance_newton, golden_section_search

plt.rcParams.update({'font.size': 11, 'figure.dpi': 130})

# ---------- overview: parabola + 5 points ----------
f = lambda x: x**2 + 5
df = lambda x: 2*x
ddf = lambda x: 2
points = [(0, 0), (-4, 0), (-8, 0), (2, 0), (6, 0)]
colors = ['#e63946', '#f4a261', '#2a9d8f', '#457b9d', '#8338ec']

fig, ax = plt.subplots(figsize=(8.5, 6))
xs = np.linspace(-10, 8, 400)
ax.plot(xs, f(xs), color='#1d3557', lw=2, label=r'$y=x^2+5$')
for (x0, y0), c in zip(points, colors):
    d, xstar = find_distance_newton(x0, y0, f, df, ddf, initial_guess=x0)
    ax.plot([x0, xstar], [y0, f(xstar)], '--', color=c, lw=1.5)
    ax.plot(xstar, f(xstar), 's', color=c, ms=6, zorder=5)
    ax.plot(x0, y0, 'o', color=c, ms=9, zorder=5, label=f'({x0}, {y0}):  d = {d:.4f}')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('Shortest distance from each point to the parabola')
ax.set_xlim(-10, 8); ax.set_ylim(-3, 45); ax.grid(alpha=0.3)
ax.legend(loc='upper center', fontsize=9, title='circle = given point, square = closest point')
plt.tight_layout(); plt.savefig('media/part1_overview.png'); plt.close()
print("saved media/part1_overview.png")

# ---------- non-polynomial: Newton vs Golden Section ----------
cases = [
    dict(name=r'$f(x)=e^x$', f=np.exp, df=np.exp, ddf=np.exp, pt=(2, 0),
         guess=0.0, a=-5, b=5, xlim=(-3, 4), ylim=(-1, 12), dom=None, color='#e63946'),
    dict(name=r'$f(x)=\ln x$', f=np.log, df=lambda x: 1/x, ddf=lambda x: -1/x**2, pt=(3, 0),
         guess=1.0, a=0.001, b=10, xlim=(-1, 7), ylim=(-4, 3), dom=0, color='#2a9d8f'),
    dict(name=r'$f(x)=\sqrt{x}$', f=np.sqrt, df=lambda x: 0.5/np.sqrt(x),
         ddf=lambda x: -0.25*x**-1.5, pt=(4, 0),
         guess=1.0, a=0.001, b=10, xlim=(-1, 8), ylim=(-1.5, 4), dom=0, color='#457b9d'),
]
fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
print("\nfunction, point, Newton (d, x*), Golden Section (d, x*)")
for ax, c in zip(axs, cases):
    x0, y0 = c['pt']
    dN, xN = find_distance_newton(x0, y0, c['f'], c['df'], c['ddf'], initial_guess=c['guess'])
    dG, xG = golden_section_search(x0, y0, c['f'], a=c['a'], b=c['b'])
    lo = c['dom'] + 0.01 if c['dom'] is not None else c['xlim'][0]
    xx = np.linspace(lo, c['xlim'][1], 500)
    ax.plot(xx, c['f'](xx), color='#1d3557', lw=2, label=c['name'])
    if c['dom'] is not None:
        ax.axvspan(c['xlim'][0], c['dom'], color='gray', alpha=0.2, label=f'outside domain (x < {c["dom"]})')
    ax.plot(x0, y0, 'o', color='black', ms=8, zorder=6, label=f'point ({x0}, {y0})')
    ax.plot([x0, xN], [y0, c['f'](xN)], '--', color=c['color'], lw=1.5)
    ax.plot(xN, c['f'](xN), 's', color=c['color'], ms=9, zorder=5, label=f'Newton: x*={xN:.4f}')
    ax.plot(xG, c['f'](xG), 'D', mfc='none', mec='black', ms=12, zorder=7,
            label=f'Golden Section: x*={xG:.4f}')
    ax.set_title(f'{c["name"]}, point ({x0}, {y0}), d = {dN:.4f}')
    ax.set_xlim(*c['xlim']); ax.set_ylim(*c['ylim']); ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.grid(alpha=0.3); ax.legend(loc='best', fontsize=8.5)
    print(f"  {c['name']}, ({x0},{y0}), Newton ({dN:.6f}, {xN:.6f}), Golden ({dG:.6f}, {xG:.6f})")
plt.tight_layout(); plt.savefig('media/part1_nonpoly.png'); plt.close()
print("saved media/part1_nonpoly.png")
