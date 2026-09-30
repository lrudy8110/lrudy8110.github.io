"""
SYDE572 A1 Part 1: extra demos.
(a) f(x) = 1/x (variable in the denominator), point (0,0), right branch x > 0
(b) a second parabola y = 0.5x^2 - 2x + 1 with two different points
Run from inside the 1/ folder: python3 part1_more.py
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial as P
from point_to_curve import find_distance_newton, golden_section_search

plt.rcParams.update({'font.size': 11, 'figure.dpi': 130})
fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))

# ---------- (a) 1/x ----------
f = lambda x: 1/x
df = lambda x: -1/x**2
ddf = lambda x: 2/x**3
x0, y0 = 0, 0
dN, xN = find_distance_newton(x0, y0, f, df, ddf, initial_guess=1.5)
dG, xG = golden_section_search(x0, y0, f, a=0.1, b=5)
print("1/x, point (0,0), domain x > 0")
print(f"  Newton (d, x*):  ({dN:.6f}, {xN:.6f})")
print(f"  Golden (d, x*):  ({dG:.6f}, {xG:.6f})")
print(f"  exact: x* = 1, d = sqrt(2) = {2**0.5:.6f}")

ax = axs[0]
xx = np.linspace(0.15, 5, 400)
ax.plot(xx, f(xx), color='#1d3557', lw=2, label=r'$f(x)=1/x$')
ax.axvspan(-1, 0, color='gray', alpha=0.2, label='outside domain (x < 0)')
ax.plot(x0, y0, 'o', color='black', ms=8, zorder=6, label='point (0, 0)')
ax.plot([x0, xN], [y0, f(xN)], '--', color='#e63946', lw=1.5)
ax.plot(xN, f(xN), 's', color='#e63946', ms=9, zorder=5, label=f'Newton: x*={xN:.4f}')
ax.plot(xG, f(xG), 'D', mfc='none', mec='black', ms=12, zorder=7, label=f'Golden Section: x*={xG:.4f}')
ax.set_xlim(-1, 5); ax.set_ylim(-1, 5)
ax.set_title(rf'$f(x)=1/x$, point (0, 0), d = {dN:.4f}')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.grid(alpha=0.3); ax.legend(loc='upper right', fontsize=9)

# ---------- (b) another parabola, analytical vs Newton vs Golden ----------
fp = P([1, -2, 0.5])          # 0.5x^2 - 2x + 1
f2 = lambda x: 0.5*x**2 - 2*x + 1
df2 = lambda x: x - 2
ddf2 = lambda x: 1.0
cases = [(4, 5, '#2a9d8f'), (-2, 6, '#8338ec')]
ax = axs[1]
xx = np.linspace(-5, 7, 400)
ax.plot(xx, f2(xx), color='#1d3557', lw=2, label=r'$y=0.5x^2-2x+1$')
print("\ny = 0.5x^2 - 2x + 1")
print("  point: analytical d | Newton d (x*) | Golden d (x*)")
for (px, py, c) in cases:
    dprime = P([-px, 1]) + (fp - py) * fp.deriv()       # D'(x)/2
    real = [r.real for r in dprime.roots() if abs(r.imag) < 1e-9]
    xa = min(real, key=lambda r: (r - px)**2 + (f2(r) - py)**2)
    da = ((xa - px)**2 + (f2(xa) - py)**2) ** 0.5
    dN, xN = find_distance_newton(px, py, f2, df2, ddf2, initial_guess=px)
    dG, xG = golden_section_search(px, py, f2, a=px-10, b=px+10)
    print(f"  ({px},{py}): {da:.6f} | {dN:.6f} ({xN:.6f}) | {dG:.6f} ({xG:.6f})")
    ax.plot(px, py, 'o', color=c, ms=8, zorder=6, label=f'({px}, {py}):  d = {dN:.4f}')
    ax.plot([px, xN], [py, f2(xN)], '--', color=c, lw=1.5)
    ax.plot(xN, f2(xN), 's', color=c, ms=8, zorder=5)
ax.set_xlim(-5, 7); ax.set_ylim(-2, 12)
ax.set_title('Another parabola, two different points')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.grid(alpha=0.3); ax.legend(loc='upper center', fontsize=9)

plt.tight_layout()
plt.savefig('media/part1_more.png'); plt.close()
print("\nsaved media/part1_more.png")
