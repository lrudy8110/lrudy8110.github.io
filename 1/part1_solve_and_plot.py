"""
SYDE572 Assignment 1 - Part 1
Analytical + numerical (Newton-Raphson, Golden Section) solutions for
distance from a point to y = x^2 + 5, for 5 given points.
Also demos the same numerical code on non-polynomial functions.
Generates all Part 1 plots into media/.

Run: python3 part1_solve_and_plot.py
Requires: sympy, numpy, matplotlib
"""
import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from point_to_curve import find_distance_newton, golden_section_search

plt.rcParams.update({'font.size': 11, 'figure.dpi': 130})

# ---------------------------------------------------------------
# 1. Analytical solution (sympy) for all 5 points
# ---------------------------------------------------------------
x = sp.symbols('x', real=True)
f_expr = x**2 + 5

points = [(0, 0), (-4, 0), (-8, 0), (2, 0), (6, 0)]

print("=== PART 1: ANALYTICAL SOLUTION ===")
print(f"{'Point':<10}{'x* (foot)':<15}{'y*':<12}{'Distance':<12}")
analytical_results = []
for (x0, y0) in points:
    D = (x - x0)**2 + (f_expr - y0)**2
    Dprime = sp.diff(D, x)
    crit_points = sp.solve(sp.Eq(Dprime, 0), x)
    real_crit = [c for c in crit_points if c.is_real]
    best = min(real_crit, key=lambda c: D.subs(x, c))
    best_f = float(best)
    y_star = best_f**2 + 5
    dist = ((best_f - x0)**2 + (y_star - y0)**2)**0.5
    analytical_results.append((x0, y0, best_f, y_star, dist))
    print(f"({x0},{y0})".ljust(10) + f"{best_f:<15.6f}{y_star:<12.6f}{dist:<12.6f}")

# Show the full symbolic derivation for the (0,0) case, since it's the one
# with a clean closed form (used in the write-up "by hand" section)
D0 = (x - 0)**2 + (f_expr - 0)**2
print("\nWorked example for (0,0):")
print("D(x) =", sp.expand(D0))
print("D'(x) =", sp.expand(sp.diff(D0, x)))

# ---------------------------------------------------------------
# 2. Numerical solutions: Newton-Raphson and Golden Section Search
#    (using the functions imported from point_to_curve.py)
# ---------------------------------------------------------------
f = lambda x: x**2 + 5
df = lambda x: 2*x
ddf = lambda x: 2

print("\n=== PART 1: NUMERICAL SOLUTIONS ===")
print(f"{'Point':<10}{'Newton d':<14}{'Newton x*':<14}{'Golden d':<14}{'Golden x*':<14}")
for (x0, y0) in points:
    dN, xN = find_distance_newton(x0, y0, f, df, ddf, initial_guess=x0)
    dG, xG = golden_section_search(x0, y0, f, a=x0-10, b=x0+10)
    print(f"({x0},{y0})".ljust(10) + f"{dN:<14.6f}{xN:<14.6f}{dG:<14.6f}{xG:<14.6f}")

# ---------------------------------------------------------------
# 3. Plot 1: overview - parabola + all 5 points + distance segments
# ---------------------------------------------------------------
xs = np.linspace(-10, 10, 400)
colors = ['#e63946', '#f4a261', '#2a9d8f', '#457b9d', '#8338ec']

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(xs, f(xs), color='#1d3557', linewidth=2, label='y = x^2 + 5')
for (x0, y0), c in zip(points, colors):
    d, xstar = find_distance_newton(x0, y0, f, df, ddf, initial_guess=x0)
    ystar = f(xstar)
    ax.plot(x0, y0, 'o', color=c, markersize=8, zorder=5)
    ax.plot(xstar, ystar, 's', color=c, markersize=6, zorder=5)
    ax.plot([x0, xstar], [y0, ystar], '--', color=c, linewidth=1.5)
    ax.annotate(f'd={d:.2f}', xy=((x0+xstar)/2, (y0+ystar)/2), fontsize=9, color=c,
                xytext=(5, 5), textcoords='offset points')
    ax.annotate(f'({x0},{y0})', xy=(x0, y0), fontsize=8, xytext=(6, -10), textcoords='offset points')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('Shortest distance from each point to y = x^2 + 5')
ax.legend(loc='upper center')
ax.grid(alpha=0.3)
ax.set_ylim(-2, 30)
plt.tight_layout()
plt.savefig('media/part1_overview.png')
plt.close()

# ---------------------------------------------------------------
# 4. Plot 2: Newton-Raphson convergence for the hardest point (-8,0)
# ---------------------------------------------------------------
d, xstar, hist = find_distance_newton(-8, 0, f, df, ddf, initial_guess=-8, record=True)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(range(len(hist)), hist, 'o-', color='#2a9d8f')
ax.axhline(xstar, color='gray', linestyle=':', label=f'converged x*={xstar:.4f}')
ax.set_xlabel('Iteration'); ax.set_ylabel('x value')
ax.set_title('Newton-Raphson convergence for point (-8, 0)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('media/part1_newton_convergence.png')
plt.close()

# ---------------------------------------------------------------
# 5. Plot 3: Golden Section Search bracket narrowing for (-8,0)
# ---------------------------------------------------------------
d2, xstar2, hist2 = golden_section_search(-8, 0, f, a=-18, b=2, record=True)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for i, (a, b) in enumerate(hist2):
    ax.plot([a, b], [i, i], color='#e76f51', linewidth=2)
ax.axvline(xstar2, color='gray', linestyle=':', label=f'converged x*={xstar2:.4f}')
ax.set_xlabel('x (bracket [a,b])'); ax.set_ylabel('Iteration')
ax.set_title('Golden Section Search bracket narrowing for point (-8, 0)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('media/part1_golden_bracket.png')
plt.close()

print(f"\nNewton iterations for (-8,0): {len(hist)-1}")
print(f"Golden section iterations for (-8,0): {len(hist2)-1}")

# ---------------------------------------------------------------
# 6. Plot 4: demo on non-polynomial functions (exp, log, sqrt)
# ---------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))

f_exp = lambda x: np.exp(x); df_exp = lambda x: np.exp(x); ddf_exp = lambda x: np.exp(x)
x0, y0 = 2, 0
d, xstar = find_distance_newton(x0, y0, f_exp, df_exp, ddf_exp, initial_guess=0.0)
xs = np.linspace(-2, 3, 300)
axs[0].plot(xs, f_exp(xs), color='#1d3557')
axs[0].plot(x0, y0, 'o', color='#e63946'); axs[0].plot(xstar, f_exp(xstar), 's', color='#e63946')
axs[0].plot([x0, xstar], [y0, f_exp(xstar)], '--', color='#e63946')
axs[0].set_title(f'f(x)=e^x, point (2,0)\nd={d:.4f}')
axs[0].set_ylim(-1, 10); axs[0].grid(alpha=0.3)

f_log = lambda x: np.log(x); df_log = lambda x: 1/x; ddf_log = lambda x: -1/x**2
x0, y0 = 3, 0
d, xstar = find_distance_newton(x0, y0, f_log, df_log, ddf_log, initial_guess=1.0)
xs = np.linspace(0.1, 6, 300)
axs[1].plot(xs, f_log(xs), color='#1d3557')
axs[1].plot(x0, y0, 'o', color='#2a9d8f'); axs[1].plot(xstar, f_log(xstar), 's', color='#2a9d8f')
axs[1].plot([x0, xstar], [y0, f_log(xstar)], '--', color='#2a9d8f')
axs[1].set_title(f'f(x)=ln(x), point (3,0)\nd={d:.4f}')
axs[1].grid(alpha=0.3)

f_sqrt = lambda x: np.sqrt(x); df_sqrt = lambda x: 0.5/np.sqrt(x); ddf_sqrt = lambda x: -0.25*x**-1.5
x0, y0 = 4, 0
d, xstar = find_distance_newton(x0, y0, f_sqrt, df_sqrt, ddf_sqrt, initial_guess=1.0)
xs = np.linspace(0.01, 8, 300)
axs[2].plot(xs, f_sqrt(xs), color='#1d3557')
axs[2].plot(x0, y0, 'o', color='#457b9d'); axs[2].plot(xstar, f_sqrt(xstar), 's', color='#457b9d')
axs[2].plot([x0, xstar], [y0, f_sqrt(xstar)], '--', color='#457b9d')
axs[2].set_title(f'f(x)=sqrt(x), point (4,0)\nd={d:.4f}')
axs[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('media/part1_nonpoly.png')
plt.close()

print("\nAll Part 1 plots saved to media/")