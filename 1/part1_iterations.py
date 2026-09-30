"""
SYDE572 A1 Part 1: iteration plots for Newton-Raphson and Golden Section.
Run from inside the 1/ folder: python3 part1_iterations.py
"""
import numpy as np
import matplotlib.pyplot as plt
from point_to_curve import find_distance_newton, golden_section_search

plt.rcParams.update({'font.size': 11, 'figure.dpi': 130})

f = lambda x: x**2 + 5
df = lambda x: 2*x
ddf = lambda x: 2
points = [(0, 0), (-4, 0), (-8, 0), (2, 0), (6, 0)]
colors = ['#e63946', '#f4a261', '#2a9d8f', '#457b9d', '#8338ec']
r = 0.3819660112501051  # 2 - golden ratio

# ---- Newton: iterates drawn on the parabola for (-8,0) ----
x0, y0 = -8, 0
d, xs_, hist = find_distance_newton(x0, y0, f, df, ddf, initial_guess=x0, record=True)
fig, ax = plt.subplots(figsize=(7.5, 5.5))
xx = np.linspace(-10, 3, 400)
ax.plot(xx, f(xx), color='#1d3557', lw=2, label=r'$y=x^2+5$')
ax.plot(x0, y0, 'o', color='black', ms=8, label=f'point ({x0}, {y0})', zorder=5)
cm = plt.cm.viridis(np.linspace(0, 0.9, len(hist)))
for i, xi in enumerate(hist):
    ax.plot(xi, f(xi), 's', color=cm[i], ms=7, zorder=4)
    ax.annotate(str(i), (xi, f(xi)), xytext=(-9, 7), textcoords='offset points', fontsize=9)
ax.plot([x0, hist[-1]], [y0, f(hist[-1])], '--', color='gray',
        label=f'final distance d = {d:.4f}')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title(f'Newton-Raphson iterates for ({x0}, {y0}): {len(hist)-1} iterations')
ax.set_ylim(-2, 75); ax.grid(alpha=0.3); ax.legend(loc='upper center')
plt.tight_layout(); plt.savefig('media/part1_newton_steps.png'); plt.close()

# ---- Newton: convergence, all five points ----
fig, ax = plt.subplots(figsize=(7, 4.5))
print("Newton iterations (start at x0, tol 1e-7):")
for (px, py), c in zip(points, colors):
    dN, xN, h = find_distance_newton(px, py, f, df, ddf, initial_guess=px, record=True)
    err = np.abs(np.array(h) - xN)
    err[err == 0] = 1e-17
    ax.semilogy(range(len(h)), err, 'o-', color=c, label=f'({px},{py})')
    print(f"  ({px},{py}): {len(h)-1} iterations, x* = {xN:.7f}, d = {dN:.6f}")
ax.set_xlabel('Iteration'); ax.set_ylabel(r'$|x_k - x^*|$')
ax.set_title('Newton-Raphson convergence, all five points')
ax.grid(alpha=0.3, which='both'); ax.legend()
plt.tight_layout(); plt.savefig('media/part1_newton_convergence.png'); plt.close()

# ---- Golden Section: first 5 bracket states for (-8,0) ----
a0, b0 = x0 - 10, x0 + 10
dG, xG, gh = golden_section_search(x0, y0, f, a=a0, b=b0, record=True)
D = lambda x: (x - x0)**2 + (f(x) - y0)**2
xx = np.linspace(a0, b0, 500)
fig, axs = plt.subplots(2, 3, figsize=(12, 7))
axs = axs.ravel()
for i in range(5):
    a, b = gh[i]
    x1 = a + r*(b - a); x2 = b - r*(b - a)
    ax = axs[i]
    ax.semilogy(xx, D(xx), color='#1d3557', lw=1.5)
    ax.axvspan(a, b, color='#e76f51', alpha=0.2)
    ax.plot([x1, x2], [D(x1), D(x2)], 'o', color='#e63946', ms=7)
    ax.axvline(xG, color='gray', ls=':')
    ax.set_title(f'Step {i}: [{a:.3f}, {b:.3f}]', fontsize=10)
    ax.set_xlabel('x'); ax.grid(alpha=0.3)
    if i % 3 == 0: ax.set_ylabel('D(x)')
widths = [b - a for a, b in gh]
axs[5].semilogy(range(len(widths)), widths, 'o-', color='#e76f51')
axs[5].set_title('Bracket width vs step', fontsize=10)
axs[5].set_xlabel('Step'); axs[5].grid(alpha=0.3, which='both')
fig.suptitle(f'Golden Section Search for ({x0}, {y0}), bracket [{a0}, {b0}]; shaded = current bracket, dots = x1, x2',
             fontsize=11)
plt.tight_layout(); plt.savefig('media/part1_golden_steps.png'); plt.close()
print(f"Golden Section for ({x0},{y0}): {len(gh)-1} iterations, x* = {xG:.7f}, d = {dG:.6f}")
