"""
SYDE572 Assignment 1 - Part 2
Fits a line (y=mx+b) and a parabola (y=ax^2+bx+c) to 4 points by
minimizing MSE. Solves analytically (normal equations, via sympy) and
numerically (coordinate-wise multivariate Newton-Raphson).
Generates both Part 2 plots into media/.

Run: python3 part2_fit_and_plot.py
Requires: sympy, numpy, matplotlib
"""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 11, 'figure.dpi': 130})

pts = np.array([(0, 0.5), (2, 3.5), (1, 1.5), (3, 7.5)], dtype=float)
X = pts[:, 0]; Y = pts[:, 1]
n = len(X)

# ---------------------------------------------------------------
# 1. Analytical solution via normal equations (sympy)
# ---------------------------------------------------------------
print("=== LINE FIT: y = m*x + b (analytical) ===")
m, b = sp.symbols('m b', real=True)
MSE_line = sp.expand(sum((Y[i] - (m*X[i]+b))**2 for i in range(n)) / n)
sol_line = sp.solve([sp.diff(MSE_line, m), sp.diff(MSE_line, b)], [m, b])
m_star, b_star = float(sol_line[m]), float(sol_line[b])
mse_line_val = float(MSE_line.subs({m: m_star, b: b_star}))
print(f"m* = {m_star:.6f}, b* = {b_star:.6f}, MSE = {mse_line_val:.6f}")

print("\n=== PARABOLA FIT: y = a*x^2 + b*x + c (analytical) ===")
a, bp, c = sp.symbols('a b c', real=True)
MSE_par = sp.expand(sum((Y[i] - (a*X[i]**2 + bp*X[i] + c))**2 for i in range(n)) / n)
sol_par = sp.solve([sp.diff(MSE_par, a), sp.diff(MSE_par, bp), sp.diff(MSE_par, c)], [a, bp, c])
a_star, b2_star, c_star = float(sol_par[a]), float(sol_par[bp]), float(sol_par[c])
mse_par_val = float(MSE_par.subs({a: a_star, bp: b2_star, c: c_star}))
print(f"a* = {a_star:.6f}, b* = {b2_star:.6f}, c* = {c_star:.6f}, MSE = {mse_par_val:.6f}")

# ---------------------------------------------------------------
# 2. Numerical solution: coordinate-wise (alternating) Newton-Raphson
# ---------------------------------------------------------------
def mse_line_fn(m, b):
    return np.mean((Y - (m*X+b))**2)

def mse_par_fn(a, b, c):
    return np.mean((Y - (a*X**2 + b*X + c))**2)

print("\n=== LINE FIT: alternating Newton-Raphson ===")
m, b = 0.0, 0.0
hist_line = [(m, b, mse_line_fn(m, b))]
for it in range(16):
    g = (2/n) * np.sum((m*X + b - Y) * X)
    gg = (2/n) * np.sum(X**2)
    m -= g / gg
    hist_line.append((m, b, mse_line_fn(m, b)))
    g = (2/n) * np.sum(m*X + b - Y)
    gg = 2.0
    b -= g / gg
    hist_line.append((m, b, mse_line_fn(m, b)))
print(f"Converged: m={m:.6f}, b={b:.6f}, MSE={mse_line_fn(m,b):.6f}")

print("\n=== PARABOLA FIT: alternating Newton-Raphson (cycling a, b, c) ===")
a, bp, c = 0.0, 0.0, 0.0
hist_par = [(a, bp, c, mse_par_fn(a, bp, c))]
for it in range(25):
    g = (2/n) * np.sum((a*X**2 + bp*X + c - Y) * X**2)
    gg = (2/n) * np.sum(X**4)
    a -= g / gg
    hist_par.append((a, bp, c, mse_par_fn(a, bp, c)))
    g = (2/n) * np.sum((a*X**2 + bp*X + c - Y) * X)
    gg = (2/n) * np.sum(X**2)
    bp -= g / gg
    hist_par.append((a, bp, c, mse_par_fn(a, bp, c)))
    g = (2/n) * np.sum(a*X**2 + bp*X + c - Y)
    gg = 2.0
    c -= g / gg
    hist_par.append((a, bp, c, mse_par_fn(a, bp, c)))
print(f"Converged: a={a:.6f}, b={bp:.6f}, c={c:.6f}, MSE={mse_par_fn(a,bp,c):.6f}")

hist_line = np.array(hist_line)
hist_par = np.array(hist_par)

# ---------------------------------------------------------------
# 3. Plot: fit progression (line and parabola at a few snapshot iterations)
# ---------------------------------------------------------------
xs = np.linspace(-0.5, 3.5, 200)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

snap_idx = [0, 2, 6, len(hist_line)-1]
cmap = plt.cm.viridis(np.linspace(0, 0.85, len(snap_idx)))
axs[0].scatter(X, Y, color='#e63946', zorder=5, label='data')
for idx, col in zip(snap_idx, cmap):
    mm, bb, mse = hist_line[idx]
    axs[0].plot(xs, mm*xs+bb, color=col, label=f'iter {idx}: MSE={mse:.3f}')
axs[0].set_title('Line fit - coordinate Newton-Raphson steps')
axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3)
axs[0].set_xlabel('x'); axs[0].set_ylabel('y')

snap_idx2 = [0, 3, 9, len(hist_par)-1]
axs[1].scatter(X, Y, color='#e63946', zorder=5, label='data')
for idx, col in zip(snap_idx2, cmap):
    aa, bb, cc, mse = hist_par[idx]
    axs[1].plot(xs, aa*xs**2+bb*xs+cc, color=col, label=f'iter {idx}: MSE={mse:.4f}')
axs[1].set_title('Parabola fit - coordinate Newton-Raphson steps')
axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3)
axs[1].set_xlabel('x'); axs[1].set_ylabel('y')

plt.tight_layout()
plt.savefig('media/part2_fit_progression.png')
plt.close()

# ---------------------------------------------------------------
# 4. Plot: MSE convergence
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(hist_line[:, 2], 'o-', color='#457b9d', label='Line MSE')
ax.plot(hist_par[:, 3], 's-', color='#e76f51', label='Parabola MSE')
ax.set_yscale('log')
ax.set_xlabel('Alternating Newton-Raphson step')
ax.set_ylabel('MSE (log scale)')
ax.set_title('MSE convergence: coordinate-wise Newton-Raphson')
ax.legend(); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('media/part2_mse_convergence.png')
plt.close()

print("\nAll Part 2 plots saved to media/")