"""
SYDE572 A1 Part 1: Newton-Raphson iterates on the parabola, one figure per point.
Run from inside the 1/ folder: python3 part1_newton_all.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from point_to_curve import find_distance_newton

plt.rcParams.update({'font.size': 11, 'figure.dpi': 130})

f = lambda x: x**2 + 5
df = lambda x: 2*x
ddf = lambda x: 2
points = [(0, 0), (-4, 0), (-8, 0), (2, 0), (6, 0)]
MIN_GAP = 26  # minimum pixel distance between two labelled iterates

for (x0, y0) in points:
    d, xstar, hist = find_distance_newton(x0, y0, f, df, ddf, initial_guess=x0, record=True)
    n = len(hist) - 1
    lo = min(min(hist), x0) - 2
    hi = max(max(hist), x0) + 2
    xx = np.linspace(lo, hi, 400)
    top = max(f(np.array(hist)).max(), y0) * 1.15 + 2

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(xx, f(xx), color='#1d3557', lw=2, label=r'$y=x^2+5$')
    ax.plot(x0, y0, 'o', color='black', ms=8, label=f'point ({x0}, {y0})', zorder=5)
    ax.plot([x0, hist[-1]], [y0, f(hist[-1])], '--', color='gray',
            label=f'final distance d = {d:.4f}')
    norm = Normalize(0, max(n, 1))
    for i, xi in enumerate(hist):
        ax.plot(xi, f(xi), 's', color=plt.cm.viridis(0.9 * norm(i)), ms=7, zorder=4)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(f'Newton-Raphson iterates for ({x0}, {y0}): {n} iterations')
    ax.set_xlim(lo, hi); ax.set_ylim(-2, top); ax.grid(alpha=0.3)
    ax.legend(loc='upper center')

    sm = ScalarMappable(norm=norm, cmap=plt.cm.viridis.resampled(256) if hasattr(plt.cm.viridis, 'resampled') else plt.cm.viridis)
    sm.set_clim(0, 0.9 * max(n, 1) / max(n, 1) * max(n, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label('iteration (dark = early, light = late)')
    plt.tight_layout()
    fig.canvas.draw()

    # label first and last iterate, then any other one that has room
    order = [0] + ([n] if n > 0 else []) + list(range(1, n))
    placed = []
    for i in order:
        px, py = ax.transData.transform((hist[i], f(hist[i])))
        if all(np.hypot(px - qx, py - qy) >= MIN_GAP for qx, qy in placed):
            placed.append((px, py))
            ax.annotate(str(i), (hist[i], f(hist[i])), xytext=(-10, 9),
                        textcoords='offset points', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))

    fname = f'media/part1_newton_steps_{x0}_{y0}.png'
    plt.savefig(fname); plt.close()
    print(f"saved {fname}  ({n} iterations, d = {d:.6f})")
