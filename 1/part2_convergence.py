"""
SYDE572 A1 Part 2: how many coordinate-wise Newton-Raphson sweeps are needed
to actually reach the analytical answer (same updates as part2_fit_and_plot.py,
but with a stopping rule instead of a fixed number of loops).
Run from inside the 1/ folder: python3 part2_convergence.py
"""
import numpy as np

X = np.array([0, 2, 1, 3], dtype=float)
Y = np.array([0.5, 3.5, 1.5, 7.5])
n = len(X)

def run(A, names, tol=1e-8, max_sweeps=100000, show_first=0):
    w_star = np.linalg.lstsq(A, Y, rcond=None)[0]
    mse_star = np.mean((A @ w_star - Y) ** 2)
    w = np.zeros(A.shape[1])
    sweeps = 0
    within = None
    for s in range(1, max_sweeps + 1):
        w_old = w.copy()
        for j in range(A.shape[1]):              # one Newton step per parameter
            r = A @ w - Y
            g = (2 / n) * A[:, j] @ r
            h = (2 / n) * A[:, j] @ A[:, j]
            w[j] -= g / h
        mse = np.mean((A @ w - Y) ** 2)
        if s <= show_first:
            print(f"  sweep {s}: " + ", ".join(f"{nm}={v:.6f}" for nm, v in zip(names, w)) + f", MSE={mse:.6f}")
        if within is None and mse - mse_star < 1e-6:
            within = s
        sweeps = s
        if np.max(np.abs(w - w_old)) < tol:
            break
    mse = np.mean((A @ w - Y) ** 2)
    print("  final: " + ", ".join(f"{nm}={v:.6f}" for nm, v in zip(names, w)) + f", MSE={mse:.6f}")
    print("  analytical: " + ", ".join(f"{nm}={v:.6f}" for nm, v in zip(names, w_star)) + f", MSE={mse_star:.6f}")
    print(f"  sweeps until MSE within 1e-6 of the minimum: {within}")
    print(f"  sweeps until every parameter changes by < {tol:g}: {sweeps}")
    print(f"  (each sweep = {A.shape[1]} one-parameter Newton steps)")
    print(f"  largest parameter error vs analytical: {np.max(np.abs(w - w_star)):.2e}")

print("LINE  y = m*x + b")
run(np.column_stack([X, np.ones(n)]), ["m", "b"], show_first=3)
print("\nPARABOLA  y = a*x^2 + b*x + c")
run(np.column_stack([X**2, X, np.ones(n)]), ["a", "b", "c"], show_first=3)
