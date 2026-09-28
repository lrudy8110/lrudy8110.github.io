"""
SYDE572 Assignment 1 - Part 1
Core distance-to-curve routines: Newton-Raphson and Golden Section Search.
Works for any differentiable f(x), polynomial or not.
"""
import math
import numpy as np


def find_distance_newton(x0, y0, f, df, ddf, initial_guess=0.0,
                          tolerance=1e-7, max_iter=100, record=False):
    x = initial_guess
    history = [x]
    for _ in range(max_iter):
        D_prime = 2 * (x - x0) + 2 * (f(x) - y0) * df(x)
        D_double_prime = 2 + 2 * (df(x) ** 2) + 2 * (f(x) - y0) * ddf(x)
        next_x = x - D_prime / D_double_prime
        history.append(next_x)
        if abs(next_x - x) < tolerance:
            x = next_x
            break
        x = next_x
    shortest_distance = ((x - x0) ** 2 + (f(x) - y0) ** 2) ** 0.5
    if record:
        return shortest_distance, x, history
    return shortest_distance, x


def golden_section_search(x0, y0, f, a, b, tolerance=1e-7, record=False):
    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi

    def dist_sq(x):
        return (x - x0) ** 2 + (f(x) - y0) ** 2

    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f_x1 = dist_sq(x1)
    f_x2 = dist_sq(x2)

    history = [(a, b)]
    while abs(b - a) > tolerance:
        if f_x1 < f_x2:
            b = x2
            x2 = x1
            f_x2 = f_x1
            x1 = a + resphi * (b - a)
            f_x1 = dist_sq(x1)
        else:
            a = x1
            x1 = x2
            f_x1 = f_x2
            x2 = b - resphi * (b - a)
            f_x2 = dist_sq(x2)
        history.append((a, b))

    best_x = (a + b) / 2
    if record:
        return math.sqrt(dist_sq(best_x)), best_x, history
    return math.sqrt(dist_sq(best_x)), best_x