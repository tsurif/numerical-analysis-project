"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random



def get_bezier_coef_with_thomas_algorithm(points):
    n = len(points) - 1
    a = np.ones(n)
    a[n - 1] = 2

    b = np.ones(n) * 4
    b[0] = 2
    b[n - 1] = 7

    c = np.ones(n)

    x = [2 * (2 * points[i][0] + points[i + 1][0]) for i in range(n)]
    x[0] = points[0][0] + 2 * points[1][0]
    x[n - 1] = 8 * points[n - 1][0] + points[n][0]

    y = [2 * (2 * points[i][1] + points[i + 1][1]) for i in range(n)]
    y[0] = points[0][1] + 2 * points[1][1]
    y[n - 1] = 8 * points[n - 1][1] + points[n][1]

    for i in range(1, n):
        w = a[i] / b[i - 1]
        b[i] = b[i] - w * c[i - 1]
        x[i] = x[i] - w * x[i - 1]
        y[i] = y[i] - w * y[i - 1]

    A = np.zeros((n, 2))
    A[n - 1][0] = x[n - 1] / b[n - 1]
    A[n - 1][1] = y[n - 1] / b[n - 1]

    for i in range(n - 2, -1, -1):
        A[i][0] = (x[i] - c[i] * A[i + 1][0]) / b[i]
        A[i][1] = (y[i] - c[i] * A[i + 1][1]) / b[i]

    # B = [0] * n
    B = np.zeros((n, 2))
    for i in range(n - 1):
        B[i] = 2 * points[i + 1] - A[i + 1]
    B[n - 1] = (A[n - 1] + points[n]) / 2

    return A, B


def get_cubic_x(a, b, c, d):
    return np.poly1d([-a[0] + 3 * b[0] - 3 * c[0] + d[0], 3 * a[0] - 6 * b[0] + 3 * c[0], -3 * a[0] + 3 * b[0], a[0]])


def get_cubic_y(a, b, c, d):
    return np.poly1d([-a[1] + 3 * b[1] - 3 * c[1] + d[1], 3 * a[1] - 6 * b[1] + 3 * c[1], -3 * a[1] + 3 * b[1], a[1]])


def derivative_at_point(f, x):
    dy = f(x + 0.000001) - f(x)
    return dy / 0.000001


def newton_raphson(f: callable, t0):
    ft0 = f(t0)
    if np.abs(ft0) < 0.00000000001:
        return t0
    else:
        dt0 = derivative_at_point(f, t0)
        t1 = t0 - (f(t0) / dt0)
        return newton_raphson(f, t1)


# evalute each cubic curve on the range [0, 1] sliced in n points
def evaluate_bezier(points, a, b):
    n = len(points) - 1
    A, B = get_bezier_coef_with_thomas_algorithm(points)
    xc = [get_cubic_x(points[i], A[i], B[i], points[i + 1]) for i in range(n)]
    yc = [get_cubic_y(points[i], A[i], B[i], points[i + 1]) for i in range(n)]

    def f(x0: int):
        i = int(np.floor(((x0 - a) / (b - a)) * n))
        root = newton_raphson((xc[i] - x0), 0)
        return yc[i](root)

    return f


class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
        """

        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time. 
        The assignment will be tested on variety of different functions with 
        large n values. 
        
        Interpolation error will be measured as the average absolute error at 
        2*n random points between a and b. See test_with_poly() below. 

        Note: It is forbidden to call f more than n times. 

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.** 
        
        Note: sometimes you can get very accurate solutions with only few points, 
        significantly less than n. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """
        if n == 1:
            output = f((b - a) / 2)
            return lambda x: output

        xs = np.linspace(a, b, n)
        points = [None] * xs.size
        for i in range(xs.size):
            points[i] = [xs[i], f(xs[i])]
        points = np.array(points)
        return evaluate_bezier(points, a, b)


##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):




    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)


if __name__ == "__main__":
    unittest.main()
