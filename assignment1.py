"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random


def get_bezier_coef(points):
    n = len(points) - 1

    # my implementation for thomas algorithm

    # magic numbers
    a = 0
    b = 1
    c = 2

    # T is the tri-diagonally matrix shift to nx3 matrix

    # building T
    T = np.ones([n, 3])
    T[1: n - 1, b] = 4

    T[n - 1, a] = 2
    T[n - 1, b] = 7
    T[n - 1, c] = 0

    T[0, a] = 0
    T[0, b] = 2
    T[0, c] = 1

    # building the solution vectors (the solution are points so we wil solve this twice for x and y)

    Dx = [2 * (2 * points[i, 0] + points[i + 1, 0]) for i in range(n)]
    Dx[0] = points[0, 0] + 2 * points[1, 0]
    Dx[n - 1] = 8 * points[n - 1, 0] + points[n, 0]

    Dy = [2 * (2 * points[i, 1] + points[i + 1, 1]) for i in range(n)]
    Dy[0] = points[0, 1] + 2 * points[1, 1]
    Dy[n - 1] = 8 * points[n - 1, 1] + points[n, 1]

    T[0, c] = np.divide(T[0, c], T[0, b])
    Dx[0] = np.divide(Dx[0], T[0, b])
    Dy[0] = np.divide(Dy[0], T[0, b])

    # TODO: change to map?? or something faster
    for i in range(1, n):
        T[i, c] = np.divide(T[i, c], np.subtract(T[i, b], np.multiply(T[i, a], T[i - 1, c])))

        Dx[i] = np.divide(np.subtract(Dx[i], np.multiply(T[i, a], Dx[i - 1])),
                          np.subtract(T[i, b], np.multiply(T[i, a], T[i - 1, c])))

        Dy[i] = np.divide(np.subtract(Dy[i], np.multiply(T[i, a], Dy[i - 1])),
                          np.subtract(T[i, b], np.multiply(T[i, a], T[i - 1, c])))

    A = np.zeros([n, 2])

    A[n - 1, 0] = Dx[n - 1]
    A[n - 1, 1] = Dy[n - 1]
    for i in range(n - 2, -1, -1):
        A[i, 0] = np.subtract(Dx[n - 1], np.multiply(T[i, c], A[i + 1, 0]))
        A[i, 1] = np.subtract(Dy[n - 1], np.multiply(T[i, c], A[i + 1, 1]))
    # code from the example
    B = [0] * n
    for i in range(n - 1):
        B[i] = 2 * points[i + 1] - A[i + 1]
    B[n - 1] = (A[n - 1] + points[n]) / 2
    # end of code from the example
    return A, B


# returns the general Bezier cubic formula given 4 control points
def get_cubic_x(a, b, c, d):
    return np.poly1d([-a[0] + 3 * b[0] - 3 * c[0] + d[0], 3 * a[0] - 6 * b[0] + 3 * c[0], -3 * a[0] + 3 * b[0], a[0]])


def get_cubic_y(a, b, c, d):
    return np.poly1d([-a[1] + 3 * b[1] - 3 * c[1] + d[1], 3 * a[1] - 6 * b[1] + 3 * c[1], -3 * a[1] + 3 * b[1], a[1]])


# evalute each cubic curve on the range [0, 1] sliced in n points
def evaluate_bezier(points, a, b):
    n = len(points) - 1
    A, B = get_bezier_coef(points)
    print(A)
    print(B)
    # xc is polynomial of t - P(t), that gives the x value of the curve for a given t, (P(t) - x0) roots will give me
    # the t that fit to a given x0
    xc = [
        get_cubic_x(points[i], A[i], B[i], points[i + 1])
        for i in range(n)
    ]

    # yc is polynomial of t, that gives the y value of the curve for a given t, with the root of the previous
    # (P(t) - x0) i can find the y value of x0
    yc = [
        get_cubic_y(points[i], A[i], B[i], points[i + 1])
        for i in range(n)
    ]

    def f(x0: int):
        if x0 == a:
            i = 0
        elif x0 == b:
            i = n - 1
        else:
            i = int((x0 - a) / ((b - a)/n))
        # print("\npoints:\n", points)
        # print("x0 =", x0)
        # print("a =", a)
        # print("b = ", b)
        # print("i =", i)
        print("\np =\n", xc[i], "x =", x0)
        print(xc[i] - x0)
        roots = np.roots(xc[i] - x0)
        t0 = roots[1]
        # t0 = 0
        print(roots)
        print("y(t) =\n", yc[i])
        # for j in range(roots.size):
        #     if 0 <= roots[j] <= 1 and np.isreal(roots[j]):
        #         t0 = roots[j]
        return yc[i - 1](t0)

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
        # bezier curves
        xs = np.arange(a, b + (b - a) / (n - 1), (b - a) / (n - 1))
        ys = f(xs)
        points = [None] * xs.size
        for i in range(xs.size):
            points[i] = [xs[i], ys[i]]
        points = np.array(points)
        return evaluate_bezier(points, a, b)


##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        numberoftest = 1
        numberofpointtotest = 10
        rangeofinterpolate = 5
        for i in tqdm(range(numberoftest)):

            f = lambda x: np.arctan(x)
            # f = lambda x: np.exp(-2 * np.power(x, 2))
            # f = lambda x: np.divide(np.sin(x), x)
            # f = lambda x: np.sin(x) / x
            # f = np.poly1d([7, 0, -10, 0])

            ff = ass1.interpolate(f, -rangeofinterpolate, rangeofinterpolate, 1000)

            xs = np.random.random(numberofpointtotest)
            xs = (xs - 0.5) * 2 * rangeofinterpolate
            xs = np.sort(xs)
            # xs = np.arange(-rangeofinterpolate, rangeofinterpolate + (2 * rangeofinterpolate) / (numberofpointtotest - 1), (2 * rangeofinterpolate) / (numberofpointtotest - 1))

            err = 0

            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)
                # print("")
                # print("f(", x, ") -> ", y)
                # print("ff(", x, ") -> ", yy)
                # print("err = ", abs(y - yy))

            err = err / numberofpointtotest
            mean_err += err
        mean_err = mean_err / numberoftest

        T = time.time() - T
        print(T)
        print(mean_err)

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
