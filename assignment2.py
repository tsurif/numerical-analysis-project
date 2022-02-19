"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.
        
        This function may not work correctly if there is infinite number of
        intersection points. 


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        def f(x: float) -> float:
            return f1(x) - f2(x)

        epsilon = 0.000001

        def derivative_at_point(x):
            dy = f(x + epsilon) - f(x)
            dx = epsilon
            return dy / dx

        def sec_derivative_at_point(x, dx):
            # return (derivative_at_point(x + epsilon) - derivative_at_point(x)) / epsilon
            return (derivative_at_point(x + epsilon) - dx) / epsilon

        def bisection(x0: float, x1: float):
            mid = (x0 + x1) / 2
            fmid = f(mid)
            if np.abs(fmid) < maxerr:
                return mid
            if fmid * f(x0) < 0:
                return bisection(x0, mid)
            else:
                return bisection(mid, x1)

        # def bisection_test(x0: float, x1: float, step_count):
        #     mid = (x0 + x1) / 2
        #     fmid = f(mid)
        #     if np.abs(fmid) < maxerr:
        #         print("#######################################################found:", mid, f(mid))
        #         print("#######################################################after", step_count, "steps")
        #         return mid
        #     print("[", f(x0), f(mid), f(x1), "]")
        #     print("[", x0, mid, x1, "]")
        #     if fmid * f(x0) < 0:
        #         print(">>>")
        #         return bisection_test(x0, mid, step_count + 1)
        #     else:
        #         print("<<<")
        #         return bisection_test(mid, x1, step_count + 1)

        def all_roots_iter() -> Iterable:
            solutions = np.array([])
            x = a
            while x < b:
                fx0 = f(x)

                if np.abs(fx0) < maxerr:
                    # print("found", x, fx0, " without bisection")
                    solutions = np.append(solutions, x)
                    x1 = x + maxerr * 1.001
                else:
                    dx0 = derivative_at_point(x)
                    ddx0 = sec_derivative_at_point(x, dx0)

                    if (f(x) > 0 and ddx0 < 0 and dx0 < -1) or (f(x) < 0 and ddx0 > 0 and dx0 > 1):
                       # print("using g")
                       # print("     f(", x, ")=", fx0)
                       # print("     f'(", x, ")=", dx0)
                       # print("     f''(", x, ")=", ddx0)
                       a0 = ddx0 / 2
                       b0 = dx0 - 2 * a0 * x
                       c0 = fx0 - a0 * x * x - b0 * x
                       x1 = max((-b0 - np.sqrt(b0 ** 2 - 4 * a0 * c0)) / (2 * a0),
                                (-b0 + np.sqrt(b0 ** 2 - 4 * a0 * c0)) / (2 * a0))

                       if x1 - x < maxerr:
                           x1 = x + maxerr * 1.001

                    elif (f(x) > 0 and ddx0 < 0 and dx0 > 0.5) or (f(x) < 0 and ddx0 > 0 and dx0 < -0.5):
                       # print("using reverse g")
                       # print("     f(", x, ")=", fx0)
                       # print("     f'(", x, ")=", dx0)
                       # print("     f''(", x, ")=", ddx0)
                       a0 = ddx0 / 2
                       b0 = dx0 - 2 * a0 * x
                       c0 = fx0 - a0 * x * x - b0 * x
                       x1 = min((-b0 - np.sqrt(b0 ** 2 - 4 * a0 * c0)) / (2 * a0),
                                (-b0 + np.sqrt(b0 ** 2 - 4 * a0 * c0)) / (2 * a0))
                       x1 = x + (x - x1)
                       if x1 - x < maxerr:
                           x1 = x + maxerr * 1.001
                    elif abs(dx0) > 0.9:
                        # print("using newton")
                        # if fx0 * dx0 > 0:
                        #     print("reverse")
                        # print("x =", x)
                        # print("     f(", x, ")=", fx0)
                        # print("     f'(", x, ")=", dx0)
                        # print("     f''(", x, ")=", ddx0)

                        x1 = max(x - (fx0 / dx0), x + (fx0 / dx0), x + maxerr * 1.001)
                    else:
                        # print("not using")
                        x1 = x + maxerr * 100

                    x1 = min(x1, b)
                    if f(x) * f(x1) < 0 and x1 > x:
                        solutions = np.append(solutions, bisection(x, x1))
                        # solutions = np.append(solutions, bisection_test(x, x1, 0))
                        x1 = x1 + maxerr * 1.001

                # print("in this round x was", x, "and x1 was", x1)
                x = x1

            return solutions

        return all_roots_iter()


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1000, 1000, maxerr=0.001)
        # print(X)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(6)

        X = ass2.intersections(f1, f2, -1000000, 1000000, maxerr=0.0001)
        # print(X)

        for x in X:
            self.assertGreaterEqual(0.0001, abs(f1(x) - f2(x)))

    def test_my_poly(self):

        ass2 = Assignment2()

        f0 = np.poly1d([2, 60, 61], True)
        f1 = lambda a: f0(a)
        f2 = lambda a: 0
        # print(f1)
        # print(f2)
        err = 0.01
        X = ass2.intersections(f1, f2, 2, 61.001, maxerr=err)
        # print(len(X))
        # print(X)

        oldx = -100000
        for x in X:
            if oldx != -100000 and x - oldx <= err:
                print("fail...")
            oldx = x

        for x in X:
            self.assertGreaterEqual(err, abs(f1(x) - f2(x)))

    def test_my_poly2(self):

        ass2 = Assignment2()

        f0 = np.poly1d([1, -10, 1, -1, -5, 0, 2, 0.5])
        f1 = lambda a: f0(a)
        f2 = lambda a: 0
        # print(f1)
        # print(f2)
        err = 0.000001
        X = ass2.intersections(f1, f2, -9999, 9999, maxerr=err)
        # print(len(X))
        print(X)

        oldx = -100000
        for x in X:
            if oldx != -100000 and x - oldx <= err:
                print("fail...")
            oldx = x

        for x in X:
            self.assertGreaterEqual(err, abs(f1(x) - f2(x)))

    def test_my_poly3(self):

        ass2 = Assignment2()

        f1 = np.poly1d([1, 2, 4], True)
        f2 = np.poly1d([1.1, 3], True)
        # print(f1)
        # print(f2)
        err = 0.001
        X = ass2.intersections(f1, f2, -100000, 20000, maxerr=err)
        # print(len(X))
        # print(X)

        oldx = -100000
        for x in X:
            if oldx != -100000 and x - oldx <= err:
                print("fail...")
            oldx = x

        for x in X:
            self.assertGreaterEqual(err, abs(f1(x) - f2(x)))

    def test_sin(self):

        ass2 = Assignment2()

        f1 = lambda a: np.sin(a)
        f2 = lambda a: 0
        err = 0.01
        X = ass2.intersections(f1, f2, 0, np.pi * 100000, maxerr=err)
        print(len(X))
        # print(X)

        oldx = -100000
        for x in X:
            if oldx != -100000 and x - oldx <= err:
                print("fail...")
            oldx = x
        for x in X:
            self.assertGreaterEqual(err, abs(f1(x) - f2(x)))

    def test_tan(self):

        ass2 = Assignment2()

        f1 = lambda a: np.tan(a)
        f2 = lambda a: 0
        err = 0.01
        X = ass2.intersections(np.tan, lambda a: 0, -np.pi/2 + 0.00000001, np.pi/2 - 0.00000001, maxerr=err)
        # print(len(X))
        print(X)

        oldx = -100000
        for x in X:
            if oldx != -100000 and x - oldx <= err:
                print("fail...")
            oldx = x
        for x in X:
            self.assertGreaterEqual(err, abs(f1(x) - f2(x)))

    def test_sin_sqr(self):

        ass2 = Assignment2()

        def f3_nr(a):
            return np.sin(pow(a, 2))

        def f10(a):
            return np.sin(np.log(a))

        err = 0.0001
        X = ass2.intersections(f3_nr, f10, 1, 10, maxerr=err)
        print(X)
        print(f3_nr(X) - f10(X))
        print(len(X))

        oldx = -100000
        for x in X:
            if oldx != -100000 and x - oldx <= err:
                print("fail...")
            oldx = x

        for x in X:
            self.assertGreaterEqual(err, abs(f3_nr(x) - f10(x)))


if __name__ == "__main__":
    unittest.main()
