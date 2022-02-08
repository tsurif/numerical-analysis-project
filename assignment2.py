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

        epsilon = 0.0001

        def derivative_at_point(x):
            return (f(x + epsilon) - f(x)) / epsilon

        def sec_derivative_at_point(x):
            return (derivative_at_point(f(x + epsilon)) - derivative_at_point(f(x))) / epsilon

        def bisection(x0: float, x1: float, step_count):
            mid = (x0 + x1) / 2
            #print(x0, mid, x1)
            #if (mid == x0) | (mid == x1):
            #    return mid
            if np.abs(f(mid)) < maxerr:
                print("#######################################################found:", f(mid))
                print("#######################################################after", step_count, "steps")
                return mid
            if f(mid) * f(x0) < 0:
                return bisection(x0, mid, step_count + 1)
            else:
                return bisection(mid, x1, step_count + 1)

        def all_roots_iter() -> Iterable:
            solutions = np.array([])
            x = a
            while x < b:
                print("starting@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                print("x =", x)
                print("     f(", x,")=", f(x))
                dx = derivative_at_point(x)
                if abs(dx) > 1:
                    x1 = x - (f(x) / dx)
                else:
                    x1 = x + 0.1
                if x1 < x:
                    x1 = x + 0.1

                if f(x) * f(x1) < 0 and x1 > x:
                    print("start bisection with:#########################################################################")
                    print("x =", x)
                    print("f(x)=", f(x))
                    print("x1 =", x1)
                    print("f(x1)=", f(x1))

                    solutions = np.append(solutions, bisection(x, x1, 0))

                print("in this round x was", x, "and x1 was", x1)
                x = x1


            return solutions

        output = all_roots_iter()
        print(f(output))
        return output


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -2, 2, maxerr=0.001)
        print(X)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)
        print(f1)
        print(f2)

        X = ass2.intersections(f1, f2, -1000, 1000, maxerr=0.01)
        print(X)

        for x in X:
            self.assertGreaterEqual(0.01, abs(f1(x) - f2(x)))

    def test_my_poly(self):

        ass2 = Assignment2()

        f0 = np.poly1d([-5, -2, 0, 2, 5], True)
        f1 = lambda a: f0(a)/10
        f2 = lambda a: 0
        print(f1)
        print(f2)

        X = ass2.intersections(f1, f2, -10, 10, maxerr=0.01)
        print(X)

        for x in X:
            self.assertGreaterEqual(0.01, abs(f1(x) - f2(x)))

    def test_sin(self):

        ass2 = Assignment2()

        f1 = np.sin
        f2 = lambda a: 0

        X = ass2.intersections(f1, f2, -10, 10, maxerr=0.01)
        print(X)

        for x in X:
            self.assertGreaterEqual(0.01, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()
