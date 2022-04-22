"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""

import numpy as np
import time
import random
from assignment2 import *



def my_range(a: float, b: float, n: int):
    #todo: delete the next line!
    n = n - 1
    if n % 2 == 0:
        n = n - 1
    return np.linspace(a, b, n)


class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    # def print_result(self, expected_result, result, T):
    #     print("expected:", expected_result)
    #     print("got:", result)
    #     print("error:", np.abs(expected_result - result))
    #     print("time:", T)

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the integration error. 
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions. 
        
        Integration error will be measured compared to the actual value of the 
        definite integral. 
        
        Note: It is forbidden to call f more than n times. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """
        Xs = my_range(a, b, n)
        h = (b - a) / (Xs.size - 1)
        F0 = f(a)
        F1 = 0
        F2 = f(b)
        for i in range(1, Xs.size - 1):
            FX = f(Xs[i])
            if i % 2 == 0:
                F0 = F0 + FX
                F2 = F2 + FX
            else:
                F1 = F1 + FX
        return np.float32((h / 3) * (F0 + 4 * F1 + F2))

    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds 
        all intersection points between the two functions to work correctly. 
        
        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area. 
        
        In order to find the enclosed area the given functions must intersect 
        in at least two points. If the functions do not intersect or intersect 
        in less than two points this function returns NaN.  
        This function may not work correctly if there is infinite number of 
        intersection points. 
        

        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """
        Xs = Assignment2.intersections(self, f1, f2, 1, 100)

        def f(x: float) -> float:
            return f1(x) - f2(x)

        output = 0
        for i in range(len(Xs) - 1):
            output = output + np.abs(self.integrate(f, Xs[i], Xs[i + 1], 200))

        return output


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment3(unittest.TestCase):

    def test_poly_deg_2(self):
        ass3 = Assignment3()
        f1 = np.poly1d([2, 0, 0])
        r = ass3.integrate(f1, 0, 1, 5)
        true_result = 2 / 3
        # Assignment3.print_result(self, true_result, r)
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_sin(self):
        ass3 = Assignment3()
        f1 = np.sin
        r = ass3.integrate(f1, 0, 10001 * np.pi, 100000)
        true_result = -np.cos(1001 * np.pi) - (-np.cos(0))
        # Assignment3.print_result(self, true_result, r)
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))


    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        r = ass3.integrate(f1, -1, 1, 10)

        self.assertEqual(r.dtype, np.float32)

    def test_area_polys(self):
        ass3 = Assignment3()
        f1 = np.poly1d([2, -11, 13])
        f2 = np.poly1d([-3, 20, -20])
        T = time.time()
        r = ass3.areabetween(f1, f2)
        T = time.time() - T
        true_result = (301 * np.sqrt(301)) / 150
        # Assignment3.print_result(self, true_result, r, T)

        self.assertGreaterEqual(0.1, abs((r - true_result) / true_result))

    def test_area_polys_2(self):
        ass3 = Assignment3()
        f1 = np.poly1d([1, 2, 4], True)
        f2 = np.poly1d([1.1, 3], True)
        T = time.time()
        r = ass3.areabetween(f1, f2)
        T = time.time() - T
        true_result = 5.66026
        # Assignment3.print_result(self, true_result, r, T)

        self.assertGreaterEqual(0.1, abs((r - true_result) / true_result))

    def test_area_polys_3(self):
        ass3 = Assignment3()
        f1 = np.poly1d([1, 100], True)
        f2 = lambda a: 0
        T = time.time()
        r = ass3.areabetween(f1, f2)
        T = time.time() - T
        true_result = 161716.5
        # Assignment3.print_result(self, true_result, r, T)

        self.assertGreaterEqual(0.1, abs((r - true_result) / true_result))

    def test_area_sin(self):
        ass3 = Assignment3()
        f1 = np.sin
        T = time.time()
        r = ass3.areabetween(f1, lambda a: 0)
        T = T - time.time()
        true_result = 60
        # Assignment3.print_result(self, true_result, r, T)

        self.assertGreaterEqual(0.1, abs((r - true_result) / true_result))

    def test_area_sin2(self):
        ass3 = Assignment3()

        def f(x: float) -> float:
            return np.sin(10 / x) - 0.3
        T = time.time()
        r = ass3.areabetween(f, lambda a: 0)
        T = T - time.time()
        true_result = 9.77332
        # Assignment3.print_result(self, true_result, r, T)

        self.assertGreaterEqual(0.1, abs((r - true_result) / true_result))







if __name__ == "__main__":
    unittest.main()
