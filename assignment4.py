"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment. 

"""

import numpy as np
import time
import random


class Assignment4A:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        f : callable. 
            A function which returns an approximate (noisy) Y value given X. 
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int 
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        a function:float->float that fits f between a and b
        """

        #
        # T0 = time.time()
        # xs = np.linspace(a, b, d + 1)
        # ys = [0] * (d + 1)
        # counts = [0] * (d + 1)
        # count = 1
        # while time.time() - T0 < maxtime - 1:
        #     for i in range(xs.size):
        #         ys[i] = ys[i] + f(xs[i])
        #         counts[i] = count
        #         if time.time() - T0 > maxtime - 1:
        #             break
        #     count = count + 1
        # print(count)
        # for i in range(xs.size):
        #     ys[i] = ys[i] / counts[i]
        #
        # def interpolate():
        #     def L(i):
        #         j = 0
        #         filterd_xs = [None] * 0
        #         for x in xs:
        #             if j != i:
        #                 filterd_xs.append(x)
        #             j = j + 1
        #         poly_i = np.poly1d(filterd_xs, True)
        #         c = poly_i(xs[i])
        #         output = poly_i * (ys[i] / c)
        #         return output
        #
        #     Ps = [None] * 0
        #     for i in range(xs.size):
        #         Ps.append(L(i))
        #     output = np.poly1d([0])
        #     for p in Ps:
        #         output = output + p
        #     return output
        #
        # return interpolate()
        def solve(A, y):
            def row_multiplication(A, y, i):
                y[i] = y[i] / A[i][i]
                A[i] = A[i] / A[i][i]

            def row_addition(A, y, j, i, k):
                A[j] = A[j] + k * A[i]
                y[j] = y[j] + k * y[i]

            n = len(A)
            for i in range(n):
                row_multiplication(A, y, i)
                for j in range(n):
                    if i != j:
                        row_addition(A, y, j, i, -A[j][i])
            return

        T0 = time.time()
        Xs = [a, b]
        Ys = [f(a), f(b)]
        while time.time() - T0 < maxtime - 1:
            x = random.uniform(a, b)
            Xs.append(x)
            Ys.append(f(x))
        A = np.vander(Xs, d + 1)
        At = np.transpose(A)
        AtA = np.matmul(At, A)
        Atb = np.matmul(At, Ys)
        solve(AtA, Atb)
        return np.poly1d(Atb)
##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))
        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=-1, b=1, d=12, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(0.1)(NOISY(0.01)(poly(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)))

        ass4 = Assignment4A()
        T = time.time()
        mt = 10
        shape = ass4.fit(f=f, a=-1, b=1, d=12, maxtime=mt)
        T = time.time() - T
        self.assertGreaterEqual(mt, T)

    def test_err(self):
        f = poly(1, -2, 2, -2, -2, -2, 2, -94)
        df = DELAYED(0.0001)(f)
        nf = NOISY(1)(f)
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=-2, b=2, d=7, maxtime=20)
        print(ff)
        T = time.time() - T
        print("done in ", T)
        mse = 0
        for x in np.linspace(0, 1, 1000):
            mse += (f(x) - ff(x)) ** 2
        mse = mse / 1000
        print(mse)


if __name__ == "__main__":
    unittest.main()
