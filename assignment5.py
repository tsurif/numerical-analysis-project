"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""

import numpy as np
import time
import random
from functionUtils import AbstractShape
from scipy.optimize import fmin




class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        """
        Compute the area of the shape with the given contour.

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """

        n = 10000
        points = contour(n)
        output = 0
        if points[n - 1][0] != points[0][0]:
            output = (points[n - 1][1] + points[0][1]) * (points[n - 1][0] - points[0][0])
        for i in range(n - 1):
            if points[i][0] != points[i + 1][0]:
                output = output + ((points[i][1] + points[i + 1][1]) * (points[i][0] - points[i + 1][0]))

        output = np.abs(output) / 2
        return output

    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        sample : callable.
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        An object extending AbstractShape.
        """

        class MyShape(AbstractShape):
            # change this class with anything you need to implement the shape
            def __init__(self, vartices):
                # super(MyShape, self).__init__()
                self._vartices = []
                for i in range(0, len(vartices), 2):
                    self._vartices.append((vartices[i], vartices[i+1]))
                pass

            def area(self) -> np.float32:
                print(self._vartices)
                n = len(self._vartices)
                area = 0
                if self._vartices[n - 1][0] != self._vartices[0][0]:
                    area = (self._vartices[n - 1][1] + self._vartices[0][1]) * (self._vartices[n - 1][0] - self._vartices[0][0])
                for i in range(n - 1):
                    if self._vartices[i][0] != self._vartices[i + 1][0]:
                        area = area + ((self._vartices[i][1] + self._vartices[i + 1][1]) * (self._vartices[i][0] - self._vartices[i + 1][0]))
                area = np.abs(area) / 2
                return area

        def min_max_vals(Array, index) -> (float, float):
            min = Array[0][index]
            max = Array[0][index]
            for tup in Array:
                if tup[index] < min:
                    min = tup[index]
                if tup[index] > max:
                    max = tup[index]
            return min, max

        def avarage(Grid):
            avg = 0
            count = 0
            for i in range(m):
                for j in range(m):
                    if Grid[i][j] > 0:
                        avg = avg + Grid[i][j]
                        count = count + 1
            avg = avg / count
            return avg

        def find_unit_size(Grid):
            buckets_len = m
            buckets = [0] * buckets_len
            for i in range(m):
                j = 0
                while j < m:
                    count = 0
                    while Grid[i][j] != 0 and j < m:
                        count = count + 1
                        j = j + 1
                    if 0 < count < buckets_len:
                        buckets[count] = buckets[count] + 1
                    j = j + 1
            print(buckets)
            first_max = 1
            for i in range(2, buckets_len):
                if buckets[i] > buckets[first_max]:
                    first_max = i
            return first_max

        def find_start(Grid):
            for i in range(m):
                for j in range(m):
                    if Grid[i][j] > 0:
                        return i, j
            return 0, 0

        def weight(Grid, i, j, size):
            """
            Parameters
            ----------
            i, j the indexes of the Bottom right corner of a square of size
            Returns
            -------
            the square's cells sum
            """
            output = 0
            for k in range(i, i + size):
                for l in range(j, j + size):
                    if 0 <= k < m and 0 <= l < m:
                        output = output + Grid[k][l]
            return output

        def step(i, j, size, expensive_side):
            print("p = ", i, j)
            max_weight_unit = (i + size, j - size)
            max_weight = weight(Grid, i + size, j - size, size)
            for k in range(-size, size + 1):
                if 0 <= i - size < m and 0 <= j + k < m:
                    current_weight = weight(Grid, i + size, j + k, size)
                    if current_weight > max_weight:
                        max_weight_unit = (i + size, j + k)
                        max_weight = current_weight
            step(max_weight_unit[0], max_weight_unit[1], size, expensive_side)







        def clean_grid(Grid, avg):
            for i in range(m):
                for j in range(m):
                    if Grid[i][j] < avg / 2:
                        Grid[i][j] = 0

        def print_grid(Grid):
            for i in range(m):
                for j in range(m):
                    if Grid[i][j] > 0:
                        print(Grid[i][j], end="\t")
                    else:
                        print(".", end="\t")
                print(" ")

        result = MyShape([])
        Points = [(0,0)] * 0
        n = 100000
        m = 35
        for i in range(n):
            Points.append(sample())

        Xmin, Xmax = min_max_vals(Points, 0)    # happy Xmas
        Ymin, Ymax = min_max_vals(Points, 1)    # happy Ynukka
        dx = (Xmax - Xmin)/m
        dy = (Ymax - Ymin)/m
        Grid = np.array([[0]* (m + 1)] * (m + 1))

        for point in Points:
            x_bucket = np.floor((point[0] - Xmin) / dx)
            y_bucket = np.floor((point[1] - Ymin) / dy)
            Grid[int(x_bucket)][int(y_bucket)] = Grid[int(x_bucket)][int(y_bucket)] + 1

        print_grid(Grid)
        avg = avarage(Grid)
        print(avg)
        clean_grid(Grid, avg * 1.5)
        print_grid(Grid)
        unit_size = find_unit_size(Grid)
        print(unit_size)
        start_unit = find_start(Grid)
        step(start_unit[0], start_unit[1], unit_size, 0)




        avg = avarage(Grid)
        # Points = [(0, 0)] * 0
        # n = 75
        # num_of_v = 4
        # for i in range(n):
        #     Points.append(sample())
        # Xmin, Xmax = min_max_vals(Points, 0)  # happy Xmas
        # Ymin, Ymax = min_max_vals(Points, 1)  # happy Ynukka
        # # print("MINMAX X:", Xmin, Xmax)
        # # print("MINMAX Y:", Ymin, Ymax)
        # print(Points)
        # Points = np.array(Points)
        #
        # def func(vertices):
        #     num_of_vartex = vertices.size - 1
        #     def find_m_n():
        #         Ms = [0] * 0
        #         Ns = [0] * 0
        #         for i in range(0, num_of_vartex - 2, 2):
        #             if np.abs(vertices[i] - vertices[i + 2]) > 0.000001:
        #                 m = (vertices[i + 1] - vertices[i + 3]) / (vertices[i] - vertices[i + 2])
        #             else:
        #                 m = 9999999 * (vertices[i + 1] - vertices[i + 3])
        #             n = vertices[i + 1] - m * vertices[i]
        #             Ms.append(m)
        #             Ns.append(n)
        #         m = (vertices[num_of_vartex] - vertices[1]) / (vertices[num_of_vartex - 1] - vertices[0])
        #         n = vertices[num_of_vartex] - m * vertices[num_of_vartex - 1]
        #         Ms.append(m)
        #         Ns.append(n)
        #         return Ms, Ns
        #
        #     def min_distance(p):
        #         Ms, Ns = find_m_n()
        #         Ds = [0] * 0
        #         for i in range(int(num_of_vartex / 2)):
        #             m = Ms[i]
        #             n = Ns[i]
        #             if is_in_interval(p, n, m, i):
        #                 d = np.power((m * p[0] - p[1] + n) / np.sqrt(m ** 2 + 1), 2)
        #                 Ds.append(d)
        #             else:
        #                 d = distance_to_vartex(p, i)
        #                 Ds.append(d)
        #         return np.min(Ds)
        #
        #     def is_in_interval(p, n, m , i):
        #         # y = mx + n
        #         # y = -1/m x + n2
        #         # mx + n = -1/m x + n2
        #         # (m + 1/m)x = n2 - n
        #         # x = (n2 - n)/(m + 1/m)
        #         # if m > 0.000000001:
        #         n2 = p[0] / m + p[1]
        #         x = (n2 - n) / (m + 1 / m)
        #         y = m * x + n
        #         v1x = (2 * i) % num_of_vartex
        #         v2x = (2 * (i + 1)) % num_of_vartex
        #         v1y = (2 * i) % num_of_vartex + 1
        #         v2y = (2 * (i + 1)) % num_of_vartex + 1
        #         if vertices[v1x] < x and vertices[v2x] < x:
        #             return False
        #         if vertices[v1x] > x and vertices[v2x] > x:
        #             return False
        #         if vertices[v1y] < x and vertices[v2y] < y:
        #             return False
        #         if vertices[v1y] > x and vertices[v2y] > y:
        #             return False
        #         return True
        #
        #     def distance_to_vartex(p, i):
        #         v1x = (2 * i) % num_of_vartex
        #         v2x = (2 * (i + 1)) % num_of_vartex
        #         v1y = (2 * i) % num_of_vartex + 1
        #         v2y = (2 * (i + 1)) % num_of_vartex + 1
        #         d1 = np.power(p[0] - vertices[v1x], 2) + np.power(p[1] - vertices[v1y], 2)
        #         d2 = np.power(p[0] - vertices[v2x], 2) + np.power(p[1] - vertices[v2y], 2)
        #         return min(d1, d2)
        #
        #     def extra_vartix_wight(v):
        #         w = np.inf
        #         for p in Points:
        #             w0 = np.power(p[0] - v[0], 2) + np.power(p[1] - v[1], 2)
        #             if w > w0:
        #                 w = w0
        #         return w
        #
        #     def line_lens():
        #         output = 0
        #         for i in range(0, int(num_of_vartex/2)):
        #             v1x = (2 * i) % num_of_vartex
        #             v2x = (2 * (i + 1)) % num_of_vartex
        #             v1y = (2 * i) % num_of_vartex + 1
        #             v2y = (2 * (i + 1)) % num_of_vartex + 1
        #             output = output + np.sqrt(np.power(v1x - v2x, 2) + np.power(v1y - v2y, 2))
        #         return output
        #
        #     s = 0
        #     for p in Points:
        #         s = s + min_distance(p)
        #     s = s ** 8
        #     for i in range(0, num_of_vartex, 2):
        #         v = (vertices[i], vertices[i + 1])
        #         s = s + extra_vartix_wight(v) ** 8
        #     s = s + line_lens()
        #     print(s)
        #     return s
        #
        # Xcenter = Xmin + (Xmax - Xmin)/2
        # Ycenter = Ymin + (Ymax - Ymin) / 2
        # radios = (Xmax - Xmin) / 3
        # # print(Xcenter, Ycenter, radios)
        #
        # vartices = [0] * 2 * num_of_v
        # for i in range(0, 2 * num_of_v, 2):
        #     vartices[i] = Xcenter + radios * np.cos(np.pi / 4 + (np.pi * i)/num_of_v)
        #     vartices[i + 1] = Ycenter + radios * np.sin(np.pi / 4 + (np.pi * i) / num_of_v)
        # for i in range(0, len(vartices), 2):
        #     print("(", vartices[i], ",", vartices[i + 1], ")")
        # output = fmin(func,
        #               vartices)
        # for i in range(0, output.size, 2):
        #     print("(", output[i], ",", output[i + 1], ")")
        # result = MyShape(output)
        # return result


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class polygon4(AbstractShape):
    def __init__(self, noise, tp):
        self._noise = noise
        self._tp = tp

    def sample(self):
        if self._tp == 0:
            side = random.random() * (2 + np.sqrt(2))
            t = 20 * random.random()
            if side <= 1:
                return -10 + t + np.random.normal() * self._noise, -10 + np.random.normal() * self._noise
            elif side <= 2:
                return - 10 + np.random.normal() * self._noise, -10 + t + np.random.normal() * self._noise
            else:
                return 10 - t + np.random.normal() * self._noise, - 10 + t + np.random.normal() * self._noise
        else:
            t = 200 * random.random()
            side = random.randint(0, 3)
            if side == 0:
                return 100 + np.random.normal() * self._noise, - 100 + t + np.random.normal() * self._noise
            if side == 1:
                return -100 + np.random.normal() * self._noise, - 100 + t + np.random.normal() * self._noise
            if side == 2:
                return -100 + t + np.random.normal() * self._noise, 100 + np.random.normal() * self._noise
            if side == 3:
                return -100 + t + np.random.normal() * self._noise, -100 + np.random.normal() * self._noise


    def contour(self, n: int):
        return 1

    def area(self):
        return 4
#

class TestAssignment5(unittest.TestCase):

    # def test_return(self):
    #     circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
    #     ass5 = Assignment5()
    #     T = time.time()
    #     shape = ass5.fit_shape(sample=circ, maxtime=5)
    #     T = time.time() - T
    #     self.assertTrue(isinstance(shape, AbstractShape))
    #     self.assertLessEqual(T, 5)
    #
    # def test_delay(self):
    #     circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
    #
    #     def sample():
    #         time.sleep(7)
    #         return circ()
    #
    #     ass5 = Assignment5()
    #     T = time.time()
    #     shape = ass5.fit_shape(sample=sample, maxtime=5)
    #     T = time.time() - T
    #     self.assertTrue(isinstance(shape, AbstractShape))
    #     self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=100, noise=3)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        print("area =", a)
        self.assertLess(abs(a - np.pi), 0.1)
        self.assertLessEqual(T, 32)

    def test_square_area(self):
        circ = polygon4(noise=10, tp=1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ.sample, maxtime=30)
        T = time.time() - T
        a = shape.area()
        print(a)
        self.assertLess(abs(a - 4), 0.2)
        self.assertLessEqual(T, 32)

    def test_triengle_area(self):
        circ = polygon4(noise=2, tp=0)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ.sample, maxtime=30)
        T = time.time() - T
        a = shape.area()
        print(a)
        self.assertLess(abs(a - 4), 0.2)
        self.assertLessEqual(T, 32)

    def test_triengle_quiet_area(self):
        circ = polygon4(noise=0, tp=0)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ.sample, maxtime=30)
        T = time.time() - T
        a = shape.area()
        print(a)
        self.assertLess(abs(a - 4), 0.2)
        self.assertLessEqual(T, 32)





    # def test_bezier_fit(self):
    #     circ = noisy_circle(cx=3, cy=3, radius=3, noise=0.3)
    #     ass5 = Assignment5()
    #     T = time.time()
    #     shape = ass5.fit_shape(sample=circ, maxtime=30)
    #     T = time.time() - T
    #     a = shape.area()
    #     self.assertLess(abs(a - np.pi), 0.01)
    #     self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
