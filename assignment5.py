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

        n = 1500
        points = contour(n)
        output = (points[n - 1][1] + points[0][1]) * (points[n - 1][0] - points[0][0])
        for i in range(n - 1):
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
                self._vartices = vartices
                pass

            def area(self) -> np.float32:
                n = len(self._vartices)
                area = 0
                if n > 2:
                    if self._vartices[n - 1][0] != self._vartices[0][0]:
                        area = (self._vartices[n - 1][1] + self._vartices[0][1]) * (
                                self._vartices[n - 1][0] - self._vartices[0][0])
                    for i in range(n - 1):
                        if self._vartices[i][0] != self._vartices[i + 1][0]:
                            area = area + ((self._vartices[i][1] + self._vartices[i + 1][1]) * (
                                    self._vartices[i][0] - self._vartices[i + 1][0]))
                    area = np.abs(area) / 2
                return area

        class MyShapeWithArea(AbstractShape):
            # change this class with anything you need to implement the shape
            def __init__(self, shape_area):
                self._area = shape_area
                pass

            def area(self) -> np.float32:
                return self._area

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
            # print(buckets)
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
                    else:
                        return - 1
            return output

        def substract_weight(Grid, i, j, size, avg):
            for k in range(i, i + size):
                for l in range(j, j + size):
                    if 0 <= k < m and 0 <= l < m:
                        # Grid[k][l] = Grid[k][l] - avg
                        Grid[k][l] = Grid[k][l] / 5
                        # Grid[k][l] = 0

        def is_point_in_neighborhood(i, j, size, p):
            # size = np.ceil(size / 1.5)
            return i - size <= p[0] <= i + size and j - size <= p[1] <= j + size

        def step(Grid, i, j, size, avg, starting_point, can_stop, TTL):
            # print_grid(Grid, i, j, size)
            if can_stop < 0 and is_point_in_neighborhood(i, j, size, starting_point):
                return True, []
            if TTL == 0:
                return False, []

            substract_weight(Grid, i, j, size, avg)
            # print_grid(Grid, i, j, size)
            max_weight_unit = (i + size, j - size)
            max_weight = weight(Grid, i + size, j - size, size)
            for k in range(-size, size + 1):
                if 0 <= i + size < m and 0 <= j + k < m:
                    current_weight = weight(Grid, i + size, j + k, size)
                    if current_weight > max_weight:
                        max_weight_unit = (i + size, j + k)
                        max_weight = current_weight

                if 0 <= i - size < m and 0 <= j + k < m:
                    current_weight = weight(Grid, i - size, j + k, size)
                    if current_weight > max_weight:
                        max_weight_unit = (i - size, j + k)
                        max_weight = current_weight

                if 0 <= i + k < m and 0 <= j + size < m:
                    current_weight = weight(Grid, i + k, j + size, size)
                    if current_weight > max_weight:
                        max_weight_unit = (i + k, j + size)
                        max_weight = current_weight

                if 0 <= i + k < m and 0 <= j - size < m:
                    current_weight = weight(Grid, i + k, j - size, size)
                    if current_weight > max_weight:
                        max_weight_unit = (i + k, j - size)
                        max_weight = current_weight
            # print(max_weight)
            # if max_weight == 0:
            #     return []

            bool, vartices = step(Grid, max_weight_unit[0],
                 max_weight_unit[1], size,
                 avg, starting_point,
                 can_stop - 1, TTL - 1)
            return True and bool, [(Xmin + i * dx + 0.5 * dx * size, Ymin + j * dy + 0.5 * dy * size)] + vartices

        def clean_grid(Grid, threshold):
            for i in range(m):
                for j in range(m):
                    if Grid[i][j] < threshold:
                        Grid[i][j] = 0

        def count_full_squares(Grid):
            count = 0
            for i in range(m):
                for j in range(m):
                    if Grid[i][j] > 0:
                        count = count + 1
            return count

        Points = [(0, 0)] * 0
        n = 50000
        m = 35
        for i in range(n):
            Points.append(sample())

        Xmin, Xmax = min_max_vals(Points, 0)  # happy Xmas
        Ymin, Ymax = min_max_vals(Points, 1)  # happy Ynukka
        dx = (Xmax - Xmin) / m
        dy = (Ymax - Ymin) / m
        Grid = np.array([[0] * (m + 1)] * (m + 1))

        for point in Points:
            x_bucket = np.floor((point[0] - Xmin) / dx)
            y_bucket = np.floor((point[1] - Ymin) / dy)
            Grid[int(x_bucket)][int(y_bucket)] = Grid[int(x_bucket)][int(y_bucket)] + 1

        avg = avarage(Grid)
        clean_grid(Grid, avg * 0.2)
        unit_size = find_unit_size(Grid)
        count = 0
        while unit_size > 3 and count < 6:
            clean_grid(Grid, avg * 0.75)
            unit_size = find_unit_size(Grid)
            count = count + 1
            avg = avarage(Grid)
        if unit_size > 4:
            area = 0.6 * count_full_squares(Grid) * dx * dy
            return MyShapeWithArea(area)
        start_unit = find_start(Grid)

        bool, vartices = step(Grid, start_unit[0], start_unit[1], unit_size, avg, (start_unit[0], start_unit[1]), 3, m * 10)
        my_shape = MyShape(vartices)
        if my_shape.area() < 6 * dx * 6 * dy or (not bool):
            area = count_full_squares(Grid) * dx * dy
            return MyShapeWithArea(area)
        return my_shape


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm



class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)


if __name__ == "__main__":
    unittest.main()
