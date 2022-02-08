"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np


def my_range(a: float, b: float, n: int):
    return np.arange(a, b + (b - a) / n, (b - a) / n)


def trapezoidal(f: [[float], float], a: float, b: float, n: int) -> float:
    Xs = my_range(a, b, n)

    def trapeze_area(a0: float, b0: float) -> float:
        return (b0 - a0) * ((f(b0) + f(a0)) / 2)

    area = 0
    for i in range(0, Xs.size - 1):
        area = area + trapeze_area(Xs[i], Xs[i + 1])
    return area


def simpson(f: [[float], float], a: float, b: float, n: int) -> float:
    Xs = my_range(a, b, n)
    print(Xs)

    h = (b - a) / (Xs.size - 1)
    F0 = f(a)
    F1 = 0
    F2 = f(b)
    for i in range(1, Xs.size - 1):
        print(i)
        if i % 2 == 0:
            F0 = F0 + f(Xs[i])
            F2 = F2 + f(Xs[i])
        else:
            F1 = F1 + f(Xs[i])
    return (h / 3) * (F0 + 4 * F1 + F2)


def poligonArea(points) -> float:
    def trapeze_area(p0: (float, float), p1: (float, float)) -> float:
        sing = 1
        if p0[0] > p1[0]: sing = -1
        return (p1[0] - p0[0]) * ((p1[1] + p0[1]) / 2)

    area = trapeze_area(points[len(points) - 1], points[0])
    for i in range(len(points) - 1):
        area = area + trapeze_area(points[i], points[i + 1])
    return area


def least_squar(points, f):
    output = 0
    for i in range(len(points)):
        output = output + (f(points[i][0]) - points[i][1]) ** 2
    return output


##########################################################################


if __name__ == "__main__":
    def f1(x: float) -> float: return 5 * np.exp(- (x ** 2))


    def f2(x: float) -> float: return np.sqrt(1 - (x ** 4))


    def f3(x: float) -> float: return np.log(6 * np.log(x))


    def f4(x: float) -> float: return np.exp(-x) / x


    def f5(x: float) -> float: return np.power(x, (np.e - 1)) * np.exp(-x)


    # print("ans 1 =", simpson(f1, 0, 4, 4))
    # print("ans 2 =", trapezoidal(f2, 0, 1, 5))
    # print("ans 3 =", simpson(f3, 2, 5, 4))
    # print("ans 4 =", simpson(f4, 2, 5, 4))
    # print("ans 5 =", simpson(f5, 2, 5, 4))

    def f6(x: float) -> float: return np.sin(x) * np.sqrt(3 - x ** 2 + x ** 3)


    def f7(x: float) -> float: return x ** 2 + 1 / (np.sin(x) + x ** 2)


    print("ans 1 =", trapezoidal(f6, 0, 2 * np.pi, 5))
    print("ans 2 =", simpson(f7, 0.5, 3, 4))

    A = [(1.35, 2.1), (2.69, 4), (5.1, 3), (7.9, 5.8), (11, 0.6), (7, -0.5), (8.7, -1.9), (5.32, -3.8), (-1, -2.5),
         (2.5, -1.5)]
    print(poligonArea(A))

    B = [(1, 2),
         (-1, 1),
         (2, 4),
         (-2, 4),
         (2, 3.21),
         (0.2, 1),
         (0.87, 0.7),
         (1.46, 1.36),
         (1.53, 1.48),
         (1.86, 1.5),
         (0.2, -0.03),
         (-1.33, -0.48),
         (-0.9, -1.3214),
         (2.3, 2.6),
         (-0.231, 0.2),
         (-0.399, 0),
         (-0.53, 0.1),
         (-0.6, -0.7),
         (3, 2.81),
         (-2, -1.8),
         (2.68, 4),
         (2.5, 2.5)]


    def f10(x: float) -> float: return x ** 2


    def f11(x: float) -> float: return 1.13 * x + 0.2


    def f12(x: float) -> float: return x


    def f13(x: float) -> float: return -x + 1


    def f14(x: float) -> float: return 0.9 * x + 0.86


    print("least squar:")
    print(least_squar(B, f10))
    print(least_squar(B, f11))
    print(least_squar(B, f12))
    print(least_squar(B, f13))
    print(least_squar(B, f14))
