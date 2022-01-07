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
    h = (b - a) / n
    F0 = f(a)
    F1 = 0
    F2 = f(b)
    for i in range(1, Xs.size - 1):
        if i % 2 == 0:
            F0 = F0 + f(Xs[i])
            F2 = F2 + f(Xs[i])
        else:
            F1 = F1 + f(Xs[i])
    return (h / 3) * (F0 + 4 * F1 + F2)

##########################################################################


if __name__ == "__main__":
    def f1(x: float) -> float: return 5 * np.exp(- (x ** 2))
    def f2(x: float) -> float: return np.sqrt(1 - (x ** 4))
    def f3(x: float) -> float: return np.log(6 * np.log(x))
    def f4(x: float) -> float: return np.exp(-x) / x
    def f5(x: float) -> float: return np.power(x, (np.e - 1)) * np.exp(-x)


    print("ans 1 =", simpson(f1, 0, 4, 4))
    print("ans 2 =", trapezoidal(f2, 0, 1, 5))
    print("ans 3 =", simpson(f3, 2, 5, 4))
    print("ans 4 =", simpson(f4, 2, 5, 4))
    print("ans 5 =", simpson(f5, 2, 5, 4))


