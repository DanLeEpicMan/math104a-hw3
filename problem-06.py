'''
Daniel Naylor
5094024
11/09/2022
'''

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate as tab
from typing import List, Tuple

# --------------------------------------------------
#                      Problem 6
# --------------------------------------------------

class NaturalSplineInterpolation:
    '''
    A natural spline interpolation (piecewise cubic polynomial) 
    of a given dataset.

    This class acts like a mathematical function, 
    i.e. it can be called with a number as input
    and gives a number as output.
    '''
    def __init__(self, data: List[Tuple[float, float]]) -> None:
        '''
        Initializes a spline interpolation based on the given data.
        Data should be of the form:
        ```
        [
            (x_0, f_0),
            (x_1, f_1),
            ...
            (x_n, f_n)
        ]
        ```
        Each `x` value should be sorted. For example, `[(0, 1), (-1, 0)]` is an invalid data input,
        and instead should be `[(-1, 0), (0, 1)]`.
        '''
        n = len(data) - 1
        assert n > 0, 'Please provide a valid dataset (at least 2 points)'
        self._data = data

        z = { # natural spline, set z_0 and z_n first
            0: 0,
            n: 0
        }
        h = [ # compute values for h_j, length of each sub-interval.
            data[j+1][0] - data[j][0]
            for j in range(n) # 0, ..., n-1
        ]

        # solve for l, m values
        # use a dictionary for ease of assigning values
        l={} # using 1, ..., n-2
        m={1: 2 * (h[0] + h[1])} # 1, ..., n-1
        for j in range(1, n-1): # 1, ..., n-2
            l[j] = h[j]/m[j]
            m[j+1] = 2 * (h[j] + h[j+1]) - (l[j] * h[j])

        # compute values of y (what the sums of z add to)
        y={
            j: -6 * (data[j][1] - data[j-1][1])/(h[j-1]) + 6 * (data[j+1][1] - data[j][1])/h[j]
            for j in range(1, n) # 1, ..., n-1
        }

        # compute the values of k (what the matrix product of right-upper and z's produce)
        k={1: y[1]}
        for j in range(2, n):
            k[j] = y[j] - l[j-1] * k[j-1]

        # finally, compute the remaining z values using k
        # note: this abuses the lower triangular property
        z[n-1] = k[n-1]/m[n-1]
        for j in range(n-2, 0, -1): # n-2, ..., 1
            z[j] = (k[j] - h[j] * z[j+1])/m[j]

        # FINALLY, compute a, b, c, d coefficients
        a, b, c, d = {}, {}, {}, {}

        for j in range(n):
            a[j] = (z[j+1] - z[j]) / (6 * h[j])
            b[j] = z[j]/2
            c[j] = (data[j+1][1] - data[j][1])/h[j] - h[j]/6 * (z[j+1] + 2 * z[j])
            d[j] = data[j][1]

        self._a, self._b, self._c, self._d = a, b, c, d

    def __call__(self, x: float, /) -> float:
        # determine suitable piecewise spline S_j using the following heuristic:
        # find maximum j such that x <= x_(j+1).
        # if no such j (for whatever reason), then use j=n-1.
        if (x >= self._data[-2][0]):
            j=len(self._data) - 2
        else:
            j=0
            while (x > self._data[j+1][0]):
                j=j+1

        shifted_x = x - self._data[j][0]

        return self._a[j] * shifted_x**3 + self._b[j] * shifted_x**2 + self._c[j] * shifted_x + self._d[j]

# test data

x_data = [
    (0, 1.5),
    (0.618, 0.90),
    (0.935, 0.60),
    (1.255, 0.35),
    (1.636, 0.20),
    (1.905, 0.10),
    (2.317, 0.50),
    (2.827, 1.00),
    (3.330, 1.50)
] 

y_data = [
    (0, 0.75),
    (0.618, 0.90),
    (0.935, 1.00),
    (1.255, 0.80),
    (1.636, 0.45),
    (1.905, 0.20),
    (2.317, 0.10),
    (2.827, 0.20),
    (3.330, 0.25)
]

x_approx = NaturalSplineInterpolation(data=x_data)
y_approx = NaturalSplineInterpolation(data=y_data)

x_coeffs_table = {
    'j': [x for x in range(8)],
    'a_j': [*x_approx._a.values()],
    'b_j': [*x_approx._b.values()],
    'c_j': [*x_approx._c.values()],
    'd_j': [*x_approx._d.values()]
}

print(tab(x_coeffs_table, headers='keys'))

#   j         a_j         b_j        c_j    d_j
# ---  ----------  ----------  ---------  -----
#   0   0.0105369   0          -0.974898   1.5
#   1   0.102103    0.0195355  -0.962825   0.9
#   2   0.987167    0.116635   -0.919659   0.6
#   3  -1.77355     1.06432    -0.541755   0.35
#   4   5.39457    -0.962851   -0.503097   0.2
#   5  -3.39333     3.39057     0.149959   0.1
#   6   0.670643   -0.803594    1.21579    0.5
#   7  -0.147442    0.22249     0.919428   1

y_coeffs_table = {
    'j': [y for y in range(8)],
    'a_j': [*y_approx._a.values()],
    'b_j': [*y_approx._b.values()],
    'c_j': [*y_approx._c.values()],
    'd_j': [*y_approx._d.values()]
}

print(tab(y_coeffs_table, headers='keys'))

#   j        a_j          b_j         c_j    d_j
# ---  ---------  -----------  ----------  -----
#   0   0.276944   0            0.136947    0.75
#   1  -3.00101    0.513454     0.454261    0.9
#   2   2.43045   -2.34051     -0.124915    1
#   3  -0.273187  -0.00727664  -0.876207    0.8
#   4   2.1739    -0.31953     -1.00072     0.45
#   5  -0.784397   1.4348      -0.700711    0.2
#   6  -0.474226   0.465289     0.0821273   0.1
#   7   0.172483  -0.260277     0.186683    0.2

# graph the parametric function...

t = np.linspace(0, 3.33, 400)

x = np.array([
    x_approx(i)
    for i in t
])

y = np.array([
   y_approx(i)
   for i in t 
])

fig, ax = plt.subplots()

ax.plot(x, y)

plt.show()
