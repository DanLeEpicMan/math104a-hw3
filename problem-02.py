'''
Daniel Naylor
5094024
10/30/2022
'''

import math
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


def compute_divided_diff_array(data: List[Tuple[float, float]]) -> List[float]:
    '''
    Given a dataset, return a 1-dimensional array of Divided Difference coefficients.

    The returned array will be of the form:
    ```
    [
        f[x_0],
        f[x_0, x_1],
        ...
        f[x_0, x_1, ..., x_n]
    ]
    '''
    n_plus_1 = len(data)
    c = [] # the array to return. 
    # the name 'c' is used since c[0] corresponds to the first coefficient; easier to read
    for j in range(n_plus_1): # j = 0, ... n
        c.append(data[j][1])

    for k in range(1, n_plus_1): # k = 1, ..., n
        for j in range(n_plus_1-1, k-1, -1): # j = n, n-1, ..., k
            c[j] = (c[j] - c[j-1])/(data[j][0] - data[j-k][0])

    return c

test_data = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4)
]

print(compute_divided_diff_array(test_data))

# output: [1, 1.0, 0.0, 0.0]. correct since this is the polynomial x+1

# ------------------------------------------------------------------------

class poly_approx:
    '''
    Creates a polynomial approximation given a dataset.
    Acts like a mathematical function.

    ### Attributes
    `data`: The given data
    `coeffs`: The Divided Difference Coefficients, computed with `data`.
    '''
    def __init__(self, data: List[Tuple[float, float]]) -> None:
        '''
        Creates a polynomial given a dataset. The data should be of the form
        ```
        [
            (x_0, f_0),
            (x_1, f_1),
            ...
            (x_n, f_n)
        ]
        ```
        For example, `[(0, 1), (1, 2), (2, 3)]` for `x+1`
        would be an appropriate dataset.
        '''
        self.data = data
        self.coeffs = compute_divided_diff_array(data)

    def __call__(self, x: float, /) -> float:
        p = self.coeffs[-1]
        for j in range(len(self.data) - 1, -1, -1): # j = n, n-1, ..., 0
            p = self.coeffs[j] + p * (x - self.data[j][0])

        return p


def f(x: float) -> float:
    '''
    `f(x) = e^(-x^2)`
    '''
    return math.exp(-x**2)

p_10_data = [
    (j/5 - 1, f(j/5 - 1))
    for j in range(11) # j = 0, ..., 10
]

p_10 = poly_approx(p_10_data)

# graphing everything...

domain = np.linspace(-1, 1, 100)

error_graph = np.array([
    f(x) - p_10(x)
    for x in domain
])

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.plot(domain, error_graph, 'r', label='f(x) - P-10(x)')

plt.legend(loc='lower left')

plt.show()
