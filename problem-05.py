'''
Daniel Naylor
5094024
11/09/2022
'''

from typing import List, Tuple

# --------------------------------------------------
#                      Problem 5
# --------------------------------------------------

# natural spline: z_0 = z_n = 0

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
            print(j)
            z[j] = (k[j] - h[j] * z[j+1])/m[j]

        # FINALLY, compute a, b, c, d coefficients
        a, b, c, d = {}, {}, {}, {}

        print('z: ', z, 'h: ',  h)
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

data = [
    (0, 5),
    (1, 3),
    (3, 8),
    (6, -1)
] 

approx = NaturalSplineInterpolation(data=data)

print(
    f'1: {approx(1)}',
    f'2: {approx(2)}', 
    f'3: {approx(3)}',
    f'5: {approx(5)}',
    sep='\n'
)

# Output
# 1: 3.0
# 2: 5.125
# 3: 8.0
# 5: 4.0
