# -*- coding: utf-8 -*-

import numpy as np


class Domain(object):
    def __init__(self, xmax=np.inf, xmin=-np.inf):
        xmax = np.inf if xmax is None else xmax
        xmin = -np.inf if xmin is None else xmin
        self._range = np.asarray(sorted([xmax, xmin]), dtype=np.float32)

    def set_props(self, xmax=None, xmin=None):
        xmax = self._range[1] if xmax is None else xmax
        xmin = self._range[0] if xmin is None else xmin
        self._range = np.asarray(sorted([xmax, xmin]), dtype=np.float32)

    def __call__(self, x):
        return self._range[0] <= x <= self._range[1]

    def __contains__(self, x):
        '''
            self: -----@@@@@@@@@@--------------
            x   : ----------@------------------
        '''
        if isinstance(x, Domain):
            return self._range[0] < x._range[0] < x._range[1] < self._range[1]
        return self.__call__(x)

    def __eq__(self, other):
        '''
            self : -----@@@@@@@----------------
            other: -----@@@@@@@----------------
        '''
        return np.all(self._range == other._range)

    def __lt__(self, other):
        '''
            self : ------@@@@@@-----------------
            other: ----------------@@@@---------
        '''
        return self._range[1] < other._range[0]

    def __le__(self, other):
        '''
            self : -----@@@@@@@@@@--------------
            other: ----------@@@@@@@@@@@@-------
        '''
        return np.all(self._range <= other._range)

    def __gt__(self, other):
        '''
            self : ---------------@@@@@@@-------
            other: -----@@@@@@------------------
        '''
        return other._range[1] < self._range[0]

    def __ge__(self, other):
        '''
            self : ----------@@@@@@@@-----------
            other : -----@@@@@@@@---------------
        '''
        return np.all(self._range >= other._range)

    def __getitem__(self, idx):
        return self._range[idx]

    def __str__(self):
        return '%e <= x <= %e' % tuple(self._range)


class Line(object):
    def __init__(self, a1=1, a2=1, b=0, domain=None):
        a = np.asarray([a1, a2], dtype=np.float32)
        self._n = a / np.sqrt(a @ a)
        self._b = b / np.sqrt(a @ a)
        self.domain = Domain() if domain is None else domain

    def __call__(self, x):
        if not self.domain(x):
            return np.nan
        return -(self._n[0] * x + self._b) / self._n[1]

    def __eq__(self, other):
        return np.all(self._n == other._n) and self._b == other._b

    def func(self, x, y, ignore_domain=False):
        if not self.domain(x) and not ignore_domain:
            return np.nan
        return self._n @ [x, y] + self._b

    def get_distance(self, x, y):
        f = self.func(x, y, ignore_domain=True)
        return np.abs(f) / np.sqrt(self._n @ self._n)

    def get_symmetry_point(self, x, margin=0.0):
        t = - (self._n @ x + self._b) / (self._n @ self._n)
        t = t - margin if t > 0 else t + margin
        return x + 2 * t * self._n

    def get_reflect_vector(self, v):
        n = self._n
        A = np.asarray((n * [1, -1], n[::-1]), dtype=np.float32)
        Ainv = np.asarray((n, n[::-1] * [-1, 1]), dtype=np.float32)
        v = (A @ v) * [-1, 1]
        return Ainv @ v

    def is_clossed(self, line):
        if np.all(self._n == line._n):
            return self._b != line._b
        if self.domain < line.domain or self.domain > line.domain:
            return False

        x1 = max(self.domain[0], line.domain[0])
        x2 = min(self.domain[1], line.domain[1])

        y1 = line(x1)
        y2 = line(x2)

        v1 = self.func(x1, y1)
        v2 = self.func(x2, y2)
        return v1 * v2 <= 0

    def __str__(self):
        return '%4.3ex + %4.3ey + %4.3e = 0' % (*self._n, self._b)


def get_line_from_coords(coord1, coord2):
    a1 = coord1[1] - coord2[1]
    a2 = -(coord1[0] - coord2[0])
    b = coord1[0] * coord2[1] - coord1[1] * coord2[0]
    domain = Domain(coord1[0], coord2[0])

    if a2 < 0:
        return Line(a1=-a1, a2=-a2, b=-b, domain=domain)
    else:
        return Line(a1=a1, a2=a2, b=b, domain=domain)


def get_line_from_gradient(gradient, coord, domain=None):
    coord = np.asarray(coord, dtype=np.float32)
    a1 = 0.0 if gradient is np.inf else 1.0
    a2 = 1.0 if gradient is np.inf else gradient
    b = -([a1, a2] @ coord)
    return Line(a1=a1, a2=a2, b=b, domain=domain)


class Region(object):
    def __init__(self, apexes):
        self.lines = list()
        self.apexes = apexes
        for i in range(len(apexes)):
            c1 = apexes[i]
            c2 = apexes[(i + 1) % len(apexes)]
            self.lines.append(get_line_from_coords(c1, c2))

    def __contains__(self, coord):
        is_contain = False
        for line in filter(lambda l: l.domain(coord[0]), self.lines):
            if line(coord[0]) >= coord[1]:
                is_contain = not is_contain
        return is_contain

    def get_crossed_line(self, coord1, coord2):
        trace = get_line_from_coords(coord1, coord2)
        closest_distance = np.inf
        clossed = None

        for line in filter(lambda l: l.is_clossed(trace), self.lines):
            distance = line.get_distance(*coord1)
            if distance < closest_distance:
                closest_distance = distance
                clossed = line
        return clossed

    def get_closest_line(self, coord):
        closest_distance = np.inf
        closest = None

        for line in self.lines:
            distance = line.get_distance(*coord)
            if distance < closest_distance:
                closest_distance = distance
                closest = line
        return (closest, closest_distance)
