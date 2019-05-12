# -*- coding: utf-8 -*-

import math
from abc import ABCMeta, abstractmethod
import numpy as np


class Object(object):
    def __init__(self, move, region, samples=100):
        self._move = move
        self._region = region
        self.samples = samples

    @abstractmethod
    def __call__(self, dt):
        raise NotImplementedError


class Ball(Object):
    def __init__(self, move, region, r, **kwargs):
        super().__init__(move, region, **kwargs)
        print('r = %f' % r)
        self.r = r
        self._prev = self._region.get_closest_line(self._move.coord)

    def _check_collision(self):
        line1, r1 = self._region.get_closest_line(self._move.coord)
        line2, r2 = self._prev

        if r1 <= self.r < r2:
            return line1

        self._prev = (line1, r1)
        return None

    def __call__(self, dt):
        self._move(dt)
        line = self._check_collision()
        if line is not None:
            self._move.coord = line.get_symmetry_point(self._move.coord,
                                                       margin=self.r)
            self._move.velocity = line.get_reflect_vector(self._move.velocity)

        t = np.asarray(range(self.samples)) * 2 * math.pi / (self.samples - 1)
        xs = np.cos(t) * self.r + self._move.coord[0]
        ys = np.sin(t) * self.r + self._move.coord[1]
        return (xs, ys)


class Trace(Object):
    def __init__(self, obj):
        self._obj = obj
        self._trace = list()

    def __call__(self, dt):
        self._trace.append(self._obj._move.coord)
        return np.asarray(self._trace, dtype=np.float32).T


class Area(Object):
    def __init__(self, move, region):
        super().__init__(move, region)
        apexes = [*self._region.apexes, self._region.apexes[0]]
        self.apexes = np.asarray(apexes, dtype=np.float32).T

    def __call__(self, dt):
        return self.apexes
