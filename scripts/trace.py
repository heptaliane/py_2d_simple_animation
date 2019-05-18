# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt


class BasicTrace(object):
    def __init__(self, coords):
        self.coords = np.asarray(coords, dtype=np.float32)
        self._idx = 0

    def __call__(self):
        coord = self.coords[self._idx]
        self._idx = (self._idx + 1) % len(self.coords)
        return coord


class ParabolaTrace(BasicTrace):
    def __init__(self, coord, velocity, height=0.0,
                 grabity=0.3, elasticity=0.9, dt=0.05, lim=[0.0, 1.0]):
        self._init_coord = np.asarray(coord, dtype=np.float32)
        self._velocity = np.asarray(velocity, dtype=np.float32)
        self._grabity = grabity
        self._elasticity = elasticity
        self._range = np.asarray(lim, dtype=np.float32)
        self.dt = dt
        self._idx = 0
        self._compute_parabola(height=height)

    def _compute_parabola(self, height=0.0):
        x0 = self._init_coord
        v = self._velocity
        a = self._grabity
        dt = self.dt
        xp = np.asarray([v[0] * v[1] / a + x0[0],
                         0.5 * v[1] ** 2 / a + x0[1]], dtype=np.float32)
        if height == 0.0:
            x1 = np.asarray([2 * v[0] * v[1] / a + x0[0], x0[1]],
                            dtype=np.float32)
        else:
            t = (v[1] + np.sqrt(v[1] ** 2 + 2 * a * height)) / a
            x1 = np.asarray([v[0] * t + x0[0], x0[1] - height],
                            dtype=np.float32)

        if (xp[0] - x0[0]) ** 2 == 0.0:
            self.coords = np.asarray(([x0[0]], [x0[1]]), dtype=np.float32)
            return

        p1 = (xp[1] - x0[1]) / ((xp[0] - x0[0]) ** 2)
        p2 = xp[0] - x0[0]
        n = math.ceil(abs((x1[0] - x0[0]) / (v[0] * dt))) + 1
        xs = np.asarray(range(1, n), dtype=np.float32) * dt * v[0] + x0[0]
        ys = -p1 * ((xs - x0[0]) ** 2) + 2 * p1 * p2 * (xs - x0[0]) + x0[1]
        xs[-1], ys[-1] = (x1[0], x1[1])

        for i in range(len(xs)):
            if xs[i] < self._range[0]:
                xs[i:] = self._range[0] * 2 - xs[i:]
                v[0] = -v[0] * self._elasticity
            elif xs[i] > self._range[1]:
                xs[i:] = self._range[1] * 2 - xs[i:]
                v[0] = -v[0] * self._elasticity
        v[1] = np.sqrt(v[1] ** 2 + 2 * a * height)

        self._velocity = v
        self.coords = np.asarray((xs, ys), dtype=np.float32)

    def __call__(self):
        coord = self.coords[:, self._idx]
        self._idx += 1
        if self._idx >= self.coords.shape[1]:
            self._init_coord = coord
            self._velocity[1] = self._velocity[1] * self._elasticity
            self._compute_parabola()
            self._idx = 0
        return coord

    def copy(self):
        trace = ParabolaTrace(self._init_coord.copy(), self._velocity.copy(),
                              self._grabity, self._elasticity, self.dt,
                              self.lim)
        trace.coords = self.coords.copy()
        trace._idx = self._idx
        return trace
