#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

from drawer import WindowDrawer, MP4Drawer
from region import Region, Domain, get_line_from_coords
from move import TraceMove, Move
from trace import ParabolaTrace, BasicTrace
from trigger import ToggleTrigger


class ScenePalabolaTrace(BasicTrace):
    def __init__(self, coord, velocity, directions, height, base=0.0,
                 margin=0.1, wait=20, **kwargs):
        self._directions = directions
        self._speed = velocity
        self._height = height
        self._base = base
        self._margin = margin
        self._kwargs = kwargs
        self._dir_idx = 0
        self._idx = 0
        self.wait = wait
        self._update_trace(coord)

    def _update_trace(self, coord):
        t = self._directions[self._dir_idx]
        velocity = np.asarray((np.cos(t), np.sin(t)), dtype=np.float32)
        velocity = velocity * self._speed
        parabola = ParabolaTrace(coord, velocity, height=self._height,
                                 **self._kwargs)

        x = np.asarray([parabola() for i in range(1000)])
        mind = np.inf
        target = -1
        n_reflect = 0
        for i in range(2, len(x)):
            if x[i][1] < self._height and x[i - 1][1] > self._height:
                d = abs(x[i][0] - coord[0])
                if d <= mind:
                    target = i
                    mind = d
            elif x[i][1] > self._height and x[i - 1][1] < self._height:
                d = abs(x[i - 1][0] - coord[0])
                if d <= mind:
                    target = i - 1
                    mind = d

            if (x[i][0] - x[i - 1][0]) * (x[i - 1][0] - x[i - 2][0]) < 0:
                n_reflect += 1
            if n_reflect >= 2:
                break
        self.coords = x[:target + 1]
        self._dir_idx = (self._dir_idx + 1) % len(self._directions)

    def __call__(self):
        if self._idx < len(self.coords):
            coord = self.coords[self._idx]
        else:
            coord = self.coords[len(self.coords) - 1]
        self._idx += 1

        if self._idx >= self.wait + len(self.coords):
            self._update_trace(coord)
            self._idx = 0
        return coord + [0.0, self._base]


class ChaseParabolaTrace(BasicTrace):
    def __init__(self, parabola):
        self._parabola = parabola
        self._idx = 0
        self._update_trace()

    def _update_trace(self):
        pc = self._parabola.coords
        start = 0
        for i in range(len(pc) - 1, 1, -1):
            if pc[i][1] < pc[i - 1][1] and pc[i - 1][1] > pc[i - 2][1]:
                start = i
                break
        self.coords = np.asarray([pc[0]] * len(pc), dtype=np.float32)
        dx = (pc[len(pc) - 1][0] - pc[0][0]) / (len(pc) - start)
        nx = np.asarray(range(len(pc) - start), dtype=np.float32) * dx
        nx = nx + pc[0][0]
        self.coords[start:len(pc), 0] = nx[:]

    def __call__(self):
        if self._idx < len(self.coords):
            coord = self.coords[self._idx]
        else:
            coord = self.coords[len(self.coords) - 1]
        self._idx += 1

        if self._idx >= self._parabola.wait + len(self.coords):
            self._update_trace()
            self._idx = 0
        return coord


class Circle(object):
    def __init__(self, move, r, samples=100):
        self._move = move
        self.r = r
        self.samples = samples

    def __call__(self, dt):
        coord = self._move(dt)
        t = np.asarray(range(self.samples), dtype=np.float32)
        t = t * 2 * math.pi / (self.samples - 1)
        xs = np.cos(t) * self.r + coord[0]
        ys = np.sin(t) * self.r + coord[1]
        return (xs, ys)


class Polynomics(object):
    def __init__(self, move, dx):
        self._move = move
        self._dx = np.asarray([*dx, dx[0]], dtype=np.float32)

    def __call__(self, dt):
        coord = self._move(dt)

        xs = self._dx[:, 0] + coord[0]
        ys = self._dx[:, 1] + coord[1]
        return (xs, ys)


if __name__ == '__main__':
    dt = 0.02
    v = 1.0
    d = (np.asarray([0.1, 0.15, 0.2, 0.3, 0.35]) + 0.5) * math.pi
    r = 0.1
    l = 0.3
    x0 = 3.0
    lim = [0.5, 4.0]

    h = 0.5 * math.sqrt(3.0) * l
    dh = 0.5 * math.sqrt(3.0) * r
    dx = np.asarray([[-0.5 * l, -h], [0.5 * l, h], [1.5 * l, -h]],
                    dtype=np.float32)
    dx = dx + [0.5 * r, -dh]

    drawer = WindowDrawer(xlim=[0.0, 4.0], ylim=[-0.1, 3.0], dt=dt)

    btrace = ScenePalabolaTrace([x0, h], v, d, h, dh, lim=lim)
    ntrace = ChaseParabolaTrace(btrace)
    bmove = TraceMove(btrace)
    nmove = TraceMove(ntrace)
    ball = Circle(bmove, r)
    tri = Polynomics(nmove, dx)

    drawer.add_object(ball)
    drawer.add_object(tri, color='b')
    drawer.start()
