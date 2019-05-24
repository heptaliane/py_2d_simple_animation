#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

from drawer import WindowDrawer, MP4Drawer
from move import TraceMove
from trace import BasicTrace, ParabolaTrace


class Polynomics(object):
    def __init__(self, move, dx):
        self._move = move
        self._dx = np.asarray([*dx, dx[0]], dtype=np.float32)

    def __call__(self, dt):
        coord = self._move(dt)

        xs = self._dx[:, 0] + coord[0]
        ys = self._dx[:, 1] + coord[1]
        return (xs, ys)


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


if __name__ == '__main__':
    r = 0.1
    l = 0.3
    a = 0.3
    v = 0.6
    wait = 20
    interval = 20
    name = 'output/scene08'

    dt = interval * 0.001
    h = 0.5 * math.sqrt(3.0) * l
    dx = np.asarray([[-0.5 * l, -h], [0.5 * l, h], [1.5 * l, -h]],
                    dtype=np.float32)
    dx2 = np.asarray([-r * 0.5, r * math.sqrt(3.0) * 0.5], dtype=np.float32)
    rect = [[0.0, 0.5 - r, 0.5 - r, 0.0, 0.0],
            [0.0, 0.0, 2.0, 2.0, 0.0]]
    x0 = [5.0, h]
    x1 = [3.0, h]

    coords = np.ones((wait * 2, 2), dtype=np.float32) * x0

    n = math.ceil((x1[0] - x0[0]) / (-v * 0.05)) + 1
    xs = np.asarray(range(n), dtype=np.float32) * -v * 0.05 + x0[0]
    ys = np.ones((n), dtype=np.float32) * x0[1]
    ncoords = np.asarray((xs, ys), dtype=np.float32)
    bcoords = ncoords.T + dx2

    # drawer = WindowDrawer(xlim=[-0.1, 4.0], ylim=[-0.1, 3.0], dt=dt)
    drawer = MP4Drawer(filename=name, xlim=[-0.1, 4.0], ylim=[-0.1, 3.0],
                       dt=dt)

    btrace = BasicTrace(np.concatenate([coords, bcoords]), repeat=False)
    ntrace = BasicTrace(np.concatenate([coords, ncoords.T]), repeat=False)
    bmove = TraceMove(btrace)
    nmove = TraceMove(ntrace)
    ball = Circle(bmove, r)
    tri = Polynomics(nmove, dx)

    drawer.add_object(ball, color='r')
    drawer.add_object(tri, color='b')
    drawer.draw_stable_object(*rect, color='k')
    drawer.start(frames=ncoords.shape[1] + wait * 3, interval=interval)
