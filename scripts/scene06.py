#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

from drawer import WindowDrawer, MP4Drawer
from move import TraceMove, Move
from trace import BasicTrace, ParabolaTrace


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
    r = 0.1
    l = 0.3
    a = 0.3
    v = 0.8
    wait = 40
    samples = 100
    interval = 20
    name = 'output/scene06'

    dt = interval * 0.001
    h = 0.5 * math.sqrt(3.0) * l
    dx = np.asarray([[-0.5 * l, -h], [0.5 * l, h], [1.5 * l, -h]],
                    dtype=np.float32)
    w = l * 3.0
    dx2 = np.asarray([[-0.5 * l, -h], [-0.5 * l - w, -h],
                      [-0.5 * l - w, 2 * h], [-0.5 * l, 2 * h]],
                      dtype=np.float32)
    rect = [[0.0, 0.5 - r, 0.5 - r, 0.0, 0.0],
            [0.0, 0.0, 2.0, 2.0, 0.0]]
    bx0 = [max(rect[0]) * 0.8, max(rect[1]) + r]
    xx0 = [0.5 * (1 + l - r) + w, h]
    nx0 = [0.5 * (1 + l - r), 4 * h]
    box = np.asarray([*dx2, dx2[0]]) + xx0

    # drawer = WindowDrawer(xlim=[-0.1, 4.0], ylim=[-0.1, 3.0], dt=dt)
    drawer = MP4Drawer(filename=name, xlim=[-0.1, 4.0], ylim=[-0.1, 3.0],
                       dt=dt)

    n = math.ceil((v * 2 / a) / 0.05) + 1
    xs = np.ones((n), dtype=np.float32) * nx0[0]
    ts = np.asarray(range(n), dtype=np.float32) * 0.05
    ys = -0.5 * a * ts ** 2 + v * ts + nx0[1]
    ncoords0 = np.ones((wait, 2), dtype=np.float32) * nx0
    ncoords1 = np.asarray((xs, ys), dtype=np.float32)
    ncoords = np.concatenate([ncoords0, ncoords1.T])

    bcoords0 = np.ones((ncoords1.shape[1] // 2 + 25, 2),
                       dtype=np.float32) * bx0
    p = ParabolaTrace(bx0, [0.4, 0.6], height=max(rect[1]), lim=[0.0, np.inf])
    bcoords1 = np.asarray([p() for i in range(500)], dtype=np.float32)
    bcoords1 = np.asarray(list(filter(lambda x: x[0] < 5.0, bcoords1)),
                          dtype=np.float32)
    bcoords = np.concatenate([bcoords0, bcoords1])

    ntrace = BasicTrace(ncoords, repeat=False)
    nmove = TraceMove(ntrace)
    tri = Polynomics(nmove, dx)

    btrace = BasicTrace(bcoords, repeat=False)
    bmove = TraceMove(btrace)
    ball = Circle(bmove, r)

    drawer.add_object(tri, color='b')
    drawer.add_object(ball, color='r')
    drawer.draw_stable_object(*box.T, color='g')
    drawer.draw_stable_object(*rect, color='k')

    drawer.start(frames=bcoords.shape[0], interval=interval)
