#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

from drawer import WindowDrawer, MP4Drawer
from move import TraceMove, Move
from trace import BasicTrace


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
    x0 = [2.025291, 0.28726044]

    r = 0.1
    l = 0.3
    a = 0.3
    interval = 20
    sec = 5
    wait = 100
    name = 'output/scene02'

    dt = interval * 0.001
    h = 0.5 * math.sqrt(3.0) * l
    dh = 0.5 * math.sqrt(3.0) * r
    dx = np.asarray([[-0.5 * l, -h], [0.5 * l, h], [1.5 * l, -h]],
                    dtype=np.float32)
    dx = dx + [0.5 * r, -dh]

    rect = [[0.0, 0.5 - r, 0.5 - r, 0.0, 0.0],
            [0.0, 0.0, 2.0, 2.0, 0.0]]
    x1 = [max(rect[0]) * 0.8, max(rect[1]) + r]
    xp = max(rect[0])
    dx0, dx1 = (x0[0] - xp) ** 2, (x1[0] - xp) ** 2
    A = - (x0[1] - x1[1]) / (dx0 - dx1)
    B = x0[1] + A * dx0
    vy = math.sqrt(2 * a * (B - x0[1]))
    vx = (xp - x0[0]) * a / vy
    bn = math.ceil(abs((x1[0] - x0[0]) / (vx * 0.05))) + 1
    xs = np.asarray(range(1, bn), dtype=np.float32) * 0.05 * vx + x0[0]
    ys = -A * (xs - xp) ** 2 + B
    xs[-1], ys[-1] = x1
    bcoords = np.asarray((xs, ys), dtype=np.float32).T

    v = -0.5
    target = [0.5 - r + l * 0.5, h + dh]
    nn = math.ceil((target[0] - x0[0]) / (v * dt)) + 1
    xs = np.asarray(range(nn), dtype=np.float32) * dt * v + x0[0]
    dy = (target[1] - x0[1]) / nn
    ys = np.asarray(range(nn), dtype=np.float32) * dy + x0[1]
    ncoords = np.zeros((bn + wait + nn, 2), dtype=np.float32) + x0
    xs[-1] = target[0]
    ncoords[bn + wait:, 0] = xs[:]
    ncoords[bn + wait:, 1] = ys[:]

    # drawer = WindowDrawer(xlim=[-0.1, 4.0], ylim=[-0.1, 3.0], dt=dt)
    drawer = MP4Drawer(filename=name, xlim=[-0.1, 4.0], ylim=[-0.1, 3.0],
                       dt=dt)

    btrace = BasicTrace(bcoords, repeat=False)
    bmove = TraceMove(btrace)
    ball = Circle(bmove, r)

    ntrace = BasicTrace(ncoords, repeat=False)
    nmove = TraceMove(ntrace)
    tri = Polynomics(nmove, dx)

    drawer.add_object(ball, color='r')
    drawer.add_object(tri, color='b')
    drawer.draw_stable_object(*rect, color='k')

    frames = bn + wait + nn + 10
    drawer.start(frames=frames, interval=interval)
