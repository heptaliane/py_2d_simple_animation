#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

from drawer import WindowDrawer, MP4Drawer
from move import TraceMove, Move
from trace import BasicTrace


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
    interval = 20
    wait = 40
    samples = 100
    name = 'output/scene03'

    dt = interval * 0.001
    h = 0.5 * math.sqrt(3.0) * l
    dx = np.asarray([[-0.5 * l, -h], [0.5 * l, h], [1.5 * l, -h]],
                    dtype=np.float32)
    nx0 = [0.5 - 0.5 * r + l * 0.5, h]
    rect = [[0.0, 0.5 - r, 0.5 - r, 0.0, 0.0],
            [0.0, 0.0, 2.0, 2.0, 0.0]]
    bx0 = [max(rect[0]) * 0.8, max(rect[1]) + r]

    # drawer = WindowDrawer(xlim=[-0.1, 4.0], ylim=[-0.1, 3.0], dt=dt)
    drawer = MP4Drawer(filename=name, xlim=[-0.1, 4.0], ylim=[-0.1, 3.0],
                       dt=dt)

    n = math.ceil((v * 2 / a) / 0.05) + 1
    xs = np.ones((n), dtype=np.float32) * nx0[0]
    ts = np.asarray(range(n), dtype=np.float32) * 0.05
    ys = -0.5 * a * ts ** 2 + v * ts + nx0[1]
    ncoords = np.ones((wait + len(xs), 2), dtype=np.float32) * nx0
    ncoords[wait:, 0] = xs
    ncoords[wait:, 1] = ys

    ntrace = BasicTrace(ncoords)
    nmove = TraceMove(ntrace)
    tri = Polynomics(nmove, dx)

    ts = np.asarray(range(samples), dtype=np.float32) / (samples - 1)
    ts = ts * (2.0 * math.pi)
    xs = np.cos(ts) * r + bx0[0]
    ys = np.sin(ts) * r + bx0[1]

    drawer.add_object(tri, color='b')
    drawer.draw_stable_object(xs, ys, color='r')
    drawer.draw_stable_object(*rect, color='k')

    frames = ncoords.shape[0] * 3 + wait
    drawer.start(frames=frames, interval=interval)
