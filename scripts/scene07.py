#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

from drawer import WindowDrawer, MP4Drawer
from move import TraceMove, Move
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


if __name__ == '__main__':
    r = 0.1
    l = 0.3
    a = 0.3
    v = 0.4
    wait = 20
    interval = 20
    name = 'output/scene07'

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
    nx0 = [0.5 * (1 + l - r), 4 * h]
    xx0 = [0.5 * (1 + l - r) + w, h]
    box = np.asarray([*dx2, dx2[0]]) + xx0

    nx1 = [0.5 * (1 + l - r) + w, 4 * h]
    n = math.ceil((nx1[0] - nx0[0]) / (v * 0.05)) + 1
    xs = np.asarray(range(n), dtype=np.float32) * (v * 0.05) + nx0[0]
    ys = np.ones((n), dtype=np.float32) * nx0[1]
    ncoords1 = np.asarray((xs, ys), dtype=np.float32)

    ny2 = h
    t = math.sqrt(2 * (nx1[1] - ny2) / a)
    n = math.ceil(t / 0.05) + 1
    ts = np.asarray(range(n), dtype=np.float32) * 0.05
    xs = v * ts + nx1[0]
    ys = -0.5 * a * ts ** 2 + nx1[1]
    ys[-1] = ny2
    ncoords2 = np.asarray((xs, ys), dtype=np.float32)
    nx2 = [xs[-1], ys[-1]]

    ncoords3 = np.ones((wait, 2), dtype=np.float32) * nx2

    nx3 = [0.5 * (1 + l - r) + w, h]
    n = math.ceil((nx3[0] - nx2[0]) / (-v * 0.05)) + 1
    xs = np.asarray(range(n), dtype=np.float32) * (-v * 0.05) + nx2[0]
    ys = np.ones((n), dtype=np.float32) * nx2[1]
    ncoords4 = np.asarray((xs, ys), dtype=np.float32)

    ncoords5 = np.ones((wait, 2), dtype=np.float32) * nx3

    nx4 = [5.2, h]
    n = math.ceil((nx4[0] - nx3[0]) / (v * 0.05 * 0.5)) + 1
    xs = np.asarray(range(n), dtype=np.float32) * (v * 0.05 * 0.5) + nx3[0]
    ys = np.ones((n), dtype=np.float32) * nx3[1]
    ncoords6 = np.asarray((xs, ys), dtype=np.float32)

    ncoords = np.concatenate([ncoords1.T, ncoords2.T, ncoords3,
                              ncoords4.T, ncoords5, ncoords6.T])

    bcoords1 = np.ones((ncoords.shape[0] - ncoords6.shape[1], 2),
                       dtype=np.float32) * xx0
    bcoords = np.concatenate([bcoords1, ncoords6.T])

    # drawer = WindowDrawer(xlim=[-0.1, 4.0], ylim=[-0.1, 3.0], dt=dt)
    drawer = MP4Drawer(filename=name, xlim=[-0.1, 4.0], ylim=[-0.1, 3.0],
                       dt=dt)

    ntrace = BasicTrace(ncoords, repeat=False)
    nmove = TraceMove(ntrace)
    tri = Polynomics(nmove, dx)

    btrace = BasicTrace(bcoords, repeat=False)
    bmove = TraceMove(btrace)
    box = Polynomics(bmove, dx2)

    drawer.add_object(tri, color='b')
    drawer.add_object(box, color='g')
    drawer.draw_stable_object(*rect, color='k')

    drawer.start(frames=ncoords.shape[0] + wait, interval=interval)
