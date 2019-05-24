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
    v = 0.5
    wait = 30
    interval = 20
    samples = 100
    name = 'output/scene04'

    dt = interval * 0.001
    h = 0.5 * math.sqrt(3.0) * l
    dx = np.asarray([[-0.5 * l, -h], [0.5 * l, h], [1.5 * l, -h]],
                    dtype=np.float32)
    w = l * 3.0
    dx2 = np.asarray([[-0.5 * l, -h], [-0.5 * l - w, -h],
                      [-0.5 * l - w, 2 * h], [-0.5 * l, 2 * h]], dtype=np.float32)
    rect = [[0.0, 0.5 - r, 0.5 - r, 0.0, 0.0],
            [0.0, 0.0, 2.0, 2.0, 0.0]]
    nx0 = [0.5 - 0.5 * r + l * 0.5, h]
    bx0 = [max(rect[0]) * 0.8, max(rect[1]) + r]

    # drawer = WindowDrawer(xlim=[-0.1, 4.0], ylim=[-0.1, 3.0], dt=dt)
    drawer = MP4Drawer(filename=name, xlim=[-0.1, 4.0], ylim=[-0.1, 3.0],
                       dt=dt)

    nx1 = [5.5, h]
    n = math.ceil((nx1[0] - nx0[0]) / (v * 0.05)) + 1
    xs = np.asarray(range(n), dtype=np.float32) * (v * 0.05) + nx0[0]
    ys = np.ones((n), dtype=np.float32) * nx0[1]
    ncoords0 = np.ones((wait, 2), dtype=np.float32) * nx0
    ncoords1 = np.asarray((xs, ys), dtype=np.float32)
    ncoords2 = np.ones((wait, 2), dtype=np.float32) * nx1

    nx2 = [nx0[0] + w, nx0[1]]
    n = math.ceil((nx1[0] - nx2[0]) / (v * 0.5 * 0.05)) + 1
    xs = np.asarray(range(n), dtype=np.float32) * (-v * 0.05 * 0.5) + nx1[0]
    ys = np.ones((n), dtype=np.float32) * nx0[1]
    ncoords3 = np.asarray((xs, ys), dtype=np.float32)
    ncoords4 = np.ones((wait, 2), dtype=np.float32) * nx2
    nx3 = [nx2[0] + l, nx2[1]]
    n = math.ceil((nx3[0] - nx2[0]) / (v * 0.5 * 0.05)) + 1
    xs = np.asarray(range(n), dtype=np.float32) * (v * 0.05 * 0.5) + nx2[0]
    ys = np.ones((n), dtype=np.float32) * nx0[1]
    ncoords5 = np.asarray((xs, ys), dtype=np.float32)
    ncoords = np.concatenate([ncoords0, ncoords1.T, ncoords2,
                              ncoords3.T, ncoords4, ncoords5.T])
    bcoords = np.ones((ncoords1.shape[1] + wait * 2, 2),
                      dtype=np.float32) * nx1
    bcoords = np.concatenate([bcoords, ncoords3.T])

    ntrace = BasicTrace(ncoords, repeat=False)
    nmove = TraceMove(ntrace)
    tri = Polynomics(nmove, dx)
    btrace = BasicTrace(bcoords, repeat=False)
    bmove = TraceMove(btrace)
    box = Polynomics(bmove, dx2)

    ts = np.asarray(range(samples), dtype=np.float32) / (samples - 1)
    ts = ts * (2.0 * math.pi)
    xs = np.cos(ts) * r + bx0[0]
    ys = np.sin(ts) * r + bx0[1]

    drawer.add_object(tri, color='b')
    drawer.add_object(box, color='g')
    drawer.draw_stable_object(xs, ys, color='r')
    drawer.draw_stable_object(*rect, color='k')

    drawer.start(frames=ncoords.shape[0] + wait, interval=interval)
