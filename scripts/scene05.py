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
    v = 0.4
    wait = 30
    samples = 100
    interval = 20
    name = 'output/scene05'

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
    nx0 = [xx0[0] + l, h]
    box = np.asarray([*dx2, dx2[0]]) + xx0

    nx1 = [0.5 * (1 + l - r) + l * 2, 4 * h]
    nxp = xx0[0] + np.max(dx2.T[0])
    ndx0, ndx1 = (nx0[0] - nxp) ** 2, (nx1[0] - nxp) ** 2
    A = - (nx0[1] - nx1[1]) / (ndx0 - ndx1)
    B = nx0[1] + A * ndx0
    vy = math.sqrt(2 * a * (B - nx0[1]))
    vx = (nxp - nx0[0]) * a / vy
    bn = math.ceil(abs((nx1[0] - nx0[0]) / (vx * 0.05))) + 1
    xs = np.asarray(range(1, bn), dtype=np.float32) * 0.05 * vx + nx0[0]
    ys = -A * (xs - nxp) ** 2 + B
    xs[-1], ys[-1] = nx1
    ncoords1 = np.asarray((xs, ys), dtype=np.float32).T
    ncoords2 = np.ones((wait, 2), dtype=np.float32) * nx1
    nx2 = [0.5 * (1 + l - r), 4 * h]
    n = math.ceil((nx1[0] - nx2[0]) / (v * 0.5 * 0.05)) + 1
    xs = np.asarray(range(n), dtype=np.float32) * (-v * 0.05 * 0.5) + nx1[0]
    ys = np.ones((n), dtype=np.float32) * nx2[1]
    ncoords3 = np.asarray((xs, ys), dtype=np.float32).T
    ncoords = np.concatenate([ncoords1, ncoords2, ncoords3])

    # drawer = WindowDrawer(xlim=[-0.1, 4.0], ylim=[-0.1, 3.0], dt=dt)
    drawer = MP4Drawer(filename=name, xlim=[-0.1, 4.0], ylim=[-0.1, 3.0],
                       dt=dt)

    ntrace = BasicTrace(ncoords, repeat=False)
    nmove = TraceMove(ntrace)
    tri = Polynomics(nmove, dx)

    ts = np.asarray(range(samples), dtype=np.float32)/ (samples - 1)
    ts = ts * (2.0 * math.pi)
    circle = np.asarray((np.cos(ts), np.sin(ts)), dtype=np.float32).T * r + bx0

    drawer.add_object(tri, color='b')
    drawer.draw_stable_object(*box.T, color='g')
    drawer.draw_stable_object(*rect, color='k')
    drawer.draw_stable_object(*circle.T, color='r')

    drawer.start(frames=ncoords.shape[0] + wait, interval=interval)
