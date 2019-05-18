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
        self.r = r
        self._prev = self._region.get_closest_line(self._move.get_coord())
        self._trigger = list()

    def _check_collision(self, coord):
        line1, r1 = self._region.get_closest_line(coord)
        line2, r2 = self._prev

        if r1 <= self.r < r2:
            return line1

        self._prev = (line1, r1)
        return None

    def __call__(self, dt):
        coord = self._move(dt)
        line = self._check_collision(coord)
        if line is not None:
            coord = line.get_symmetry_point(coord, margin=self.r)
            self._move.set_coord(coord)
            velocity = line.get_reflect_vector(self._move.get_velocity())
            self._move.set_velocity(velocity)
            for trig in filter(lambda t: t is not None, self._trigger):
                trig.call(coord=coord, velocity=velocity)

        t = np.asarray(range(self.samples)) * 2 * math.pi / (self.samples - 1)
        xs = np.cos(t) * self.r + coord[0]
        ys = np.sin(t) * self.r + coord[1]
        return (xs, ys)

    def add_collision_trigger(self, trigger):
        self._trigger.append(trigger)
        return len(self._trigger) - 1

    def disable_collision_trigger(self, idx):
        self._trigger[idx] = None

    def get_coord(self):
        return self._move.get_coord()


class Trace(Object):
    def __init__(self, obj, idx=0):
        if isinstance(obj._move, list):
            self._move = obj._move[idx]
        else:
            self._move = obj._move
        self._trace = list()

    def __call__(self, dt):
        self._trace.append(self._move.get_coord())
        return np.asarray(self._trace, dtype=np.float32).T


class Particle(Object):
    def __call__(self, dt):
        self._move(dt)
        return self._move.coord


class Nodes(Object):
    def __init__(self, move, region, apexes):
        super().__init__(move, region)
        apexes = np.asarray([*apexes, *apexes[0:2]], dtype=np.float32)
        self._nodes = list()
        for i in range(len(apexes) - 2):
            c1, c2, c3 = apexes[i:i + 3]
            v1, v2 = c1 - c2, c2 - c3
            l1, l2 = np.sqrt(v1 @ v1), np.sqrt(v2 @ v2)
            v1, v2 = v1 / l1, v2 / l2
            t1, t2 = math.atan2(*v1[::-1]), math.atan2(*v2[::-1])
            self._nodes.append((t2 - t1, l2))
        self._move = [move.copy() for i in range(len(apexes) - 3)]
        self._move = [move, *self._move]
        for i, move in enumerate(self._move):
            move.set_coord(apexes[i])

    def on_collision(self, idx, dt):
        n = len(self._move)
        cs = [move.get_coord() for move in self._move]

        c1 = self._move[idx].get_coord()
        v1 = self._move[idx].get_velocity()
        line, _ = self._region.get_closest_line(c1)
        cp1 = line.get_symmetry_point(c1)
        vp1 = line.get_reflect_vector(v1)
        self._move[idx].set_coord(cp1)
        self._move[idx].set_velocity(vp1)

        c2 = self._move[(idx + 1) % n].get_coord()
        v2 = self._move[(idx + 1) % n].get_velocity()
        s1 = -v2 @ (c2 - cp1)
        s2 = np.sqrt((v2 @ (c2 - cp1)) ** 2 - (v2 @ v2) *
                     ((c2 - cp1) @ (c2 - cp1) - (c2 - c1) @ (c2 - c1)))
        s = s1 + s2 if abs(s1 + s2) < abs(s1 - s2) else s1 - s2
        cp2 = c2 + v2 * s / (v2 @ v2)
        vp2 = (cp2 - c2) / dt + v2
        tvp2 = math.atan2(*vp2[::-1])
        vp2 = np.asarray([np.cos(tvp2), np.sin(tvp2)], dtype=np.float32)
        print('(%.3f, %.3f) -> (%.3f, %.3f)' % (*v2, *vp2))
        self._move[(idx + 1) % n].set_coord(cp2)
        self._move[(idx + 1) % n].set_velocity(vp2)

        for i in range(2, n):
            t0 = math.atan2(*(cp2 - cp1)[::-1])
            t3, l3 = self._nodes[(idx + i - 1) % n]
            d3 = np.asarray([math.cos(t0 + t3), math.sin(t0 + t3)],
                            dtype=np.float32) * l3
            c3 = self._move[(idx + i) % n].get_coord()
            v3 = self._move[(idx + i) % n].get_velocity()
            cp3 = cp2 + d3
            vp3 = (c3 - cp3) / dt + v3
            tvp3 = math.atan2(*vp3[::-1])
            vp3 = np.asarray([np.cos(tvp3), np.sin(tvp3)], dtype=np.float32)
            print('(%.3f, %.3f) -> (%.3f, %.3f)' % (*v3, *vp3))
            self._move[(idx + i) % n].set_coord(cp3)
            self._move[(idx + i) % n].set_velocity(vp3)
            cp1 = cp2
            cp2 = cp3
        cs = [move.get_coord() for move in self._move]

    def __call__(self, dt):
        coords = [move(dt) for move in self._move]
        for i, coord in enumerate(coords):
            if coord not in self._region:
                self.on_collision(i, dt)
        coords = [move.get_coord() for move in self._move]
        return np.asarray([*coords, coords[0]], dtype=np.float32).T

    def get_coord(self):
        return [move.get_coord() for move in self._move]


class Area(Object):
    def __init__(self, move, region):
        super().__init__(move, region)
        apexes = [*self._region.apexes, self._region.apexes[0]]
        self.apexes = np.asarray(apexes, dtype=np.float32).T

    def __call__(self, dt):
        return self.apexes
