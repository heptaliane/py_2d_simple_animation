# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import numpy as np


class Move(object):
    def __init__(self, coord):
        self.coord = np.asarray(coord, dtype=np.float32)

    @abstractmethod
    def __call__(self, dt):
        raise NotImplementedError


class StableMove(Move):
    def __call__(self, dt):
        return self.coord


class StandardMove(Move):
    def __init__(self, coord, velocity, accel):
        super().__init__(coord)
        self.velocity = np.asarray(velocity, dtype=np.float32)
        self.accel = np.asarray(accel, dtype=np.float32)

    def __call__(self, dt):
        self.coord = self.coord + self.velocity * dt
        self.velocity = self.velocity + self.accel * dt
        return self.coord
