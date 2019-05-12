# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import numpy as np


class Move(object):
    def __init__(self, coord=[0, 0]):
        self._coord = np.asarray(coord, dtype=np.float32)

    @abstractmethod
    def __call__(self, dt):
        raise NotImplementedError

    def set_coord(self, coord):
        self._coord = np.asarray(coord, dtype=np.float32)

    def get_coord(self):
        return self._coord.copy()


class StableMove(Move):
    def __call__(self, dt):
        return self.coord.copy()


class StandardMove(Move):
    def __init__(self, coord, velocity=[0, 0], accel=[0, 0]):
        super().__init__(coord)
        self._velocity = np.asarray(velocity, dtype=np.float32)
        self._accel = np.asarray(accel, dtype=np.float32)

    def __call__(self, dt):
        self._coord = self._coord + self._velocity * dt
        self._velocity = self._velocity + self._accel * dt
        return self._coord.copy()

    def set_velocity(self, velocity):
        self._velocity = np.asarray(velocity, dtype=np.float32)

    def get_velocity(self):
        return self._velocity.copy()

    def set_accel(self, accel):
        self._accel = np.asarray(accel, dtype=np.float32)

    def get_accel(self):
        return self._accel.copy()


class AttenuateMove(StandardMove):
    def __init__(self, coord, velocity=[0, 0], accel=[0, 0],
                 elasticity=1.0, gamma=0.01):
        super().__init__(coord, velocity, accel)
        self._elasticity = elasticity
        self._gamma = gamma

    def __call__(self, dt):
        amp = (1 - self._gamma * dt)
        self._coord = self._coord + self._velocity * dt
        self._velocity = self._velocity * amp + self._accel * dt
        return self._coord.copy()

    def set_velocity(self, velocity):
        super().set_velocity(velocity * self._elasticity)
