# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

class BaseTrigger(object):
    @abstractmethod
    def __bool__(self):
        return False


class ToggleTrigger(BaseTrigger):
    def __init__(self, default=False):
        self._state = default

    def __call__(self, *largs, **kwargs):
        if len(largs) > 0 and isinstance(largs[0], bool):
            self._state = largs[0]
        else:
            self.toggle()

    def toggle(self):
        self._state = not self._state

    def __bool__(self):
        return self._state


class RegionTrigger(BaseTrigger):
    def __init__(self, move, region):
        self._move = move
        self._region = region

    def __bool__(self):
        coord = self._move.get_coord()
        return coord in self._region


class DomainTrigger(BaseTrigger):
    def __init__(self, move, domain, mode='both'):
        self._move = move
        self._prev = move.get_coord()
        self._domain = domain
        self._mode = mode
        self._state = mode not in ('left', 'right')
        if self._prev[0] in self._domain:
            self._state = True

    def __bool__(self):
        coord = self._move.get_coord()
        prev = self._prev
        self._prev = coord
        is_range = bool(coord[0] in self._domain)
        if self._mode == 'left' and self._domain > prev[0]:
            if self._domain > prev[0]:
                self._state = is_range
            elif not is_range:
                self._state = False
        elif self._mode == 'right':
            if self._domain < prev[0]:
                self._state = is_range
            elif not is_range:
                self._state = False

        return is_range and self._state


class FunctionalTrigger(BaseTrigger):
    def __init__(self, func):
        self._func = func

    def __bool__(self):
        return self._func()


class SumEventTrigger(BaseTrigger):
    def __init__(self):
        self._triggers = list()

    def add_trigger(self, trig):
        self._triggers.append(trig)

    def __bool__(self):
        for trig in self._triggers:
            if trig:
                return True
        return False


class ProductEventTrigger(BaseTrigger):
    def __init__(self):
        self._triggers = list()

    def add_trigger(self, trig):
        self._triggers.append(trig)

    def __bool__(self):
        for trig in self._triggers:
            if not trig:
                return False
        return True


class SceneTrigger(BaseTrigger):
    def __init__(self):
        self._triggers = list()
        self._prev = False
        self._idx = 0

    def add_trigger(self, trig):
        self._triggers.append(trig)

    def __bool__(self):
        trig = bool(self._triggers[self._idx])
        if self._prev and not trig:
            self._idx = (self._idx + 1) % len(self._triggers)
        self._prev = trig
        return trig


class NullTrigger(BaseTrigger):
    def __init__(self, state=True):
        self._state = state

    def __bool__(self):
        return self._state
