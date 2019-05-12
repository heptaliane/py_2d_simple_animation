# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Drawer(object):
    def __init__(self, xlim=[-1, 1], ylim=[-1, 1], aspect=1, dt=0.02):
        self.fig = plt.figure()
        self._ax = self.fig.add_subplot(111)
        self._ax.set_xlim(*xlim)
        self._ax.set_ylim(*ylim)
        self._ax.set_aspect(aspect)
        self._obj = list()
        self.dt = dt

    def add_object(self, obj, color='r'):
        canvas, = self._ax.plot([], [], color=color)
        self._obj.append((obj, canvas))

    def update(self, frame):
        layers = list()
        for obj, canvas in self._obj:
            xs, ys = obj(self.dt)
            canvas.set_data(xs, ys)
            layers.append(canvas)

        return layers

    def start(self, frames=1000, interval=20):
        self.animation = FuncAnimation(self.fig, self.update,
                                       frames=frames, interval=interval)
        plt.show()
