# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class BaseDrawer(object):
    def __init__(self, filename='output', xlim=[-1, 1],
                 ylim=[-1, 1], dt=0.02):
        self.fig = plt.figure()
        self._ax = self.fig.add_subplot(111)
        self._ax.set_xlim(*xlim)
        self._ax.set_ylim(*ylim)
        self._ax.tick_params(labelbottom=False, bottom=False)
        self._ax.tick_params(labelleft=False, left=False)
        self._ax.set_aspect((ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))
        for k in ('left', 'right', 'top', 'bottom'):
            self._ax.spines[k].set_visible(False)
        self._obj = list()
        self.dt = dt
        self.filename = filename

    def add_object(self, obj, color='r'):
        canvas, = self._ax.plot([], [], color=color)
        self._obj.append((obj, canvas))

    def draw_stable_object(self, xs, ys, color='r'):
        self._ax.plot(xs, ys, color=color)

    def update(self, frame):
        layers = list()
        for obj, canvas in self._obj:
            xs, ys = obj(self.dt)
            canvas.set_data(xs, ys)
            layers.append(canvas)

        return layers

    def start(self, frames=1000, interval=20):
        self._frames = frames
        self.animation = FuncAnimation(self.fig, self.update,
                                       frames=frames, interval=interval)


class WindowDrawer(BaseDrawer):
    def start(self, **kwargs):
        super().start(**kwargs)
        plt.show()


class MP4Drawer(BaseDrawer):
    def update(self, frame):
        super().update(frame)
        print(frame)
        if frame % (self._frames // 100) == 0:
            print('%6d / %6d frames' % (frame, self._frames))

    def start(self, **kwargs):
        super().start(**kwargs)
        self.animation.save('%s.mp4' % self.filename, writer='ffmpeg')
