#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from drawer import WindowDrawer, MP4Drawer
from region import Region
from move import StandardMove
from objects import Ball, Trace, Area


if __name__ == '__main__':
    # drawer = WindowDrawer()
    drawer = MP4Drawer()
    region = Region([[-0.8, -0.8], [-0.8, 0.8], [0.8, 0.8], [0.8, -0.8]])
    move = StandardMove([0.5, 0.5], [-0.5, 0], [0, -0.3])

    ball = Ball(move, region, 0.1)
    trace = Trace(ball)
    area = Area(move, region)

    drawer.add_object(ball)
    drawer.add_object(trace, color='b')
    drawer.add_object(area, color='g')
    drawer.start()
