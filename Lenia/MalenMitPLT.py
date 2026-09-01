"""Minimal Matplotlib mouse interaction demo.

This script opens a 2D plot and moves a marker to the current mouse position
while holding the left mouse button (or when clicking).

It is used as a quick sandbox for experimenting with Matplotlib event handling.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton



fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_aspect('equal')
line, = ax.plot([], [], marker="o", color="lime")

def on_move(event):

    """Handle mouse movement while holding the left button."""
    if event.inaxes and event.button is MouseButton.LEFT:
        update_plot(event.xdata, event.ydata)

def on_click(event):

    """Handle left mouse clicks inside the axes."""
    if event.inaxes and event.button is MouseButton.LEFT:
        update_plot(event.xdata, event.ydata)

def update_plot(x:float,y:float):

    """Update the marker position and print coordinates."""
    print(f'{x}, {y}\r', end='')
    line.set_data([x],[y])
    plt.draw()

plt.connect('motion_notify_event', on_move)
plt.connect('button_press_event', on_click)

plt.show()

print()