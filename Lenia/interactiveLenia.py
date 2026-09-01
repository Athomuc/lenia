"""Interactive Lenia painter.

This script runs a Lenia simulation and lets you "paint" additional mass onto
the board using the mouse.

- Hold the left mouse button and move to draw continuously.
- Click left mouse button to draw at a point.

The brush ("pen") is generated as a small normalized kernel and is merged into
the board via `overwrite_array`.

Note: This module is primarily designed to be run as a script.
"""

from lenia_main import main as lenia_main
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backend_bases import MouseButton

from array_creator import Array


def setup_pen(draw_size:int = 5):
	"""Create and normalize the drawing brush ("pen").

	Parameters
	----------
	draw_size:
		Brush size in pixels. The brush is created as a square array.

	Returns
	-------
	tuple
		`(pen_array, offset)` where `pen_array` is normalized to a max of 1 and
		`offset` is the half-size used to center the brush around the cursor.
	"""

	def array_func(x, y): #Wie soll der "Stift aussehen?"
		a = (x**2 + y**2)**0.5
		f = 10**( - 2* a**2 )
		return f

	draw_array = Array(array_func, (draw_size, draw_size)) #"Stift" erzeugen und ausgeben
	draw_array = draw_array.main()
	draw_array *= 1/np.max(draw_array)

	offset = int((draw_size-1)/2) #Welchen Offset hat der "Stift" vom Mauszeiger

	return draw_array, offset



def setup_board(board_size: tuple[int,int] = (100, 100), array_size : tuple[int,int] = (10, 10)):
	"""Set up figure, event handlers and animation update function.

	Parameters
	----------
	board_size:
		Size of the Lenia board.
	array_size:
		Unused in the current implementation (kept for compatibility).

	Returns
	-------
	tuple
		`(fig, updatefig)` where `updatefig(frame)` advances Lenia and applies any
		queued brush strokes.
	"""

	lenia, _ = lenia_main(board_size)

	fig = plt.figure(figsize = (4, 4), dpi = 200)
	im = plt.imshow(lenia.board, vmin = 0.0, vmax = 1.0, animated=True)

	def on_move(event):
		"""Track cursor positions while holding the left mouse button."""
		if event.inaxes and event.button is MouseButton.LEFT:
			x.append(event.xdata)
			y.append(event.ydata)

	def on_click(event):
		"""Track a single click position for the left mouse button."""
		if event.inaxes and event.button is MouseButton.LEFT:
			x.append(event.xdata)
			y.append(event.ydata)


	plt.connect('motion_notify_event', on_move)
	plt.connect('button_press_event', on_click)


	def updatefig(frame):
		"""Advance simulation and apply pending drawing operations."""
		new_array = lenia.step(auto_restart)

		while len(x) != 0:
			set_to_x, set_to_y = int(x.pop(0)+0.5) - offset, int(y.pop(0)+0.5) -offset
			new_array = overwrite_array(new_array, pen, (set_to_x, set_to_y))

		im.set_array(new_array)

		lenia.board = new_array

		if frame == frames - 1 and frames != 1:	lenia.main()

		return im,

	return fig, updatefig


def overwrite_array(A:np.ndarray, B:np.ndarray, coords: tuple[int, int]) -> np.ndarray:
	"""Overlay array `B` onto `A` (taking the elementwise maximum).

	The top-left corner of `B` is placed at `coords` in `A`. The result is
	equivalent to writing `max(A, placed_B)`.

	If the placement would run out of bounds, the original `A` is returned.

	Parameters
	----------
	A:
		Target 2D array.
	B:
		Source 2D array (typically smaller than `A`).
	coords:
		Placement coordinates `(x, y)` for the top-left corner of `B`.

	Returns
	-------
	numpy.ndarray
		A new array containing the overlay result (or `A` if out of bounds).
	"""
	try:
		B = np.pad(B, ((coords[1], len(A[1]) - coords[1] - len(B[1])), (coords[0], len(A[0]) - coords[0] - len(B[0]))))
		C = A > B
		B[C] = A[C]

		return B
	except ValueError:
		return A


global x, y, pen, offset  # Listen für Pixel mit denen im Moment interagiert wird
x, y = [], []

draw_size: int = 30  # Wie groß soll der "Stift" sein
board_size: tuple[int, int] = (200, 200)
array_size: tuple[int, int] = (20, 20)

pen, offset = setup_pen(draw_size)

auto_restart: bool = True
fig, updatefig = setup_board(board_size, array_size)

frames: int = 1

ani = animation.FuncAnimation(fig, updatefig, interval=0, blit=True, cache_frame_data=False, repeat=True, frames=frames)
		
plt.show()