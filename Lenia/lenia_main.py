"""Lenia simulation (continuous cellular automaton).

This module contains a minimal Lenia implementation based on a 2D convolution
kernel ("activation") and a growth function that updates the board state.

The typical workflow is:
- Define a kernel radial profile (`arr_func`) and a growth function.
- Create a `Lenia` instance via `main()` (or directly).
- Iterate `Lenia.step()` and visualize `Lenia.board`.

Notes:
- The convolution is performed with periodic boundary conditions (wrap).
- Cell values are kept in the range [0, 1].
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.signal import convolve2d as conv2d
import matplotlib.animation as animation
from typing import Callable
from time import sleep

from array_creator import Array

random.seed(0)


class Lenia():
	def __init__(
		self,
		arr_func: Callable[[float, float], float],
		growth_func: Callable[[float], float],
		board_size: tuple[int, int] = (1000, 1000),
		act_size: tuple[int, int] = (50, 50),
		t_step_size: float = 0.1,
	):
		"""Create a Lenia simulation instance.

		Parameters
		----------
		arr_func:
			Kernel generator function. It is sampled on a 2D grid (size `act_size`) to
			create the convolution kernel `self.act`. Conventionally this is a radial
			profile taking `(x, y)` coordinates.
		growth_func:
			Growth function applied to the convolution result. It should accept a NumPy
			array (the activation field) and return a NumPy array of the same shape.
		board_size:
			Size of the simulation board `(height, width)`.
		act_size:
			Size of the kernel array `(height, width)`.
		t_step_size:
			Time step size (Delta t) for Euler integration.

		Attributes
		----------
		board:
			Current board state in [0, 1]. Initialized via `init_board()`.
		act:
			Convolution kernel derived from `arr_func`.
		conv:
			Last convolution result (activation field), available after `step()`.
		"""
		self.board_size = board_size  # Spielfeld Größe
		arr_conv = Array(arr_func, act_size)
		arr_conv.main()
		self.act = arr_conv.arr  # Array für Conv
		self.t_step_size = t_step_size
		self.last_diff = 1
		self.growth_func = growth_func
	
	def init_board(self) -> None:
		"""Initialize the board state.

		By default this sets `self.board` to all zeros and `self.last_board` to ones.
		If the global pattern `orbium` is not `None`, that pattern is inserted near
		the top-left corner. Otherwise the board is filled with random values in
		[0, 1].

		Notes
		-----
		- The current implementation relies on the global variable `orbium`.
		- Random initialization uses Python's `random` module (seeded at import).
		"""
		self.board = np.zeros(self.board_size)
		self.last_board = np.ones(self.board_size)
		if orbium is not None:
			for i1, v1 in enumerate(orbium):
				for i2, v2 in enumerate(v1):
					self.board[i1+10][i2+10] = orbium[i1][i2]
		else:
			for i1, v1 in enumerate(self.board):
				for i2, v2 in enumerate(v1):
					self.board[i1][i2] = random.random()

	def step(self, auto_restart: bool = True):
		"""Advance the simulation by one time step.

		This computes the activation field via 2D convolution (`self.board` * `self.act`)
		with wrap-around boundary conditions, applies the growth function, and updates
		the board using an Euler step.

		In simplified form:
		- A = conv2d(board, act)
		- board <- clip(board + t_step_size * growth_func(A), 0, 1)

		Parameters
		----------
		auto_restart:
			If True, re-initialize the board when the change compared to the previous
			state becomes very small (simple stagnation detection).

		Returns
		-------
		numpy.ndarray
			The updated board (same object as `self.board`).
		"""
		conv = conv2d(self.board, self.act, mode = 'same', boundary = 'wrap')
		self.board = np.clip(self.board + self.t_step_size * self.growth_func(conv), 0, 1)

		#print(np.sum(self.last_board - self.board))
		if np.max(np.abs(self.last_board - self.board)) <= 0.08 and auto_restart:
			self.init_board()
			sleep(0.3)

		#self.last_diff = np.sum(self.last_board - self.board)
		self.last_board = self.board

		self.conv = conv
		return self.board

	
	def main(self):
		"""Initialize the simulation.

		Currently this is a thin wrapper around `init_board()` and exists mostly for
		compatibility with older code.
		"""
		self.init_board()


orbium = [[0,0,0,0,0,0,0.1,0.14,0.1,0,0,0.03,0.03,0,0,0.3,0,0,0,0],
				[0,0,0,0,0,0.08,0.24,0.3,0.3,0.18,0.14,0.15,0.16,0.15,0.09,0.2,0,0,0,0],
				[0,0,0,0,0,0.15,0.34,0.44,0.46,0.38,0.18,0.14,0.11,0.13,0.19,0.18,0.45,0,0,0],
				[0,0,0,0,0.06,0.13,0.39,0.5,0.5,0.37,0.06,0,0,0,0.02,0.16,0.68,0,0,0],
				[0,0,0,0.11,0.17,0.17,0.33,0.4,0.38,0.28,0.14,0,0,0,0,0,0.18,0.42,0,0],
				[0,0,0.09,0.18,0.13,0.06,0.08,0.26,0.32,0.32,0.27,0,0,0,0,0,0,0.82,0,0],
				[0.27,0,0.16,0.12,0,0,0,0.25,0.38,0.44,0.45,0.34,0,0,0,0,0,0.22,0.17,0],
				[0,0.07,0.2,0.02,0,0,0,0.31,0.48,0.57,0.6,0.57,0,0,0,0,0,0,0.49,0],
				[0,0.59,0.19,0,0,0,0,0.2,0.57,0.69,0.76,0.76,0.49,0,0,0,0,0,0.36,0],
				[0,0.58,0.19,0,0,0,0,0,0.67,0.83,0.9,0.92,0.87,0.12,0,0,0,0,0.22,0.07],
				[0,0,0.46,0,0,0,0,0,0.7,0.93,1,1,1,0.61,0,0,0,0,0.18,0.11],
				[0,0,0.82,0,0,0,0,0,0.47,1,1,0.98,1,0.96,0.27,0,0,0,0.19,0.1],
				[0,0,0.46,0,0,0,0,0,0.25,1,1,0.84,0.92,0.97,0.54,0.14,0.04,0.1,0.21,0.05],
				[0,0,0,0.4,0,0,0,0,0.09,0.8,1,0.82,0.8,0.85,0.63,0.31,0.18,0.19,0.2,0.01],
				[0,0,0,0.36,0.1,0,0,0,0.05,0.54,0.86,0.79,0.74,0.72,0.6,0.39,0.28,0.24,0.13,0],
				[0,0,0,0.01,0.3,0.07,0,0,0.08,0.36,0.64,0.7,0.64,0.6,0.51,0.39,0.29,0.19,0.04,0],
				[0,0,0,0,0.1,0.24,0.14,0.1,0.15,0.29,0.45,0.53,0.52,0.46,0.4,0.31,0.21,0.08,0,0],
				[0,0,0,0,0,0.08,0.21,0.21,0.22,0.29,0.36,0.39,0.37,0.33,0.26,0.18,0.09,0,0,0],
				[0,0,0,0,0,0,0.03,0.13,0.19,0.22,0.24,0.24,0.23,0.18,0.13,0.05,0,0,0,0],
				[0,0,0,0,0,0,0,0,0.02,0.06,0.08,0.09,0.07,0.05,0.01,0,0,0,0,0]]
orbium = None

def main(board_size:tuple[int,int] = (100, 100), act_size:tuple[int, int] = (30,30), 
	t_step_size:float = 0.1, my:float = 0.175, sigma:float = 0.0135):
	"""Convenience factory to create a configured `Lenia` instance.

	This function wires together a commonly used kernel profile (`arr_func`) and a
	Gaussian "bell" growth function parameterized by `my` (mean) and `sigma`
	(standard deviation).

	Parameters
	----------
	board_size:
		Board size `(height, width)`.
	act_size:
		Kernel array size `(height, width)`.
	t_step_size:
		Time step size for the Euler update.
	my:
		Center of the bell-shaped growth function.
	sigma:
		Width of the bell-shaped growth function.

	Returns
	-------
	tuple
		`(lenia_instance, (growth_func, '_'))` to preserve the original calling
		convention used below in the `__main__` demo.
	"""


	def arr_func(x:float, y:float=0):
		"""Radial kernel profile used to generate the convolution kernel.

		The function is evaluated on a grid by `array_creator.Array`.
		"""
		a = (x**2 + y**2)**0.5
		f = 2**(-10 * a**2) * a**3 #~double Bell
		return f

	bell = lambda x: np.exp(-((x-my)/sigma)**2 / 2)

	def growth_func(U):
		"""Compute growth from the activation field.

		Parameters
		----------
		U:
			Activation field (result of the convolution). Typically a NumPy array.

		Returns
		-------
		numpy.ndarray
			Growth values in [-1, 1] (approximately), matching the shape of `U`.
		"""
		return bell(U)*2-1
	
	test = Lenia(arr_func, growth_func, board_size = board_size, act_size = act_size, t_step_size = t_step_size)
	test.main()

	return(test, (growth_func, '_'))



if __name__ == '__main__':

	test, arguments = main(board_size = (200, 200), act_size = (20, 20))
	growth_func = arguments[0]

	frames:int = 1

	
	plt.imshow(test.act)
	plt.show()
	
	plt.plot(np.linspace(0,1,500), growth_func(np.linspace(0,1,500)), label = 'Growth-Funnktion')
	plt.show()


	fig = plt.figure(figsize = (5, 5), dpi = 200)
	im = plt.imshow(test.board, vmin = 0.0, vmax = 1.0, animated=True)
			
		
	def updatefig(frame):
		im.set_array(test.step())

		if frame == frames - 1 and frames != 1:	test.main()

		return im,
			
	ani = animation.FuncAnimation(fig, updatefig, interval=0, blit=True, cache_frame_data=False, repeat = True, frames = frames)
			
	plt.show()

			