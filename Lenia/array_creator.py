"""Utilities to sample a 2D function into a normalized NumPy array.

The `Array` helper class evaluates a callable `func(x, y)` on an evenly spaced
grid in the range [-1, 1] x [-1, 1]. The resulting array is normalized by its
sum (so it integrates to ~1 over the grid).

This is used by the Lenia code to build convolution kernels from radial profile
functions.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Callable

class Array():
	def __init__(self, func:Callable[[float, float], float], size:tuple[int,int] = (10,10), feedback:bool = False):
		"""Create an array sampler.

		Parameters
		----------
		func:
			Callable of the form `func(x, y) -> z`.
		size:
			Requested output array size `(width, height)`. Even sizes are corrected
			to odd sizes so that the center cell corresponds to (0, 0).
		feedback:
			If True, prints basic information while correcting sizes / normalizing.
		"""
		self.func = func
		self.feedback = feedback
		self.corr_size(size)
		self.arr = np.zeros(self.size)

	def corr_size(self, size):
		"""Correct the requested size to odd dimensions.

		Lenia kernels are typically centered; odd dimensions ensure there is a
		well-defined center cell.
		"""
		size = list(size)
		if size[0] % 2 == 0:
			size[0] += 1
			if self.feedback: print('Enlarged x-Axis by 1')
		if size[1] % 2 == 0:
			size[1] += 1
			if self.feedback: print('Enlarged y-Axis by 1')
		self.size = size

	def calc_arr(self):
		"""Compute and normalize the sampled array."""
		xaxis = np.linspace(start = -1, stop = 1, num = self.size[0])
		yaxis = np.linspace(start = -1, stop = 1, num = self.size[1])

		for i1, x in enumerate(xaxis):
			for i2, y in enumerate(yaxis):
				self.arr[i1][i2] = self.func(x, y)

		sum = np.sum(self.arr)
		self.arr /= sum
		if self.feedback: print(sum)
		
	def main(self):
		"""Public entry point: compute array and return it."""
		self.calc_arr()
		return self.arr



if __name__ == '__main__':
	def func(x, y):
		a = (x**2 + y**2)**0.5
		f = 10** ( -a**2 )#2** (-5 * a**2) * (2 * a**2 -0.1) 
		return(f)
	#size = (1000,1000)
	size = (10, 10)
	test = Array(func, size)
	test.main()


	plt.imshow(test.arr)
	plt.show()