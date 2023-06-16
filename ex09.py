import os, sys
import numpy as np

path = os.path.join(os.path.dirname(__file__), '..', '')
sys.path.insert(1, path)
from matrix import Matrix

if __name__ == "__main__":
	m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
	print("original:", m1)
	print("transposed:", m1.T()) # Output: Matrix([[0., 2., 4.], [1., 3., 5.]])

	print()
	m1 = Matrix([[0., 2., 4.], [1., 3., 5.]])
	print("original:", m1)
	print("transposed:", m1.T()) # Output: Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
