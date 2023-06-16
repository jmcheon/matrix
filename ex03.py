import os, sys
import numpy as np

path = os.path.join(os.path.dirname(__file__), '..', '')
sys.path.insert(1, path)
from matrix import Matrix, Vector

if __name__ == "__main__":
	u = Vector([[0., 0.]]);
	v = Vector([[1.], [1.]]);
	print(u.dot(v)); # 0.0

	u = Vector([[1., 1.]]);
	v = Vector([[1.], [1.]]);
	print(u.dot(v)); # 2.0

	u = Vector([[-1., 6.]]);
	v = Vector([[3.], [2.]]);
	print(u.dot(v)); # 9.0
