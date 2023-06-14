import os, sys
import numpy as np

path = os.path.join(os.path.dirname(__file__), '..', '')
sys.path.insert(1, path)
from matrix import Matrix, Vector

if __name__ == "__main__":
	v = Vector([[1., 0.]]);
	u = Vector([[1., 0.]]);
	print(Vector.angle_cos(u, v)); # 1.0

	u = Vector([[1., 0.]]);
	v = Vector([[0., 1.]]);
	print(Vector.angle_cos(u, v)); # 0.0

	u = Vector([[-1., 1.]]);
	v = Vector([[ 1., -1.]]);
	print(Vector.angle_cos(u, v)); # -1.0

	u = Vector([[2., 1.]]);
	v = Vector([[4., 2.]]);
	print(Vector.angle_cos(u, v)); # 1.0

	u = Vector([[1., 2., 3.]]);
	v = Vector([[4., 5., 6.]]);
	print(Vector.angle_cos(u, v)); # 0.974631846
