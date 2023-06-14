import os, sys

path = os.path.join(os.path.dirname(__file__), '..', '')
sys.path.insert(1, path)
from matrix import Matrix, Vector

if __name__ == "__main__":
	print("\nVector operations\n")
	u = Vector([[2., 3.]])
	v = Vector([[5., 7.]])
	print("addition:\n", u + v) # [7.0][10.0]

	print("subtraction:\n", u - v) # [-3.0][-4.0]

	print("scalar multiplication:\n", u * 2) # [4.0][6.0]

	print("\nMatrix operations\n")
	u = Matrix([[1., 2.], [3., 4.]])
	v = Matrix([[7., 4.], [-2., 2.]])
	print("addition:\n", u + v) # [8.0, 6.0][1.0, 6.0]

	print("subtraction:\n", u - v) # [-6.0, -2.0][5.0, 2.0]

	print("scalar multiplication:\n", u * 2) # [2.0, 4.0][6.0, 8.0]
