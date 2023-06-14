import os, sys
import numpy as np

path = os.path.join(os.path.dirname(__file__), '..', '')
sys.path.insert(1, path)
from matrix import Vector

if __name__ == "__main__":
	u = Vector([[0., 0., 1.]]);
	v = Vector([[1., 0., 0.]]);
	print(Vector.cross_product(u, v)); # [0.][1.][0.]
	print(np.cross([0., 0., 1.], [1., 0., 0.]))

	u = Vector([[1., 2., 3.]]);
	v = Vector([[4., 5., 6.]]);
	print(Vector.cross_product(u, v)); # [-3.][6.][-3.]

	u = Vector([[4., 2., -3.]]);
	v = Vector([[-2., -5., 16.]]);
	print(Vector.cross_product(u, v)); # [17.][-58.][-16.]
