import os, sys
import numpy as np

path = os.path.join(os.path.dirname(__file__), '..', '')
sys.path.insert(1, path)
from matrix import Matrix, Vector

if __name__ == "__main__":
	u = Matrix([[1., 0.],[0., 1.]]);
	print(u.trace()); # 2.0

	u = Matrix([[2., -5., 0.],[4., 3., 7.],[-2., 3., 4.]]);
	print(u.trace()); # 9.0

	u = Matrix([[-2., -8., 4.],[1., -23., 4.],[0., 6., 4.]]);
	print(u.trace()); # -21.0
