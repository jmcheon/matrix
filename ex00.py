from matrix import Matrix, Vector

def vector_addition():
	print("\nVector addition\n")
	u = Vector([[0., 0.]])
	v = Vector([[0., 0.]])
	print(u + v)# '[0, 0]'

	u = Vector([[1., 0.]])
	v = Vector([[0., 1.]])
	print(u + v)# '[1, 1]'

	u = Vector([[1., 1.]])
	v = Vector([[1., 1.]])
	print(u + v)# '[2, 2]'

	u = Vector([[21., 21.]])
	v = Vector([[21., 21.]])
	print(u + v)# '[42, 42]'

	u = Vector([[-21., 21.]])
	v = Vector([[21., -21.]])
	print(u + v)# '[0, 0]'

	u = Vector([[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]])
	v = Vector([[9., 8., 7., 6., 5., 4., 3., 2., 1., 0.]])
	print(u + v)# '[9, 9, 9, 9, 9, 9, 9, 9, 9, 9]'

def vector_subtraction():
	print("\nVector subtraction\n")
	u = Vector([[0., 0.]])
	v = Vector([[0., 0.]])
	print(u - v)# '[0, 0]'

	u = Vector([[1., 0.]])
	v = Vector([[0., 1.]])
	print(u - v)# '[1, -1]'

	u = Vector([[1., 1.]])
	v = Vector([[1., 1.]])
	print(u - v)# '[0, 0]'

	u = Vector([[21., 21.]])
	v = Vector([[21., 21.]])
	print(u - v)# '[0, 0]'

	u = Vector([[-21., 21.]])
	v = Vector([[21., -21.]])
	print(u - v)# '[-42, 42]'

	u = Vector([[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]])
	v = Vector([[9., 8., 7., 6., 5., 4., 3., 2., 1., 0.]])
	print(u - v)# '[-9, -7, -5, -3, -1, 1, 3, 5, 7, 9]'

def vector_scaling():
	print("\nVector scaling\n")
	u = Vector([[0., 0.]])
	print(u * 1)# '[0, 0]'

	u = Vector([[1., 0.]])
	print(u * 1)# '[1, 0]'

	u = Vector([[1., 1.]])
	print(u * 2)# '[2, 2]'

	u = Vector([[21., 21.]])
	print(u * 2)# '[42, 42]'

	u = Vector([[42., 42.]])
	print(u * 0.5)# '[21, 21]'

def matrix_addition():
	print("\nMatrix addition\n")
	u = Matrix([[0., 0.], [0., 0.]])
	v = Matrix([[0., 0.], [0., 0.]])
	print(u + v)# '[[0, 0], [0, 0]]'

	u = Matrix([[1., 0.], [0., 1.]])
	v = Matrix([[0., 0.], [0., 0.]])
	print(u + v)#  '[[1, 0], [0, 1]]'

	u = Matrix([[1., 1.], [1., 1.]])
	v = Matrix([[1., 1.], [1., 1.]])
	print(u + v)#  '[[2, 2], [2, 2]]'

	u = Matrix([[21., 21.], [21., 21.]])
	v = Matrix([[21., 21.], [21., 21.]])
	print(u + v)# '[[42, 42], [42, 42]]'

def matrix_subtraction():
	print("\nMatrix subtraction\n")
	u = Matrix([[0., 0.], [0., 0.]])
	v = Matrix([[0., 0.], [0., 0.]])
	print(u - v)# '[[0, 0], [0, 0]]'

	u = Matrix([[1., 0.], [0., 1.]])
	v = Matrix([[0., 0.], [0., 0.]])
	print(u - v)#  '[[1, 0], [0, 1]]'

	u = Matrix([[1., 1.], [1., 1.]])
	v = Matrix([[1., 1.], [1., 1.]])
	print(u - v)# '[[0, 0], [0, 0]]'

	u = Matrix([[21., 21.], [21., 21.]])
	v = Matrix([[21., 21.], [21., 21.]])
	print(u - v)# '[[0, 0], [0, 0]]'

def matrix_scaling():
	print("\nMatrix scaling\n")
	u = Matrix([[0., 0.], [0., 0.]])
	print(u * 0)# '[[0, 0], [0, 0]]'

	u = Matrix([[1., 0.], [0., 1.]])
	print(u * 1)#  '[[1, 0], [0, 1]]'

	u = Matrix([[1., 2.], [3., 4.]])
	print(u * 2)# '[[2, 4], [6, 8]]'

	u = Matrix([[21., 21.], [21., 21.]])
	print(u * 0.5)# '[[10.5, 10.5], [10.5, 10.5]]'

def main():
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

# O(n) / O(n)
if __name__ == "__main__":
	vector_addition();
	vector_subtraction();
	vector_scaling();
	matrix_addition();
	matrix_subtraction();
	matrix_scaling();
