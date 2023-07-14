import numpy as np

class Matrix:

	def __init__(self, data):
		"""
		data: list of lists
		shape: the dimensions of the matrix as a tuple (rows, columns)
		"""
		self.data = []
		# the elements of the matrix as a list of lists: Matrix([[1.0, 2.0], [3.0, 4.0]])
		if isinstance(data, list):
			if all(isinstance(elem, list) and len(data[0]) == len(elem) and all(type(i) in [int, float, complex] for i in elem) for elem in data):
				self.data = data
				self.shape = (len(data), len(data[0])) 
		# a shape: Matrix((3, 3)) (the matrix will be filled with zeros by default)
		elif isinstance(data, tuple) and len(data) == 2 and all(isinstance(elem, int) and elem >= 0 for elem in data):
			for i in range(data[0]):
				row = []
				for j in range(data[1]):
					row.append(0)
				self.data.append(row)
				self.shape = (data[0], data[1])
		else:
			raise ValueError("Invalid form of data,", data)

	def T(self):
		".T() method which returns the transpose of the matrix."
		transposed = []
		for j in range(self.shape[1]):
			row = []
			for i in range(self.shape[0]):
				row.append(self.data[i][j])
			transposed.append(row)
		return Matrix(transposed)

	def mul_vec(self, other):
		if isinstance(other, Vector):
			if self.shape[1] != other.size:
			#if self.shape[1] != other.shape[0]:
				raise ValueError("Matrices cannot be multiplied, dimensions don't match.")
			other.data = np.reshape(other.data, (self.shape[1], -1)).tolist()
			other.shape = (self.shape[1], 1)
			result = [[sum([self.data[i][k] * other.data[k][j] for k in range(self.shape[1])]) for j in range(other.shape[1])] for i in range(self.shape[0])]
			return Vector(result)
		else:
			raise TypeError("Invalid type of input value.")

	def mul_mat(self, other):
		if isinstance(other, Matrix):
			if self.shape[1] != other.shape[0]:
				raise ValueError("Matrices cannot be multiplied, dimensions don't match.")
			result = [[sum([self.data[i][k] * other.data[k][j] for k in range(self.shape[1])]) for j in range(other.shape[1])] for i in range(self.shape[0])]
			return Matrix(result)
		else:
			raise TypeError("Invalid type of input value.")

	def trace(self):
		if self.shape[0] != self.shape[1]:
			raise TypeError("Trace is undefined for non-square matrices.")
		trace = 0.0
		for i in range(self.shape[0]):
			trace += self.data[i][i]
		return trace

	def row_echelon(self):
		# gaussian elimination with back-substitution for reduced row echelon from
		pivot = 0
		for row in range(self.shape[0]):
			if pivot >= self.shape[1]:
				break
			# find a non-zero pivot element in the current pivot
			while self.data[row][pivot] == 0:
				pivot += 1
				if pivot >= self.shape[1]:
					return self
			# swap the current row with a row containing a non-zero pivot element
			for i in range(row + 1, self.shape[0]):
				if self.data[i][pivot] != 0:
					self.data[row], self.data[i] = self.data[i], self.data[row]
					break
			# scale the current row to make the pivot element 1
			divisor = self.data[row][pivot]
			self.data[row] = [elem / divisor for elem in self.data[row]]

			# perform the row operations to eliminate other non-zero elements in the current column
			for i in range(self.shape[0]):
				if i != row:
					multiplier = self.data[i][pivot]
					self.data[i] = [elem - multiplier * self.data[row][j] for j, elem in enumerate(self.data[i])]
			pivot += 1
		return self

	def determinant2(self):
		if self.shape[0] != self.shape[1]:
			raise TypeError("Determinant is undefined for non-square matrices.")
		# base case for 2 x 2 matrix
		if self.shape[0] == 2:
			return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
		# cofactor expansion for more than 3 x 3 matrix
		determinant = 0.0
		for j in range(self.shape[1]):
			# compute the cofactor of element self.data[0][j]
			submatrix = [[self.data[i][k] for k in range(self.shape[1]) if k != j] for i in range(1, self.shape[0])]
			submatrix = Matrix(submatrix)
			cofactor = (-1) ** j * submatrix.determinant()
			# add the cofactor multiplied by the element matrix[0][j] to the determinant
			determinant += self.data[0][j] * cofactor
		return determinant

	def determinant(self):
		if self.shape[0] != self.shape[1]:
			raise TypeError("Determinant is undefined for non-square matrices.")
	
		# base case for 2 x 2 matrix
		if self.shape[0] == 2:
			return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]

		matrix_copy = [row.copy() for row in self.data]
		det = 1.0
		
		# gaussian elimination
		for i in range(self.shape[0]):
			# find the pivot
			for j in range(i, self.shape[0]):
				if matrix_copy[j][i] != 0:
					# swap rows if necessary
					if i != j:
						matrix_copy[i], matrix_copy[j] = matrix_copy[j], matrix_copy[i]
						det *= -1
					
					# scale the current row to make the pivot element 1
					pivot = matrix_copy[i][i]
					det *= pivot
					matrix_copy[i] = [elem / pivot for elem in matrix_copy[i]]
					
					# eliminate other non-zero elements in the same column
					for k in range(i + 1, self.shape[0]):
						factor = matrix_copy[k][i]
						matrix_copy[k] = [x - y * factor for x, y in zip(matrix_copy[k], matrix_copy[i])]
					break
		return det

	
	def inverse(self):
		if self.shape[0] != self.shape[1]:
			raise TypeError("Inverse is undefined for non-square matrices.")
		if self.determinant() == 0:
			raise ValueError(f"Matrix is not invertable.")
		# create an augmented matrix [A|I]
		augmented_matrix = [row + [float(i == j) for j in range(self.shape[0])] for i, row in enumerate(self.data)]
		augmented_matrix = Matrix(augmented_matrix)

		# apply Gauss-Jordan elimination to obtain the reduced row-echelon form
		rref_matrix = augmented_matrix.row_echelon()

		# extract the inverse matrix [I|B]
		inverse_matrix = [row[self.shape[0]:] for row in rref_matrix.data]
		return inverse_matrix
	
	def rank(self):
		matrix_copy = Matrix(self.data)
		# apply Gauss-Jordan elimination to obtain the reduced row-echelon form
		matrix_copy.row_echelon()
		rank = 0
		# count the number of non-zero rows
		for row in matrix_copy.data:
			if any(row):
				rank += 1
		return rank

	# add : only matrices of same dimensions.
	def __add__(self, other):
		if not isinstance(other, Matrix):
			raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))
			#raise TypeError(f"Invalid input: {func.__name__} requires a Matrix object.")
		if self.shape != other.shape:
			raise ValueError(f"Invalid input: addition requires a Matrix of same shape.")
		result = [[self.data[i][j] + other.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
		return Matrix(result)

	def __radd__(self, other):
		if not isinstance(other, Matrix):
			raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))
		if self.shape != other.shape:
			raise ValueError("Invalid input: __add__ requires matrics of the same shape.")
		return other + self

	# sub : only matrices of same dimensions.
	def __sub__(self, other):
		if not isinstance(other, Matrix):
			raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))
			#raise TypeError(f"Invalid input: {func.__name__} requires a Matrix object.")
		if self.shape != other.shape:
			raise ValueError(f"Invalid input: subtraction requires a Matrix of same shape.")
		result = [[self.data[i][j] - other.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
		return Matrix(result)

	def __rsub__(self, other):
		if not isinstance(other, Matrix):
			raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))
		if self.shape != other.shape:
			raise ValueError("Invalid input: __add__ requires matrics of the same shape.")
		return other - self

	# div : only scalars.
	def __truediv__(self, scalar):
		if isinstance(scalar, Matrix):
			raise NotImplementedError("Division with a Matrix object is not implemented.")
		if not any(isinstance(scalar, scalar_type) for scalar_type in [int, float, complex]):
			raise TypeError("Invalid input of scalar value.")
		if scalar == 0:
			raise ValueError("Can't divide by 0.")
		result = [[self.data[i][j] / scalar for j in range(self.shape[1])] for i in range(self.shape[0])]
		return Matrix(result)

	def __rtruediv__(self, scalar):
		raise NotImplementedError("Division of a scalar by a Matrix object is not defined here.")

	# mul : scalars, vectors and matrices , can have errors with vectors and matrices,
	# returns a Vector if we perform Matrix * Vector mutliplication.
	def __mul__(self, other):
		if any(isinstance(other, scalar_type) for scalar_type in [int, float, complex]):
			result = [[self.data[i][j] * other for j in range(self.shape[1])] for i in range(self.shape[0])]
			return Matrix(result)
		elif isinstance(other, Vector):
			if self.shape[1] != other.shape[0]:
				raise ValueError("Matrices cannot be multiplied, dimensions don't match.")
			result = [[sum([self.data[i][k] * other.data[k][j] for k in range(self.shape[1])]) for j in range(other.shape[1])] for i in range(self.shape[0])]
			return Vector(result)
		elif isinstance(other, Matrix):
			if self.shape[1] != other.shape[0]:
				raise ValueError("Matrices cannot be multiplied, dimensions don't match.")
			result = [[sum([self.data[i][k] * other.data[k][j] for k in range(self.shape[1])]) for j in range(other.shape[1])] for i in range(self.shape[0])]
			return Matrix(result)
		else:
			raise TypeError("Invalid type of input value.")

	def __rmul__(self, x):
		return self * x

	def __str__(self):
		txt = f"Matrix({self.data}) {self.shape}"
		return txt

	def __repr__(self):
		txt = f"Matrix({self.data}) {self.shape}"
		return txt

	def tolist(self):
		return np.reshape(self.data, self.shape).tolist()

class Vector(Matrix):
	
	def __init__(self, data):
		self.data = []
		# when data is a list
		if isinstance(data, list):
			# initialize a list of a list of floats : Vector([[0.0, 1.0, 2.0, 3.0]])
			if len(data) == 1 and isinstance(data[0], list) and len(data[0]) > 0 and all(type(i) in [int, float, complex] for i in data[0]):	
				self.data = data
				self.shape = (1, len(data[0]))
				self.size = len(data[0])
			# initialize a list of lists of single float : Vector([[0.0], [1.0], [2.0], [3.0]])
			elif all(isinstance(elem, list) and len(elem) == 1 and all(type(i) in [int, float, complex] for i in elem) for elem in data):
				self.data = data
				self.shape = (len(data), 1)
				self.size = len(data)
			else:
				raise ValueError("Invalid form of list,", data)
		else:
			raise ValueError("Invalid form of data,", data)

	@staticmethod
	def linear_combination(lst_vectors, coefs):
		if not all(isinstance(lst, list) for lst in [lst_vectors, coefs]):
			raise ValueError("Invalid form of list")
		if not all(isinstance(v, Vector) for v in lst_vectors):
			raise TypeError("Invalid input: list should contain only Vectors.", lst_vectors)
		if not all(v.size == lst_vectors[0].size for v in lst_vectors):
			raise TypeError("Invalid input: list of Vectors should contain Vectors of the same shape.", lst_vectors)
		if len(coefs) != len(lst_vectors) or not all(type(i) in [int, float] for i in coefs):
			raise TypeError("Invalid input: unsupported type or uncompatiable length with list of Vectors", coefs)

		v_size = lst_vectors[0].size
		v = Vector([[0.0] * v_size])
		for vector, coef in zip(lst_vectors, coefs):
			v += vector * coef
		return (v)

	@staticmethod
	def lerp(u, v, t):
		if type(u) != type(v):
			raise TypeError("Invalid input: uncompatiable type")
		if not (isinstance(t, float) and (0 <= t <= 1)):
			raise ValueError("Invalid value: a real number from 0 to 1 required.", t)
		if any(isinstance(u, accepted_type) for accepted_type in [int, float, complex, Vector, Matrix]):
			return u + (v - u) * t 
		else:
			raise TypeError("Invalid input: unsupported type")

	def norm_1(self):
		abs_sum = 0.0
		lst_data = np.reshape(self.data, (1, -1))[0]
		for elem in lst_data:
			if elem >= 0:
				abs_sum += elem
			else:
				abs_sum -= elem
		return abs_sum

	def norm(self):
		squared_sum = 0.0
		lst_data = np.reshape(self.data, (1, -1))[0]
		for elem in lst_data:
			squared_sum += elem ** 2
		return squared_sum ** 0.5

	def norm_inf(self):
		max_abs_value = float('-inf')
		lst_data = np.reshape(self.data, (1, -1))[0]
		for elem in lst_data:
			if elem >= 0:
				abs_value = elem
			else:
				abs_value = -elem
			if abs_value > max_abs_value:
				max_abs_value = abs_value
		return max_abs_value

	def angle_cos(u, v):
		if not all(isinstance(vec, Vector) for vec in [u, v]):
			raise TypeError("Invalid input: it requires a Vector of compatible shape.")
		if u.size != v.size:
			raise TypeError("Invalid input: it requires a Vector of compatible shape.")
		cosine_similarity = u.dot(v) / (u.norm() * v.norm())
		return np.around(cosine_similarity, decimals=10)
	
	@staticmethod
	def cross_product(u, v):
		if not (u.size == 3 and u.size == v.size):
			raise TypeError("Invalid input: it requires two-3dimensional Vectors.")
		x1, y1, z1 = u.tolist()
		x2, y2, z2 = v.tolist()

		cross_x = (y1 * z2 - y2 * z1)
		cross_y = (z1 * x2 - z2 * x1)
		cross_z = (x1 * y2 - x2 * y1)
		return Vector([[cross_x, cross_y, cross_z]])
	
	def dot(self, other):
		if not isinstance(other, Vector):
			raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))
		if self.size != other.size:
			raise TypeError("Invalid input: it requires a Vector of compatible shape.")
		dot_product = 0.0
		for i in range(self.size):
			dot_product += self.tolist()[i] * other.tolist()[i]
		return dot_product

	def dot2(self, other):
		if not isinstance(other, Vector):
			raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))
		if self.shape[1] != other.shape[0]:
			raise TypeError("Invalid input: dot product requires a Vector of compatible shape.")
		result = 0.0
		for i in range(self.shape[0]):
			for j in range(self.shape[1]):
				result += self.data[i][j] * other.data[j][i]
		return result

	def T(self):
		".T() method which returns the transpose of the matrix."
		transposed = []
		for j in range(self.shape[1]):
			row = []
			for i in range(self.shape[0]):
				row.append(self.data[i][j])
			transposed.append(row)
		return Matrix(transposed)

	def __add__(self, other):
		if not isinstance(other, Vector):
			raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))
		if self.shape != other.shape:
			raise ValueError("Invalid input: __add__ requires vectors of the same shape.")
		result = [[self.data[i][j] + other.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
		return Vector(result)
	
	def __sub__(self, other):
		if not isinstance(other, Vector):
			raise TypeError("unsupported operand type(s) for -: '{}' and '{}'".format(type(self), type(other)))
		if self.shape != other.shape:
			raise ValueError("Invalid input: __sub__ requires vectors of the same shape.")
		result = [[self.data[i][j] - other.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
		return Vector(result)
	
	def __mul__(self, other):
		if any(isinstance(other, scalar_type) for scalar_type in [int, float, complex]):
			result = [[self.data[i][j] * other for j in range(self.shape[1])] for i in range(self.shape[0])]
			return Vector(result)
		elif isinstance(other, Vector):
			if self.shape[1] != other.shape[0]:
				raise ValueError("Vectors cannot be multiplied, dimensions don't match.")
			result = [[self.data[i][j] * other for j in range(self.shape[1])] for i in range(self.shape[0])]
			return Vector(result)
		elif isinstance(other, Matrix):
			if self.shape[1] != other.shape[0]:
				raise ValueError("Matrices cannot be multiplied, dimensions don't match.")
			result = [[sum([self.data[i][k] * other.data[k][j] for k in range(self.shape[1])]) for j in range(other.shape[1])] for i in range(self.shape[0])]
			return Matrix(result)
		else:
			raise TypeError("Invalid type of input value.")
	
	def __truediv__(self, scalar):
		if isinstance(scalar, Vector):
			raise NotImplementedError("Vector division is not implemented.")
		elif not any(isinstance(scalar, scalar_type) for scalar_type in [int, float, complex]):
			raise TypeError("Invalid input of scalar value.")
		if scalar == 0:
			raise ValueError("Can't divide by 0.")
		result = [[self.data[i][j] / scalar for j in range(self.shape[1])] for i in range(self.shape[0])]
		return Vector(result)

	def __str__(self):
		txt = f"Vector({self.data}) {self.shape}"
		return txt

	def __repr__(self):
		txt = f"Vector({self.data}) {self.shape}"
		return txt

	def tolist(self):
		return np.reshape(self.data, (1, -1))[0].tolist()

