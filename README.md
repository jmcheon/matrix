# matrix - An introduction to Linear Algebra

> _**Summary: Vectors and matrices, basically.**_


#### [Read this in French](README-fr.md)

## Project structure
- matrix.py

### Class Matrix, Vector

[Exercise 00 - Add, Subtract and Scale](#ex00)<br>
[Exercise 01 - Linear combination](#ex01)<br>
[Exercise 02 - Linear interpolation](#ex02)<br>
[Exercise 03 - Dot product](#ex03)<br>
[Exercise 04 - Norm](#ex04)<br>
[Exercise 05 - Cosine](#ex05)<br>
[Exercise 06 - Cross product](#ex06)<br>
[Exercise 07 - Linear map, Matrix multiplication](#ex07)<br>
[Exercise 08 - Trace](#ex08)<br>
[Exercise 09 - Transpose](#ex09)<br>
[Exercise 10 - Reduced row-echelon form](#ex10)<br>
[Exercise 11 - Determinant](#ex11)<br>
[Exercise 12 - Inverse](#ex12)<br>
[Exercise 13 - Rank](#ex13)<br>
[Exercise 14 - Bonus: Projection matrix](#ex14)<br>

---
<a id="ex00"></a>
<table>
<tr><th>Exercise 00 -  Add, Subtract and Scale </th></tr>
<tr><td>Allowed mathematical functions : None </tr>
<tr><td>Maximum time complexity : O(n) </tr>
<tr><td>Maximum space complexity : O(n)</tr>
</table>

####  Methods of class `Matrix`
```python
# add : only matrices of same dimensions.
def __add__(self, other):
	if not isinstance(other, Matrix):
		raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))
	if self.shape != other.shape:
		raise ValueError(f"Invalid input: addition requires a Matrix of same shape.")
	result = [[self.data[i][j] + other.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
	return Matrix(result)

# sub : only matrices of the same dimensions.
def __sub__(self, other):
	if not isinstance(other, Matrix):
		raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))
	if self.shape != other.shape:
		raise ValueError(f"Invalid input: subtraction requires a Matrix of same shape.")
	result = [[self.data[i][j] - other.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
	return Matrix(result)

# mul : scalars, vectors, and matrices can have errors with vectors and matrices,
# returns a Vector if we perform Matrix * Vector multiplication.
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

```
#### how they work:

1.  `__add__(self, other)`  : adds two `Matrix`objects with the same size.
2.  `__sub__(self, other)`  : subtracts a `Matrix`object by another object with the same size.
3.  `__mul__(self, other)`  : performs a multiplication between a `Matrix` object and a scalar, a `Matrix` object, and a `Vector` object.
4.  `__truediv__(self, scalar)`  : divides the elements of a `Matrix`object by a scalar.

#### Methods of class `Vecotr`
```python
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
```
#### how they work:

1.  `__add__(self, other)`  : adds two `Vector`objects with the same size.
2.  `__sub__(self, other)`  : subtracts a `Vector`object by another object with the same size.
3.  `__mul__(self, other)`  : performs a multiplication between two `Vector` objects, a `Vector` object, a `Matrix` object, and a scalar.
4.  `__truediv__(self, scalar)`  : divides the elements of a `Vector`object by a scalar.

#### Complexity:

The time and space complexities of all the methods are **O(n)**, where `n` is the size of the `Vector` and `Matrix` objects because they iterate over each element of the `Vector` and `Matrix` objects.

---
<a id="ex01"></a>
<table>
<tr><th>Exercise 01 -  Linear combination </th></tr>
<tr><td>Allowed mathematical functions : fused multiply-add function </tr>
<tr><td>Maximum time complexity : O(n) </tr>
<tr><td>Maximum space complexity : O(n)</tr>
</table>


#### Linear combination

A linear combination is a mathematical operation performed on a set of elements by multiplying each by an appropriate coefficient and then adding the results.

Specifically, in the context of a vector space, a linear combination is obtained by multiplying each basis vector by an appropriate coefficient and adding the products.

```python
def linear_combination(lst_vectors, coefs):
	if not all(isinstance(lst, list) for lst in [lst_vectors, coefs]):		
		raise ValueError("Invalid form of list")
	if not all(isinstance(v, Vector) for v in lst_vectors):
		raise TypeError("Invalid input: list should contain only Vectors.", lst_vectors)
	if not all(v.size == lst_vectors[0].size for v in lst_vectors):
		raise TypeError("Invalid input: list of Vectors should contain Vectors of the same shape.", lst_vectors)
	if len(coefs) != len(lst_vectors) or not all(type(i) in [int, float] for i in coefs):
		raise TypeError("Invalid input: unsupported type or incompatible length with list of Vectors", coefs)
                        
	v = Vector([[0.0] * lst_vectors[0].size])
	for vector, coef in zip(lst_vectors, coefs):
		v += vector * coef
	return v
```
The `linear_combination` function takes as input `lst_vecteors` a list of vectors and `coefs` a list of coefficients. It performs a linear combination of vectors by multiplying each vector by its corresponding coefficient and then adding them to get the resulting vector. The result is returned as a new vector.

#### How it works:

1. The function first performs several verifications to ensure that the inputs are valid. It checks if  `lst_vectors` and `coefs` are both lists if all the elements of `lst_vectors`are objects of class `Vector`and if the vectors have the same size and if the coefficients are the right type and of the correct length.
2. The size of vectors is taken from the first vector in `lst_vectors`.
3. An initial vector `v` is created with components initialized to zero. The vector size is determined by the size of the input vectors.
4. Using `zip` function, the `for` loop simultaneously iterates over each vector in `lst_vectors`and each corresponding coefficient in `coefs`.
5. At each iteration of the loop, the product of each component of the vector and the corresponding coefficient is calculated and added to the corresponding components of the vector `v`.
6. Once all linear combinations have been performed, the resulting vector `v` is returned.

#### Complexity:

The time complexity of this function is **O(n)**, where `n` is the number of vectors in `lst_vectors`.

The space complexity of this function is also linear **O(n)**, because the function creates a new vector `v` which stores the intermediate results and has a size equal to the size of input vectors.

This function effectively uses the `zip` function to iterate over the vectors and the coefficients in parallel, which avoids the use of nested loops and thus reduces the time complexity to **O(n)**.

---
<a id="ex02"></a>
<table>
<tr><th>Exercise 02 - Linear interpolation </th></tr>
<tr><td>Allowed mathematical functions : fused multiply-add function </tr>
<tr><td>Maximum time complexity : O(n) </tr>
<tr><td>Maximum space complexity : O(n)</tr>
</table>

```python
def lerp(u, v, t):
	if type(u) != type(v): 
		raise TypeError("Invalid input: incompatible type") 
	if not (isinstance(t, float) and (0 <= t <= 1)): 
		raise ValueError("Invalid value: a real number from 0 to 1 required.", t) 
	if any(isinstance(u, accepted_type) for accepted_type in [int, float, complex, Vector, Matrix]):
		return u + (v - u) * t
	else: 
		raise TypeError("Invalid input: unsupported type")
```

The `lerp` function performs a linear interpolation between two vectors `u` and `v` using a mixing factor `t`.

#### How it works:

1. It first checks if the types of `u` and `v` are compatible and raises a `TypeError` exception if they are not.
2. It checks if `t` is a real number between 0 and 1 inclusive and raises a `ValueError` exception if it is not.
3. Then, if the type of `u` is `int`, `float`, `complex`, `Vector`, or `Matrix`, the linear interpolation is calculated using the  `u + (v - u) * t` formula, where `v - u` represents the difference between `v` and `u`.
4. The result of the interpolation is returned.

#### Complexity:

The time complexity of this function depends on the type of `u` and `v`.

---
<a id="ex03"></a>
<table>
<tr><th>Exercise 03 - Dot product </th></tr>
<tr><td>Allowed mathematical functions : fused multiply-add function </tr>
<tr><td>Maximum time complexity : O(n) </tr>
<tr><td>Maximum space complexity : O(n)</tr>
</table>

#### Dot product

A dot product is a mathematical operation performed between two vectors. It is the sum of the products of the corresponding components of the two vectors.

```python
def dot(self, other):
	if not isinstance(other, Vector):
		raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))
	if self.size != other.size:
		raise TypeError("Invalid input: it requires a Vector of compatible shape.")
	dot_product = 0.0
	for i in range(self.size):
		dot_product += self.tolist()[i] * other.tolist()[i]
	return dot_product
```

#### How it works:

1. It checks if the `other` object is an instance of the `Vector` class, if not, it raises a `TypeError` exception.
2. It checks if the two vectors have the same size, if not, it raises a `TypeError` exception.
3. It initializes a `dot_product` variable to 0 to store the result of the dot product.
4. It loops through the elements of the two vectors using a `for` loop, multiplying the corresponding elements and adding the product to the `dot_product` variable.
5. It returns the final result in `dot_product`.

#### Complexity:

The complexity of this function is **O(n)**, where `n` is the size of the vectors because the function iterates through each element of the two vectors once.

---
<a id="ex04"></a>
<table>
<tr><th>Exercise 04 - Norm </th></tr>
<tr><td>Allowed mathematical functions : fused multiply-add function, pow, max </tr>
<tr><td>Maximum time complexity : O(n) </tr>
<tr><td>Maximum space complexity : O(n)</tr>
</table>

#### L1, L2, L-infinity norm

They are different measures of the size of a vector.

1. L1 norm (also known as `Manhattan` or `taxicab` norm) of a vector. L1 norm is the sum of absolute values of the elements of the vector.
2. L2 norm (also known as `Euclidean` norm) of a vector. L2 norm is the square root of the sum of the squares of the elements of the vector.
3. L-infinity norm (also known as `Maximum` norm) of a vector. L-infinity norm is the maximum absolute value of the elements of the vector.
    
```python
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
```
The `norm_1` function calculates the L1 norm of a vector.
The `norm` calculates the L2 norm of a vector.
The `norm_inf` calculates the L-infinity norm of a vector.

#### How it works:

The `norm_1` function loops through each element and adds its absolute value (positive or negative) to a sum and returns the total sum at the end.

The `norm` function loops through each element of the vector, adds the square of its value to a sum and then returns the square root of the total sum at the end.

The `norm_inf` function loops through each element of the vector, compares its absolute value to a maximum value and updates the maximum value if the current absolute value is greater.
It returns the maximum absolute value at the end.

#### Complexity:

The complexity of these three functions is **O(n)**, where `n` is the number of elements in the vector.

---
<a id="ex05"></a>
<table>
<tr><th>Exercise 05 - Cosine </th></tr>
<tr><td>Allowed mathematical functions : fused multiply-add function </tr>
<tr><td>Maximum time complexity : O(n) </tr>
<tr><td>Maximum space complexity : O(n)</tr>
</table>
    
```python
def angle_cos(u, v):
	if not all(isinstance(vec, Vector) for vec in [u, v]):
		raise TypeError("Invalid input: it requires a Vector of compatible shape.")
	if u.size != v.size:
		raise TypeError("Invalid input: it requires a Vector of compatible shape.")
	cosine_similarity = u.dot(v) / (u.norm() * v.norm())
	return np.around(cosine_similarity, decimals=10)
```
The `angle_cos` function calculates the cosine similarity between two vectors `u` and `v`.
The cosine similarity measures the angle between the two vectors and it is used as a measure of similarity between them - similar vectors will have a cosine similarity close to 0.

#### How it works:

1. It checks if `u` and `v` are both instances of the `Vector` class. If not, it raises a `TypeError` exception.
2. If checks if the two vectors have the same size. If not, it raises a `TypeError` exception.
3. If calculates the cosine similarity using this formula: $(u ⋅ v) / (||u|| * ||v||)$, where $(u ⋅ v)$ denotes the dot product between the `u` and `v` vectors and  $||u||$ and $||v||$ represent the respective (Eucilide) norms  of `u` and `v`.
4. It rounds the cosine similarity to 10 decimals and returns the result.

#### Complexity:

The complexity of this function depends on the `dot` and `norm` methods which are both **O(n)**, where `n` is the size of the vectors. Therefore, the total complexity of `angle_cos` is also  **O(n)**.

---
<a id="ex06"></a>
<table>
<tr><th>Exercise 06 - Cross product </th></tr>
<tr><td>Allowed mathematical functions : fused multiply-add function </tr>
<tr><td>Maximum time complexity : N/A </tr>
<tr><td>Maximum space complexity : N/A</tr>
</table>
    
```python
def cross_product(u, v):
	if not (u.size == 3 and u.size == v.size):
		raise TypeError("Invalid input: it requires two-3dimensional Vectors.")
	x1, y1, z1 = u.tolist()
	x2, y2, z2 = v.tolist()

	cross_x = (y1 * z2 - y2 * z1)
	cross_y = (z1 * x2 - z2 * x1)
	cross_z = (x1 * y2 - x2 * y1)
	return Vector([[cross_x, cross_y, cross_z]])
```
The `cross_product` function calculates the cross product between two 3D vectors `u` and `v`. The cross product is an orthogonal(perpendicular) vector to the two given vectors.

#### How it works:

1. It checks if the two vectors `u` and `v` are three-dimensional (of size 3), otherwise, it raises an exception.
2. It extracts the x, y, and z coordinates of the two vectors using the `tolist` method.
3. It calculates the x, y, and z components of the cross product using the formulas:
	-   cross_x = y1 * z2 - y2 * z1
	-   cross_y = z1 * x2 - z2 * x1
	-   cross_z = x1 * y2 - x2 * y1
4. It creates a new vector with the calculated x, y, and z components and returns it.

---
<a id="ex07"></a>
<table>
<tr><th>Exercise 07 - Linear map, Matrix multiplication </th></tr>
<tr><td>Allowed mathematical functions : fused multiply-add function </tr>
<tr><td>Maximum time complexity : see below </tr>
<tr><td>Maximum space complexity : see below </tr>
</table>

```python
def mul_vec(self, other):
	if isinstance(other, Vector):
		if self.shape[1] != other.size:
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
```
Both of these functions are implementations of matrix and vector multiplication.

#### How it works:

The `mul_vec` function multiplies a matrix (self) by a vector (other)

1. It first checks if `other` is an instance of the `Vector` class.
2. Then, it checks if the dimensions of the matrix and of the vector match:
	- The number of the column of the matrix must be equal to the size of the vector.
	- If the dimensions don't match, an exception is raised.
3. The matrix product is calculated using a nested loop which iterates over each element of the matrix and of the vector to calculate the sum product of each element of the matrix multiplied by the corresponding element of the vector.
4. The function returns a result vector.

The `mul_mat` function calculates the multiplication of two matrices.

1. It checks if `other` is an instance of the `Matrix` class.
2. Then, it checks if the dimensions of the first  `self` matrix match the dimensions of the second `other` matrix.
3. The matrix product is calculated using two nested loops to iterate over the elements of two matrices and calculate the sum product of each row element of the first matrix multiplied by the corresponding element of the second matrix.
4. The function returns a result matrix.

#### Complexity:

The complexity of these functions is **O(n*m*p)**, where `n`, `m`, and `p` are the dimensions of the input matrices and vectors.

The `mul_vec` function has a complexity of  **O(n*m)** because it loops for each row of the `self` matrix (`n` loops) and for each column of the `other` vector (`m` loops) to perform the corresponding dot product.

The `mul_mat` function has a complexity of  **O(n*m*p)** because it loops for each row of the `self` matrix (`n` loops) and for each column of the `other` matrix (`p` loops) and for each column of the `self` matrix and for each row of the `other` matrix (`m` loops) to perform the corresponding dot product.

The space complexity of the `mul_mat` function is  **O(n*p)**, where `n` is the number of rows of the `self` matrix and `p` is the number of columns of the `other` matrix.


---
<a id="ex08"></a>
<table>
<tr><th>Exercise 08 - Trace </th></tr>
<tr><td>Allowed mathematical functions : None </tr>
<tr><td>Maximum time complexity :  O(n) </tr>
<tr><td>Maximum space complexity : N/A</tr>
</table>
    
```python
def trace(self):
	if self.shape[0] != self.shape[1]:
		raise TypeError("Trace is undefined for non-square matrices.")
	trace = 0.0
	for i in range(self.shape[0]):
		trace += self.data[i][i]
	return trace
```
The `trace` function calculates the trace of a square matrix which is the sum of the diagonal elements of the matrix (the elements for which the row and column indices are identical)

#### How it works:

1. It checks if the `self` matrix is square (the number of rows must be equal to the number of columns), otherwise, it raises an exception.
2. It initializes a `trace` variable to 0 to store the sum of diagonal elements of the matrix.
3. It loops for each index `i` of the matrix and adds the diagonal element to the `trace` variable.
4. It returns the `trace` variable value.

#### Complexity:

The time complexity of this function is **O(n)**, where `n` is the number of rows (also the number of columns since the matrix is square).

---
<a id="ex09"></a>
<table>
<tr><th>Exercise 09 - Transpose </th></tr>
<tr><td>Allowed mathematical functions : None </tr>
<tr><td>Maximum time complexity : O(nm) </tr>
<tr><td>Maximum space complexity : O(nm) </tr>
</table>
    
```python
def T(self):
	transposed = []
	for j in range(self.shape[1]):
		row = []
		for i in range(self.shape[0]):
			row.append(self.data[i][j])
		transposed.append(row)
	return Matrix(transposed)
```

The `T` function calculates the transpose of a matrix. The transpose of a matrix is obtained by swapping its rows and columns.

#### How it works:

1. It initializes an empty list `transposed` which will store the data of the transposed matrix.
2. It performs a `j` loop for each column of the matrix.
	- It creates a new empty list `row` to store the data of the new row of the transposed matrix.
3. It performs an `i` loop for each row of the matrix.
	- It adds the element `self.data[i][j]` to the `row` list.
4. It adds the `row` list to the `transposed` list.
5. It returns the new instance of the `Matrix` class containing the data of the transposed matrix.

#### Complexity:

The time complexity of this function is  **O(n*m)**, where `n` is the number of rows of the initial matrix and `m` is the number of columns.
The space complexity is also **O(n*m)** because it creates a new matrix to store the transpose result.

---
<a id="ex10"></a>
<table>
<tr><th>Exercise 10 - Reduced row-echelon form </th></tr>
<tr><td>Allowed mathematical functions : None </tr>
<tr><td>Maximum time complexity : O(n^3) </tr>
<tr><td>Maximum space complexity : O(n^3) </tr>
</table>
    
```python
def row_echelon(self):
	# Gaussian elimination with back-substitution for reduced row echelon from
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
```
The `row_echelon` function performs the transformation of the matrix into a reduced row echelon form, using the Gaussian elimination with the backward substitution.

#### How it works:

1. It initializes a `pivot` variable to 0.
2. It loops for each row of the matrix (`row` variable).
	- If the `pivot` is greater than or equal to the width of the matrix, the function breaks the loop.
3. It looks for a non-zero `pivot` element in the current row.
	- If `pivot` reaches the width of the matrix, the function returns the transformed matrix so far.
4. It swaps the current row with a row containing a non-zero `pivot` element.
5. It scales the current row so that the `pivot` element is equal to 1.
6. It performs row operations to eliminate the other non-zero elements in the current column.
7. It increments `pivot`.
8. It returns the matrix in reduced row echelon form.

#### Complexity:

The overall time complexity of this function is approximately **O(n^3)** for an `n` x `n` matrix.

---
<a id="ex11"></a>
<table>
<tr><th>Exercise 11 - Determinant </th></tr>
<tr><td>Allowed mathematical functions : fused multiply-add function </tr>
<tr><td>Maximum time complexity : O(n^3) </tr>
<tr><td>Maximum space complexity : O(n^2) </tr>
</table>
    
```python
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
```
The `determinant` function uses the Gaussian elimination to transform the matrix into echelon form and calculate its determinant.

#### How it works:

1. It first checks if the matrix is square, otherwise it raises an exception.
2. It creates a copy of the input matrix to avoid modifying the original data.
3. The `det` variable is initialized to 1.0 which will be used to store the determinant.
4. The outer loop `for i in range(self.shape[0])` iterates over each row of the matrix.
5. For each row, the loop `for j in range(self.shape[0])` finds the pivot.
6. If necessary, it swaps the rows to place the selected pivot in the appropriate location. Then, it updates the `det` variable by multiplying it by `-1`, because swapping rows changes the sign of the determinant.
7. It divides the pivot row by the pivot element to normalize the pivot row. The `det` variable is multiplied by the pivot element before normalization.
8. The loop `for k in range(i + 1, self.shape[0])` eliminates the other non-zero elements in the same column by subtracting the appropriate multiples from the pivot row.
9. It returns the `det` variable as a determinant of the input matrix.

#### Complexity:

The complexity of this function is **O(n^3)**, where `n` is the number of rows (or the number of columns) of the matrix:
	- Finding the pivot and swapping rows has a complexity of **O(n)** because the loop iterates over the elements in the same size.
	- The scaling of the pivot row has a complexity of **O(n)** because the loop iterates over the elements of the row.
	- Eliminating the non-zero elements in the same column has a complexity of **O(n^2)** because the nested loops iterate over rows and elements in each row.

Combining the complexities, we get **O(n * (n + n + n^2)) = O(n^3)**

The space complexity of this function is **O(1)** because this function simply modifies a copy of the input matrix without creating new matrices.

---
<a id="ex12"></a>
<table>
<tr><th>Exercise 12 - Inverse </th></tr>
<tr><td>Allowed mathematical functions : fused multiply-add function </tr>
<tr><td>Maximum time complexity : O(n^3) </tr>
<tr><td>Maximum space complexity : O(n^2) </tr>
</table>
    
```python
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
```
The `inverse` function calculates the inverse of a square matrix.

#### How it works:

1. It checks if the matrix is square and if the determinant is equal to 0 for exceptions.
2. It creates an augmented matrix `[A|I]`, where `A` is the input matrix and `I` is an identity matrix of the same size as `A`. This is achieved by adding the columns of the identity matrix to each row.
3. It applies the Gaussian elimination to the augmented matrix to get the reduced echelon form using the `row_echelon` method.
4. Finally, it extracts the inverse matrix `[I|B]` from the reduced echelon form and returns it as a result.

#### Complexity:

The time complexity of this function is mainly dominated by the Gauss-Jordan elimination which has a time complexity of **O(n^3)**, where `n` is the number of rows of the matrix.

The space complexity of this function is **O(n^2)** because the augmented matrix supports space for both the input matrix and the identity matrix.

---
<a id="ex13"></a>
<table>
<tr><th>Exercise 13 - Rank </th></tr>
<tr><td>Allowed mathematical functions : None </tr>
<tr><td>Maximum time complexity : O(n^3) </tr>
<tr><td>Maximum space complexity : N/A </tr>
</table>
    
```python
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
```

The `rank` function calculates the rank of a matrix. The rank of a matrix is defined as the maximum number of linearly independent rows (or columns).

#### How it works:

1. It applies the Gauss-Jordan elimination to the matrix to get the reduced row echelon form using the `row_echelon` method.
2. It initializes the `rank` variable to 0, this variable counts the number of non-zero rows.
3. It iterates over each row of `self.data` and checks if at least one element in the row is non-zero with the `any` function. If so, the `rank` variable is incremented by 1.
4. Finally, it returns the `rank` variable.

#### Complexity:

The time complexity of this function is dominated by the Gauss-Jordan elimination which has a complexity of **O(n^3)** for a matrix of size `(n x n)`.

The space complexity is **O(n^2)** because we store a copy of the input matrix.

---
<a id="ex14"></a>
<table>
<tr><th>Exercise 14 - Bonus: Projection matrix </th></tr>
<tr><td>Allowed mathematical functions : tan, fused multiply-add function </tr>
<tr><td>Maximum time complexity : N/A </tr>
<tr><td>Maximum space complexity : N/A </tr>
</table>

#### Homogeneous coordinates 

Homogeneous coordinates are an alternative representation of points and vectors in a Euclidean space, unifying and simplifying geometric operations such as translation, rotation, and scaling, as well as integrating the notion of points at infinity.

#### Homogeneous coordinate transformation

Homogeneous coordinate transformation is a technique used in geometry, computer graphics, and robotics to perform geometric transformations such as translations, rotations, and scaling on objects using a single matrix.

In general, for a 2D space, homogeneous coordinates are obtained by adding a third component (generally equal to 1 for points and 0 for directional vectors) to Cartesian coordinates.

```python
def	projection(fov, ratio, near, far):
	projection_matrix = [[0.0] * 4 for _ in range(4)]

	projection_matrix[0][0] = 1 / (ratio * math.tan(fov / 2))
	projection_matrix[1][1] = -1 / (math.tan(fov / 2))
	projection_matrix[2][2] = (far) / (far - near) 
	projection_matrix[2][3] = (2 * far * near) / (far - near) 
	projection_matrix[3][2] = 1

	return projection_matrix
```

The `projection` function calculates the perspective projection matrix, which is used to transform the coordinates of a 3D object into a 2D space, usually for rendering purposes.

#### How it works:

1. It takes 4 parameters: field of view (`fov`), aspect ratio (`ratio`), near distance (`near`), and far distance (`far`).
2. It first creates an empty 4x4 matrix `projection_matrix` filled with zeros.
3. Then, it fills the specific elements of the matrix with the appropriate values using the given arguments.
	- `projection_matrix[0][0]`: this term determines the scale according to the X axis and is calculated using aspect ratio `ratio` and field of view `fov`.
	- `projection_matrix[1][1]`: this term determines the scale according to the Y axis and is calculated using field of view `fov`.
	- `projection_matrix[2][2]` and `projection_matrix[2][3]`: these terms are responsible for the transformation in depth (Z axis) and are calculated using the `near`, `far` distances.
	- `projection_matrix[3][2]`: this term is set to 1 to ensure homogenous coordinate transformation.
