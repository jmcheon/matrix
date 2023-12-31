from matrix import Matrix, Vector

def ex1():
	print(Vector.lerp(0., 1., 0.)); # 0.0
	print(Vector.lerp(0., 1., 1.)); # 1.0
	print(Vector.lerp(0., 42., 0.5)); # 21
	print(Vector.lerp(2.2, 3.5, 0.5)); # 2.85
	print(Vector.lerp(-42., 42., 0.5)); # 0
	print(Vector.lerp(0., 1., 0.5)); # 0.5
	print(Vector.lerp(21., 42., 0.3)); # 27.3

def ex2():
	print(Vector.lerp(Vector([[-42.], [42.]]), Vector([[42.], [-42.]]), 0.5)); # [0.0][0.0]
	#print(Vector.lerp(Vector([[2., 1.]]), Vector([[4., 2.]]), 0.3)); # [2.6][1.3]
	print(Vector.lerp(Vector([[2.], [1.]]), Vector([[4.], [2.]]), 0.3)); # [2.6][1.3]
	print(Vector.lerp(Matrix([[2., 1.], [3., 4.]]), Matrix([[20., 10.], [30., 40.]]), 0.5)); # [[11., 5.5][16.5, 22.]]

def errors():
	print(Vector.lerp(0., 1., 5)); # 0.0

if __name__ == "__main__":
	ex1()
	ex2()
