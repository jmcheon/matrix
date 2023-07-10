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

def complex_ex():
	print("\nComplex number examples.")
	print(Vector.lerp(-1 + 0j, 1 + 1j, 0.)); # -1+0j
	print(Vector.lerp(-1 + 0j, 1 + 1j, 1.)); # 1+1j
	print(Vector.lerp(2.2 + 0j, 3.5 + 42j, 0.5)); # 2.85+21j
	print(Vector.lerp(-42j, 42j, 0.5)); # 0j
	print(Vector.lerp(0j, 1j, 0.5)); # 0.5j
	print(Vector.lerp(21j, 42j, 0.3)); # 27.3j
	print(Vector.lerp(Vector([[-42j], [42j]]), Vector([[42j], [-42j]]), 0.5)); # [0j][0j]
	print(Vector.lerp(Vector([[4 - 2j], [-2 + 1j]]), Vector([[4 - 4j], [1 + 2j]]), 0.3)); # [4-2.6j][-1.1+1.3j]
	print(Vector.lerp(Matrix([[2j, 1j], [3j, 4j]]), Matrix([[20j, 10j], [30j, 40j]]), 0.5)); # [[11j, 5.5j][16.5j, 22j]]

def errors():
	print(Vector.lerp(0., 1., 5)); # 0.0

if __name__ == "__main__":
	ex1()
	ex2()
	complex_ex()
