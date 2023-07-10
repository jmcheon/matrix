from matrix import Vector

def ex1():
	u = Vector([[0., 0.]]);
	v = Vector([[0.], [0.]]);
	print(u.dot(v)); # 0.0

	u = Vector([[1., 0.]]);
	v = Vector([[0.], [0.]]);
	print(u.dot(v)); # 0.0

	u = Vector([[1., 0.]]);
	v = Vector([[1.], [0.]]);
	print(u.dot(v)); # 1.0

	u = Vector([[1., 0.]]);
	v = Vector([[0.], [2.]]);
	print(u.dot(v)); # 0.0

	u = Vector([[1., 1.]]);
	v = Vector([[1.], [1.]]);
	print(u.dot(v)); # 2.0

	u = Vector([[4., 2.]]);
	v = Vector([[2.], [1.]]);
	print(u.dot(v)); # 10.0

def complex_ex():
	u = Vector([[0j, 0j]]);
	v = Vector([[0j], [0j]]);
	print(u.dot(v)); # 0j

	u = Vector([[1j, 0j]]);
	v = Vector([[0j], [0j]]);
	print(u.dot(v)); # 0j

	u = Vector([[1j, 0j]]);
	v = Vector([[1j], [0j]]);
	print(u.dot(v)); # -1+0j

	u = Vector([[1j, 0j]]);
	v = Vector([[0j], [2j]]);
	print(u.dot(v)); # 0j

	u = Vector([[1j, 1j]]);
	v = Vector([[1j], [1j]]);
	print(u.dot(v)); # -2+0j

	u = Vector([[4j, 2j]]);
	v = Vector([[2j], [1j]]);
	print(u.dot(v)); # -10+0j

def main():
	u = Vector([[0., 0.]]);
	v = Vector([[1.], [1.]]);
	print(u.dot(v)); # 0.0

	u = Vector([[1., 1.]]);
	v = Vector([[1.], [1.]]);
	print(u.dot(v)); # 2.0

	u = Vector([[-1., 6.]]);
	v = Vector([[3.], [2.]]);
	print(u.dot(v)); # 9.0

if __name__ == "__main__":
	ex1()
	complex_ex()
