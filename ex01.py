from matrix import Vector

def ex1():
	v1 = Vector([[-42., 42.]]);
	v2 = Vector([[-42.]]);
	v3 = Vector([[1., 3.]]);
	v4 = Vector([[10., 20.]]);
	v5 = Vector([[-42., 100., -69.5]]);
	v6 = Vector([[1., 3., 5.]]);

	print(Vector.linear_combination([v1], [-1.])); # '[42., -42.]'
	print(Vector.linear_combination([v2, v2, v2], [-1., 1., 0.])); # '[0.]'
	print(Vector.linear_combination([v1, v3, v4], [1., -10., -1.])); # '[-62., -8.]'
	print(Vector.linear_combination([v5, v6], [1., -10.])); # '[-52., 70., -119.5]'

def main():
	e1 = Vector([[1., 0., 0.]]);
	e2 = Vector([[0., 1., 0.]]);
	e3 = Vector([[0., 0., 1.]]);
	v1 = Vector([[1., 2., 3.]]);
	v2 = Vector([[0., 10., -100.]]);
	print(Vector.linear_combination([e1, e2, e3], [10., -2., 0.5])); # [10.][-2.][0.5]
	print(Vector.linear_combination([v1, v2], [10., -2.])); # [10.][0.][230.]

def errors():
	v1 = Vector([[1., 2., 3.]]);
	v2 = Vector([[0., 10., -100.]]);
	#print(Vector.linear_combination([v1, v2], 10.));
	#print(Vector.linear_combination(v1, [10., -2.]));
	#print(Vector.linear_combination(v1, v2, [10., -2.]));

if __name__ == "__main__":
	ex1()
	#errors()
