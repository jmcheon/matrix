from matrix import Matrix, Vector


def	projection(fov, ratio, near, far):
	projection_matrix = [[0.0] * 4 for _ in range(4)]

	return projection_matrix


if __name__ == "__main__":
	# field of view in degrees
	fov = 0.0
	# window size ratio (width / height)
	ratio = 0.0
	# distance of the near plane
	near = 0.0
	# distance of the far plane
	far = 0.0
	projection_matrix = projection(fov, ratio, near, far)
