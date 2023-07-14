#### [Read this in English](README.md)

## La Structure du projet
- matrix.py

### Classe Matrix, Vector

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

#### Les méthodes de `Matrix`
```python
# add : only matrices of same dimensions.
def __add__(self, other):
	if not isinstance(other, Matrix):
		raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))
	if self.shape != other.shape:
		raise ValueError(f"Invalid input: addition requires a Matrix of same shape.")
	result = [[self.data[i][j] + other.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
	return Matrix(result)

# sub : only matrices of same dimensions.
def __sub__(self, other):
	if not isinstance(other, Matrix):
		raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))
	if self.shape != other.shape:
		raise ValueError(f"Invalid input: subtraction requires a Matrix of same shape.")
	result = [[self.data[i][j] - other.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
	return Matrix(result)

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
```
#### Comment elles fonctionnent:

1.  `__add__(self, other)`  : additionne deux objets  `Matrix`  ayant la même forme.
2.  `__sub__(self, other)`  : soustrait un objet  `Matrix`  par un autre ayant la même forme.
3.  `__truediv__(self, scalar)`  : divise les éléments d'un objet  `Matrix`  par un scalaire.
4.  `__mul__(self, other)`  : effectue une multiplication entre un objet  `Matrix`  et un scalaire, un objet  `Matrix`, un objet  `Vector`.

#### Les méthodes de `Vecotr`
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
#### Comment elles fonctionnent:

1.  `__add__(self, other)`  : additionne deux objets  `Vector`  ayant la même forme.
2.  `__sub__(self, other)`  : soustrait un objet  `Vector`  par un autre ayant la même forme.
3.  `__mul__(self, other)`  : effectue une multiplication entre deux objets  `Vector`, un objet  `Vector`  et un objet  `Matrix`, ou un objet  `Vector`  et un scalaire.
4.  `__truediv__(self, scalar)`  : divise les éléments d'un objet  `Vector`  par un scalaire.

#### Complexité:

Les complexités temporelles et spatiales de toutes les méthodes sont **O(n)**, où `n` est la taille des objets `Vector` et `Matrix`, car elles itèrent sur chaque élément des objets `Vector` et `Matric`.

---
<a id="ex01"></a>
<table>
<tr><th>Exercise 01 -  Linear combination </th></tr>
<tr><td>Allowed mathematical functions : fused multiply-add function </tr>
<tr><td>Maximum time complexity : O(n) </tr>
<tr><td>Maximum space complexity : O(n)</tr>
</table>


#### La combinaison linéaire

Une combinaison linéaire est une opération mathématique effectuée sur un ensemble d'éléments en les multipliant chacun par un coefficient approprié, puis en additionnant les résultats. 

Plus précisément, dans le contexte d'un espace vectoriel, une combinaison linéaire est obtenue en multipliant chaque vecteur de base par un coefficient approprié et en additionnant les produits.

```python
def linear_combination(lst_vectors, coefs):
	if not all(isinstance(lst, list) for lst in [lst_vectors, coefs]):		
		raise ValueError("Invalid form of list")
	if not all(isinstance(v, Vector) for v in lst_vectors):
		raise TypeError("Invalid input: list should contain only Vectors.", lst_vectors)
	if not all(v.size == lst_vectors[0].size for v in lst_vectors):
		raise TypeError("Invalid input: list of Vectors should contain Vectors of the same shape.", lst_vectors)
	if len(coefs) != len(lst_vectors) or not all(type(i) in [int, float] for i in coefs):
		raise TypeError("Invalid input: unsupported type or uncompatiable length with list of Vectors", coefs)
                        
	v = Vector([[0.0] * lst_vectors[0].size])
	for vector, coef in zip(lst_vectors, coefs):
		v += vector * coef
	return v
```
La fonction `linear_combination` prend en entrée une liste de vecteurs `lst_vecteors` et une liste de coefficients `coefs`. Elle effectue une combinaison linéaire des vecteurs en multipliant chaque vecteur par son coefficient correspondant, puis en les additionnant pour obtenir le vecteur résultant. Le résultat est renvoyé sous la forme d'un nouveau vecteur.

#### Comment elle fonctionne:

1. La fonction effectue d'abord plusieurs vérifications pour s'assurer que les entrées sont valides. Elle vérifie si `lst_vectors` et `coefs` sont tous deux des listes, si tous les éléments de `lst_vectors` sont des objets de la classe `Vector` et si les vecteurs ont la même taille et si les coefficients sont du bon type et de la bonne longueur.
2. La taille des vecteurs est extraite à partir du premier du vecteur dans `lst_vectors`.
3. Un vecteur initial `v` est créé avec des composantes initialisées à zéro. La taille du vecteur est déterminée par la taille des vecteurs d'entrée.
4. En utilisant la fonction `zip`, la boucle `for` itère simultanément sur chaque vecteur dans `lst_vectors` et chaque coefficient correspondant dans `coefs`.
5. À chaque itération de la boucle, le produit de chaque composante du vecteur et le coefficient correspondant sont calculés et ajoutés aux composantes correspondantes du vecteur `v`.
6. Une fois que toutes les combinaisons linéaire ont été effectuées, le vecteur résultant `v` est renvoyé.

#### Complexité

La complexité temporelle de cette fonction est **O(n)**, où `n` est le nombre de vecteurs dans `lst_vectors`.

La complexité spatiale est également  linéaire **O(n)**, car la fonction crée un nouveau vecteur `v` qui stocke les résultats intermédiaires et a une taile égale à la taille des vecteurs d'entrée

Cette fonction utilise efficacement la fonction `zip` pour itérer sur les vecteurs et les coefficients en parallèle, ce qui permet d'éviter l'utilisation de boucles imbriquées et de réduire ainsi la complexité temporelle à **O(n)**.


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
		raise TypeError("Invalid input: uncompatiable type") 
	if not (isinstance(t, float) and (0 <= t <= 1)): 
		raise ValueError("Invalid value: a real number from 0 to 1 required.", t) 
	if any(isinstance(u, accepted_type) for accepted_type in [int, float, complex, Vector, Matrix]):
		return u + (v - u) * t
	else: 
		raise TypeError("Invalid input: unsupported type")
```
La fonction `lerp` effectue une interpolation linéaire entre deux valeurs `u` et `v` en utilisant un facteur de mélange `t`.

#### Comment elle fonctionne:

1. Elle vérifie d'abord si si les types de `u` et `v` sont compatibles et lève une exception `TypeError` si ce n'est pas le cas.
2. Elle vérifie si `t` est un nombre réel entre 0 et 1 inclus et lève une exception `ValueError` si ce n'est pas le cas.
3. Ensuite, si le type de `u` est de type `int`, `float`, `complex`, `Vector`, `Matrix`, l'interpolation linéaire est calculée à l'aide de la formule `u + (v - u) * t` où `v - u` représente la différence entre `v` et `u`.
4. Le résultat de l'interpolation est renvoyé.

#### Complexité

La complexité temporelle de cette fonction dépend du type de `u` et `v`. 


---
<a id="ex03"></a>
<table>
<tr><th>Exercise 03 - Dot product </th></tr>
<tr><td>Allowed mathematical functions : fused multiply-add function </tr>
<tr><td>Maximum time complexity : O(n) </tr>
<tr><td>Maximum space complexity : O(n)</tr>
</table>

#### Le produit scalaire

Le produit scalaire est une opération mathématique effectuée entre deux vecteurs. It s'agit de la somme des produits des composantes correspondantes des deux vecteurs.

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

#### Comment elle fonctionne:

1. Elle vérifie si l'objet `other` est une instance de la classe `Vector`, si ce n'est pas le cas, elle lève une exception de type `TypeError`.
2. Elle vérifie si les deux vectuers ont la même taille, si ce n'est pas le cas, elle lève une exception de type `TypeError`.
3. Elle initialise une variable `dot_product` à 0 pour stocker le résultat du produit scalaire.
4. Elle parcourt les éléments des deux vecteurs à l'aide d'une boucle `for`, en multipliant les éléments correspondants et en ajoutant le produit à la variable `dot_product`.
5. Elle retourne le résultat final dans `dot_product`.


#### Complexité

La complexité de cette fonction est **O(n)**, où `n` est la taille des vecteurs, car la fonction parcourt chaque élément des deux vecteurs une fois.

---
<a id="ex04"></a>
<table>
<tr><th>Exercise 04 - Norm </th></tr>
<tr><td>Allowed mathematical functions : fused multiply-add function, pow, max </tr>
<tr><td>Maximum time complexity : O(n) </tr>
<tr><td>Maximum space complexity : O(n)</tr>
</table>

#### La norme L1, L2, optimale

Elles sont différentes mesures de la taille d'un vecteur.

1. La norme L1 (aussi appelée norme `Manhattan` ou `taxicab`) d'un vecteur. La norme L1 est la somme des valeurs absolues des éléments du vecteur. 
2. La norme L2 (aussi appelée norme `Euclidienne`) d'un vecteur. La norme L2 est la racine carrée de la somme des carrés des éléments du vacteur.
3. La norme optimale (aussi appelée nomre `Maximum`) d'un vecteur. La norme optimale est la valeur absolue maximale des éléments du vecteur.
    
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
La fonction `norm_1` calcule la norme L1 d'un vecteur.
La fonction `norm` calcule la norme L2 d'un vecteur.
La fonction `norm_inf` calcule la norme optimale d'un vecteur.

#### Comment elle fonctionne:

La fonction `norm_1` parcourt chaque élément et ajoute sa valeur absolue (positif ou négatif) à une somme et retourne la somme totale à la fin.

La fonction `norm` parcourt chaque élément du vecteur, ajoute le carré de sa valeur à une somme, puis retourne la racine carrée de la somme totale à la fin.

La fonction `norm_inf` parcourt chaque élément du vecteur, compare sa valeur absolue à une valeur maximale, et met à jour la valeur maximale si la valeur absolue courante est supérieuse. Elle retourne la valeur absolue maximale à la fin.

#### Complexité

La complexité de ces trois fonctions est **O(n)**, où `n` est le nombre d'éléments dans le vacteur.

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
La fonction `angle_cos` calcule la similarité cosinus entre deux vecteur `u` et `v`.
La similarité cosinus mesure l'angle entre les deux vecteurs, et elle est utilisée comme une mesure de similarité entre eux - des vecteurs similaires auront une similarité de consinus proche de 0.

#### Comment elle fonctionne:

1. Elle vérifie si `u` et `v` sont tous les duex des instances de la classe `Vector`. Si ce n'est pas le cas, elle lève une exception de type `TypeError`.
2. Elle vérifie si les deux vecteurs ont la même taile. Si ce n'est pas le cas, elle lève une exception de type `TypeError`.
3. Elle calcule la similarité cosinus à l'aide de cette fonction: $(u ⋅ v) / (||u|| * ||v||)$, où $(u ⋅ v)$ désigne le produit scalaire entre les vecteurs `u` et `v`, et $||u||$ et $||v||$ répresentent les normes (Eucilidiennes) respectives de `u` et `v`.
4. Elle arrondit la similarité de cosinus à 10 décimales et retourne le résultat.

#### Complexité

La complexité de cette fonction dépend des méthodes `dot` et `norm` qui sont toutes deux **O(n)**, où `n` est la taille des vecteurs. Par conséquent, la complexité totale de `angle_cos` est également **O(n)**.

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
La fonction `cross_product` calcule le produit vectoriel entre deux vecteurs 3D `u`et `v`.
Le produit vectoriel est un vecteur orthogonal (perpendiculaire) aux deux vecteurs donnés.


#### Comment elle fonctionne:

1. Elle vérifie si les deux vectuers `u` et `v` sont tridimensionnels (de taille 3), sinon elle lève une exception.
2. Elle extrait les coordonnées x, y et z des deux vecteurs à l'aide la méthode `tolist`.
3. Elle calcule les composantes x, y et z du produit vectoriel en utilisant les formules:
	-   cross_x = y1 * z2 - y2 * z1
	-   cross_y = z1 * x2 - z2 * x1
	-   cross_z = x1 * y2 - x2 * y1
4. Elle crée un nouveau vecteur avec les composantes x, y et z calculées et le retourne.

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
Ces deux fonctions sont des implémentations de la multiplication de matrices et de vecteurs.

#### Comment elle fonctionne:

La fonction `mul_vec` multiplie une matrice (self) par un vecteur (other)

1. Elle vérifie d'abord si `other` est une instance de la classe `Vector`.
2. Ensuite, elle vérifie si les dimensions de la matrice et du vecteur sont correspondent:
	- Le nombre de la colonne de la matrice doit être égal à la taille du vecteur.
	- Si les dimensions ne correspondent pas, une exception est levée.
3. Le produit matriciel est calculé à l'aide d'une boucle imbriquée qui itère sur chaque élément de la matrice et du vecteur pour calculer la somme produit de chaque élément de la matrice multiplié par l'élément correspondant du vectuer.
4. La fonction retourne un vecteur résultant.

La fonction `mul_mat` calcule la multiplication de deux matrices.

1. Elle vérifile si `other` est une instance de la classe `Matrix`.
2. Ensuite, elle vérifie si les dimensions de la première matrice `self` correspondent aux dimensions de la seconde matrice `other`.
3. Le produit matriciel est calculé à l'aide de deux boucles imbriquées pour iterer sur les éléments de deux matrices et calculer la somme produit de chaque élément de ligne de la première matrice multiplié par l'élément correspondant de la seconde matrice.
4. La fonction retourne une matrice résultante.

#### Complexité

La complexité de ces fonctions est **O(n*m*p)**, où `n`, `m` et `p` sont les dimensions des matrices et vecteurs en entrée.

La fonction `mul_vec` a une complexité de **O(n*m)**, car elle effectue une boucle pour chaque ligne de la matrice `self` (`n` boucles) et pour chaque colonne du vecteur `other` (`m` boucles) pour effectuer le produit scalaire correspondant.

La fonction `mul_mat` a une complexité de **O(n*m*p)**, car elle effectue une boucle pour chaque linge de la matrice `self` (`n` boucles) et pour chaque colonne de la matrice `other` (`p` boucles) et pour chaque colonne de la matrice `self` et pour chaque ligne de la matrice `other` (`m` boucles) pour effectuer le produit scalaire correspondant.

La complexité spatiale de la fonction `mul_mat` est **O(n*p)**, où `n` est le nombre de lignes de la matrice `self` et `p` est le nombre de colonnes de la matrice `other`.



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
La fonction `trace` calcule la trace d'une matrice carrée qui est la somme des élément diagonaux de la matrice (les éléments pour lesquels les indices de ligne et de colonne sont identiques)

#### Comment elle fonctionne:

1. Elle vérifie si la matrice `self` est carrée (le nombre de linges doit être égal au nombre de colonnes), sinon elle lève une exception.
2. Elle initialise une variable `trace` à 0 pour stocker la somme des éléments diagonaux de la matrice.
3. Elle effectue une boucle pour chaque indice `i` de la matrice et ajoute l'élément diagonal à la variable `trace`.
4. Elle retourne la valeur de la variable `trace`.

#### Complexité

La complexité temporelle de cette fonction est **O(n)**, où `n` est le nombre de linges (également le nombre de colonnes puisque la matrice est carrée).

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
La fonction `T` calcule la transposée d'une matrice. La transposée d'une matrice est obtenue en échangeant ses lignes et ses colonnes.

#### Comment elle fonctionne:

1. Elle initialise une liste vide `transposed` qui stockera les données de la matrice transposée.
2. Elle effectue une boucle `j` pour chaque colonne de la matrice.
	- Elle crée une nouvelle liste vide `row` pour stocker les données de la nouvelle ligne de la matrice transposée.
3. Elle effectue une boucle `i` pour chaque ligne de la matrice.
	- Elle ajoute l'élément `self.data[i][j]` à la liste `row`.
4. Elle ajoute la liste `row` à la liste `transposed`.
5. Elle retourne la nouvelle instance de la classe `Matrix` contenant les données de la matrice transposée.

#### Complexité

La complexité temporelle de cette fonction est **O(n*m)**, où `n` est le nombre de lignes de la matrice initiale et `m` est le nombre de colonnes.
La compexité spatiale est également **O(n*m)**, car elle crée une nouvelle matrice pour stocker le résultat de la transposition.

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
La fonction `row_echelon` réalise la transformation de la matrice en forme échelonnée réduite, en utilisant l'élimination de Gauss avec la substitution arrière. 

#### Comment elle fonctionne:

1. Elle initialise une variable `pivot` à 0.
2. Elle effectue une boucle pour chaque rangée de la matrice (variable `row`).
	- Si `pivot` est supérieur ou égal à la largeur de la matrice, la fonction interrompt la boucle.
3. Elle cherche un élément `pivot` non nul dans la rangée actuelle.
	- Si `pivot` atteint la largeur de la matrice, la fonction retourne la matrice transformée jusqu'à présent.
4. Elle échange la rangée actuelle avec une rangée contenant un élément `pivot` non nul.
5. Elle met à l'echelle la rangée actuelle pour que l'élément `pivot` soit égal à 1.
6. Elle effectue les opérations sur les rangées pour éliminer les autres éléments non nuls dans la colonne actuelle.
7. Elle incrémente `pivot`.
8. Elle retourne la matrice sous forme échelonnée réduite.

#### Complexité

La complexité temporelle globale de cette fonction est approximativement **O(n^3)** pour une matrice `n` x `n`.

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
La fonction `determinant` utilise l'élimination de Gauss pour transformer la matrice en forme échelonnée et calculer son déterminant.

#### Comment elle fonctionne:

1. Elle vérifie d'abord si la matrice est carrée, sinon elle lève une exception.
2. Elle crée une copie de la matrice d'entrée pour éviter de modifier les données originales.
3. La variable `det` est initialisée à 1.0 qui sera utilisée pour stocker le déterminant.
4. La boucle externe `for i in range(self.shape[0])` itère sur chaque ligne de la matrice.
5. Pour chaque ligne, la boucle `for j in range(self.shape[0])` trouve le pivot.
6. Si nécessaire, elle échange les lignes pour placer le pivot sélectionné à l'emplacement approprié. Ensuite, elle met à jour la variable `det` en la multipliant par `-1`, car l'échange des linges change le signe du déterminant.
7.  Elle divise la ligne pivot par l'élément pivot pour normaliser la ligne pivot. La variable `det` est multipliée par l'élément pivot avant la normalisation.
8. La boucle `for k in range(i + 1, self.shape[0])` élimine les autres éléments non nuls de la même colonne en soustrayant les multiples appropriés de la ligne pivot.
9. Elle retourne la variable `det` comme déterminant de la matrice d'entrée.

#### Complexité

La complexité de cette fonction est **O(n^3)**, où `n` est le nombre de lignes (ou le nombre de colonnes) de la matrice:
- La recherche du pivot et l'échange des lignes ont une complexité de **O(n)**, car la boucle itère sur les éléments dans la même colonne.
- La mise à l'échelle de la ligne pivot a une complexité de **O(n)**, car la boucle itère sur les éléments de la ligne.
- L'élimination des éléments non nuls dans la même colonne a une complexité de  **O(n^2)**, car les boucles imbriquées itèrent sur les lignes et les éléments de chaque ligne.

En combinant les complexités, on obtient **O(n * (n + n + n^2)) = O(n^3)**

La complexité spatiale de cette fonction est **O(1)**, car cette fonction modifie simplement une copie de la matrice d'entrée sans créer de nouvelles matrices.


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
La fonction `inverse` calcule l'inverse d'une matrice carrée.

#### Comment elle fonctionne:

1. Elle vérifie si la matrice est carrée et si le déterminant égal à 0 pour les exceptions.
2. Elle crée une matrice augmentée `[A|I]`, où `A` est la matrice d'entrée et `I` est une matrice d'identité de la même taille que `A`. Cela est réalisé en ajoutant les colonnes de la matrice d'identité à chaque ligne.
3. Elle applique l'élimination de Gauss à la matrice augmentée pour obtenir la forme échelonnée réduite en utilisnat la méthode `row_echelon`.
4. Finalement, elle extrait la matrice inverse `[I|B]` à partir de la forme échelonnée réduite et la retourne comme résultat.

#### Complexité

La complexité temporelle de cette fonction est principalement dominée par l'élimination de Gauss-Jordan qui a une complexité temporelle de **O(n^3)**, où `n` est le nombre de lignes de la matrice.

La complexité spatiale de cette fonction est  **O(n^2)**, car la matrice augmentée prend en charge la fois l'espace pour la matrice d'entrée et la matrice d'identité.

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
La fonction `rank` calcule le rang d'une matrice. Le rang d'une matrice est défini comme le nombre maximum de lignes indépendantes linéairement (ou colonnes indépendantes linéairement).

#### Comment elle fonctionne:

1. Elle applique l'élimination de Gauss-Jordan à la matrice pour obtenir la forme échelonnée réduite en utilisant la méthode `row_echelon`.
2. Elle initialise la variable `rank` à 0, cette variable compte le nombre de lignes non nulles.
3. Elle itère sur chaque ligne de `self.data` et vérifie si au moins un élément de la ligne est différent de zéro avec la fonction `any`. Si c'est le cas, la variable `rank` est incrémentée de 1.
4. Finalement, elle retourne la variable `rank`.

#### Complexité

La complexité temporelle de cette fonction est dominée par l'élimination de Gauss-Jordan qui a une complexité de **O(n^3)** pour une matrice de taille `(n x n)`.

La complexité spatiale est  **O(n^2)**, car on stocke une copie de la matrice d'entrée.

---
<a id="ex14"></a>
<table>
<tr><th>Exercise 14 - Bonus: Projection matrix </th></tr>
<tr><td>Allowed mathematical functions : tan, fused multiply-add function </tr>
<tr><td>Maximum time complexity : N/A </tr>
<tr><td>Maximum space complexity : N/A </tr>
</table>

#### Les coordonnées homogènes

Les coordonnées homogènes sont une représentation alternative des points et des vecteurs dans un espace euclidien, permettant d'unifier et de simplifier les opérations géométriques telles que la translation, la rotation et la mise à l'echelle, ainsi que d'intégrer la notion de points à l'infini.

#### La transformation homogène de coordonnées.

La transformation homogène de coordonnées est une technique utilisée en géométrie, en infographie et en robotique pour effectuer des transformations géométriques telles que les translations, les rotations et les mises à l'echelle sur des objets en utilisant une seule matrice.

En général, pour un espace 2D, les coordonnées homogènes sont obtenues en ajoutant une troisième composante (généralement égale à 1 pour les points et à 0 pour les vecteurs directionnels) aux coordonnées cartésiennes.

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
La fonction `projection` calcule la matrice de projection perspective, qui est utilisée pour transformer les coordonnées d'un objet 3D dans un espase 2D, généralement à des fins de rendu.

#### Comment elle fonctionne:

1. Elle prend 4 parametères: champ de vision (`fov`), rapport d'aspect (`ratio`), distance proche (`near`), distance loinaine (`far`).
1. Elle crée d'abord une matrice 4x4 vide `projection_matrix` remplie de zéros.
2. Ensuite, elle remplit les éléments spécifiques de la matrice avec les valeurs appropriées en utilisant les arguments donnés.
	- `projection_matrix[0][0]`: ce terme détermine l'echelle en fonction de l'axe X et est calculé en utilisant le rapport d'aspect `ratio` et le champ de vision `fov`.
	- `projection_matrix[1][1]`: ce terme détermine l'echelle en fonction de l'axe Y et est calculé en utilisant le champ de vision `fov`.
	- `projection_matrix[2][2]` et `projection_matrix[2][3]`: ces terme sont responsables de la transformation en profondeur (axe Z) et sont calculés en utilisant les distances `near`, `far`.
	- `projection_matrix[3][2]`: ce terme est défini sur 1 pour assurer la transformation homogène de coordonnées.

