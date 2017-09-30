#import csv, random, math, operator

'''
#Función que lee el archivo .csv
def loadDataset(filename, split, trainingSet, testSet):
	with open(filename, 'r') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		#dataset.pop(0) 	#Para eliminar el encabezado del archivo
		caracteristics = len(dataset[0])
		#print(caracteristics)
		#print(dataset)
		
	for x in range(len(dataset)):
		for y in range(int(caracteristics)-1):
			dataset[x][y] = float(dataset[x][y])
		if random.random() < split:
			trainingSet.append(dataset[x])
		else:
			testSet.append(dataset[x])

#Función de la distancia euclidiana
def euclideanaDistance(instancia1, instancia2, length):
	distance = 0
	for x in range(length):
		distance += math.pow((instancia1[x] - instancia2[x]), 2)
	return math.sqrt(distance)

#Función optener los vecinos
def getNeighbors(trainingSet, testInstance, k):
	distances = list()
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanaDistance(testInstance, trainingSet[x], 2)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = list()
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

#Defeinir la respuesta basada en los vecinos tomando votos
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

#Suma las predicciones correctas totales y devuelve la predicción como un porcentaje de las clasificaciones correctas
#Optener exactitud
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	accuracy = (correct/float(len(testSet))) * 100.0
	return accuracy
'''


'''
	#preparar los datos
	trainingSet = list()
	testSet = list()
	split = 0.67
	loadDataset('iris.data', split, trainingSet, testSet)
	print('Entrenamiento: ' + str(len(trainingSet)))
	print('Analisis: ' + str(len(testSet)))

	#Generar predicciones
	predictions = list()
	k = int(input("Ingrese el número de K: "))
	
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
		print('> predicted=' + str(result) + ', actual=' + str(testSet[x][-1]))
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + str(accuracy) + '%')
	'''







	# Establezca la semilla para el generador de números aleatorios
	np.random.seed(0)

	# Generar muestras
	X, y = make_circles(n_samples=500, factor=0.2, noise=0.04)

	# Perform PCA
	pca = PCA()
	X_pca = pca.fit_transform(X)

	# Realizar Kernel PCA
	kernel_pca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
	X_kernel_pca = kernel_pca.fit_transform(X)
	X_inverse = kernel_pca.inverse_transform(X_kernel_pca)

	# Trazar los datos originales
	class_0 = np.where(y == 0)
	class_1 = np.where(y == 1)

	plt.figure()
	plt.plot(X_kernel_pca[class_0, 0], X_kernel_pca[class_0, 1], "ko", mfc='none')
	plt.plot(X_kernel_pca[class_1, 0], X_kernel_pca[class_1, 1], "kx")
	plt.title("Data transformed using Kernel PCA")
	plt.xlabel("Componentes principales 1")
	plt.ylabel("Componentes principales 2")

	plt.show()






	# Número de vecinos
	n_neighbors = 15

	# Cargar el archivo dataset
	dataset = loadDataset()
	#print(dataset)

	# sólo tomamos las dos primeras características. Podríamos evitar este feo
	# corte mediante el uso de un conjunto de datos de dos dimensiones
	X = dataset.ix[:,0:1].values #Selecciona las caracteristicas columna 0 a 1
	z = dataset.ix[:,4].values #Selecciona la clase columna 4

	# Lista de los target en números
	y = list()
	for row in z:
		if row == 'Iris-setosa':
			y.append(0)
		elif row == 'Iris-versicolor':
			y.append(1)
		else:
			y.append(2)
	#print(y)

	h = .02  # tamaño del paso en la malla

	# Crear mapas de color
	cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
	cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

	#for weights in ['uniform', 'distance']:
	for weights in ['uniform']:
		# creamos una instancia de Neighbors Classifier y ajustamos los datos.
		clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
		clf.fit(X, y)

		# Trazar el límite de decisión. Para ello, asignaremos un color a cada
		# punto en la malla [x_min, x_max] x[y_min, y_max].
		x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
		y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
					np.arange(y_min, y_max, h))
		Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

		# Coloca el resultado en un gráfico de color
		Z = Z.reshape(xx.shape)
		plt.figure()
		plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

		# Trazar también los puntos de entrenamiento
		plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
					edgecolor='k', s=20)
		plt.xlim(xx.min(), xx.max())
		plt.ylim(yy.min(), yy.max())
		plt.title("3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights))

	plt.show()










	# Cargar el archivo dataset
	dataset = loadDataset()
	#print(dataset)

	# Se extrane las columnas del dataset
	dataset_columns = list()
	for x in dataset:
		dataset_columns.append(x)
	#print(dataset_columns)

	dataset.columns = dataset_columns
	#dataset.dropna(how="all", inplace=True)
	
	X = dataset.ix[:,0:3].values #Selecciona las caracteristicas columna 0 a 3
	y = dataset.ix[:,4].values #Selecciona la clase columna 4

	# Estandarización de los datos
	X_std = StandardScaler().fit_transform(X)
	#print(X_std)

	# Matriz de covarianzas
	cov_mat = np.cov(X_std.T)
	#print('Covariance matrix \n%s' %cov_mat)

	# Eigen vectores y eigen valores
	eig_vals, eig_vecs = np.linalg.eig(cov_mat)
	#print('Eigenvectors \n%s' %eig_vecs)
	#print('\nEigenvalues \n%s' %eig_vals)

	for ev in eig_vecs:
	    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
	#print('Everything ok!')

	# Hacer una lista de (eigenvalue, eigenvector) tuplas
	eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

	# Ordenar las (eigenvalue, eigenvector) tuplas de menor a mayor
	eig_pairs.sort(key=lambda x: x[0], reverse=True)

	# Confirme visualmente que la lista se clasifica correctamente mediante la disminución de eigenvalues
	#print('Eigenvalues in descending order:')
	#for i in eig_pairs:
		#print(i[0])

	# Matriz de proyección
	matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                      eig_pairs[1][1].reshape(4,1)))
	#print('Matrix W:\n', matrix_w)

	# Proyección en el espacio
	Y = X_std.dot(matrix_w)

	# Gráfica
	with plt.style.context('seaborn-whitegrid'):
		plt.figure(figsize=(6, 4))
		for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
							('blue', 'red', 'green')):
		    plt.scatter(Y[y==lab, 0],
						Y[y==lab, 1],
						label=lab,
						c=col)
		plt.xlabel('Componentes principales 1')
		plt.ylabel('Componentes principales 2')
		plt.legend(loc='lower center')
		plt.tight_layout()
	plt.show()






# Función que carga el archivo y pregunta si tiene encabezado
def loadDataset():
	header = input("El archivo tiene encabezado? y/n: ")
	if header == 'y' or header == 'yes':
		header = 0
	else:
		header = None
	dataset = pd.read_csv(filename, header=header)
	return dataset