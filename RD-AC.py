import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math, operator, random, copy, os
from numpy import linalg as la



# Función que retorna la variable con el nombre de la carpeta de dataset
def folder():
	carpeta = 'dataset/'
	return carpeta

# Función que retorna una lista con los archivos del dataset
def files(carpeta):
	archivos = os.listdir(carpeta)
	return archivos

# Función que selecciona el archivo
def seleccionar():
	print()
	archivos = files(folder())
	for x in range(len(archivos)):
		print(str(x+1) + ". " + archivos[x])
	nArchivo = int(input("Ingrese el número del dataset que desea seleccionar: "))

	while (nArchivo < 1 or nArchivo > len(archivos)):
		nArchivo = int(input("El número es incorrecto, debe estar entre " + str(1) + " y " + str(len(archivos)) + ": "))
	return archivos[nArchivo-1]

# Función que carga el archivo y pregunta si tiene encabezado
def loadDataset():
	header = input("\nEl archivo tiene encabezado? y/n: ")
	header = header.lower()
	if header == 'y' or header == 'yes' or header == 's' or  header == 'si':
		header = 0
	else:
		header = None
		print("El archivo debe de tener encabezado para determinar las caracteristicas y las clases")
		exit()
	dataset = pd.read_csv(str(folder()) + str(seleccionar()), header=header)
	return dataset

# Función que recibe que n individuo de la matriz normalizada va a devolver
def vector_registro_matrizNormalizada(n, matrizNormalizada, columns):
	registro_matrizNormalizada = list()
	for i in range(columns):
		valor = matrizNormalizada[n][i]
		registro_matrizNormalizada.append(valor)
	return registro_matrizNormalizada

# Función de la distancia euclidiana
def dist_euclidiana(v1, v2):
	dimension = len(v1)
	suma = 0
	for i in range(dimension):
		suma += math.pow(float(v1[i]) - float(v2[i]), 2)
	return math.sqrt(suma)

# Función de la distancia de manhattan
def dist_manhattan(v1, v2):
	dimension = len(v1)
	resutl = 0
	rest = 0
	for x in range(dimension):
		rest += (float(v1[x]) - float(v2[x]))
	result = abs(rest)
	return result

# Función que determina el numero menor de unal ista y devuelve su posición
def minimo(lista):
	dimension = len(lista)
	minimo = lista[0]
	posicion = 0
	for x in range(1, dimension):
		if (minimo > lista[x]):
			minimo = lista[x]
			posicion = x
	return posicion

# Lista de metricas de distancia
lista_metricas = ['Distancia euclidiana', 'Distancia de manhattan']

# Función que selecciona la metrica que se quiere
def seleccion_metrica():
	print()
	
	for x in range(len(lista_metricas)):
		print(str(x+1) + ". " + str(lista_metricas[x]))
	mOpcion = int(input("Seleccione la metrica que quiere utilizar: "))

	while (mOpcion < 1 or mOpcion > len(lista_metricas)):
		mOpcion = int(input("El número es incorrecto, debe estar entre " + str(1) + " y " + str(len(lista_metricas)) + ": "))
	return mOpcion

# Función que escoge la metrica que se quiere
def seleccion_metrica_distancia(v1, v2, mOpcion):
	if (mOpcion == 1):
		resultado = dist_euclidiana(v1, v2) # Distancia euclidiana
	elif (mOpcion == 2):
		resultado = dist_manhattan(v1, v2) # Distancia de manhattan
	return resultado


# ALGORITMOS DE ANÁLISIS DE COMPONENTES Y CLUSTERING
# =======================================================================================================================
# =======================================================================================================================


def ACP():
	print("Algoritmo de ACP")

	# Cargar el archivo dataset
	dataset = loadDataset()
	#print(dataset)
	#print(dataset.head())

	# Se extraen las columnas del dataset
	dataset.columns
	#print(dataset.columns)

	print("Descripción del dataset: ")
	print(dataset.describe()) # Descripción estadistica de los datos
	columns = len(dataset.columns) # Número total de columnas
	#print(columns)
	rows = len(dataset.index) # Número total de filas
	#print(rows)

	# Agrupando columnas por tipo de datos
	tipos = dataset.columns.to_series().groupby(dataset.dtypes).groups

	# Armando lista de columnas categóricas
	try:
		ctext = tipos[np.dtype('object')]
	except KeyError:
		ctext = list() # lista de columnas vacia en caso de que no haya categóricas
	print("\nNúmero de columnas categoricas: " + str(len(ctext))) # cantidad de columnas con datos categóricos

	# Armando lista de columnas numéricas
	columnas = dataset.columns  # lista total las columnas
	cnum = list(set(columnas) - set(ctext)) # Total de columnas menos columnas no numéricas
	print("Número de columnas numericas: " + str(len(cnum)))

	# Lista de medias
	#print(dataset.mean())
	'''media = list()
	for x in range(len(dataset.mean())):
		media.append(dataset.mean()[x])
	print(media)'''

	media = dataset.mean().values.tolist() # Convertir la media del dataset a una lista
	print("\nLista de medias: " + str(media))
	#print(media)

	# Lista de desviación estandar
	#print(dataset.std())
	'''destandar = list()
	for x in range(len(dataset.std())):
		destandar.append(dataset.std()[x])
	print(destandar)'''

	destandar = dataset.std().values.tolist() # Convertir la desviación estandar del dataset a una lista
	print("Lista de desviación estandar: " + str(destandar))
	#print(destandar)

	# Lista de caracteristicas
	lcaracteristicas = list()
	for x in range(len(cnum)):
		lcaracteristicas.append(columnas[x])
	print("Lista de características: " + str(lcaracteristicas))
	#print(lcaracteristicas)

	# Todas las filas y todas las columnas del dataset en una lista
	matriz = dataset.loc[:,].values
	print("\nMatriz:")
	print(matriz)

	# ===========================================================================================================================

	# Datos ajustados de la matriz
	matrizDatosAjustados = list()
	MDArow = list()
	for x in range(rows):
		for y in range(columns):
			result = (matriz[x][y] - media[y])
			MDArow.append(result)
		matrizDatosAjustados.append(MDArow)
		MDArow = list()
	print("\nMatriz de datos ajustados:")
	#print(matrizDatosAjustados)
	MDA = pd.DataFrame(np.array(matrizDatosAjustados)) # Matriz de datos ajustados con pandas
	print(MDA)

	# Transpuesta de la matriz de datos ajustados
	Transpuesta = np.array(matrizDatosAjustados) # Convertir la lista a formato numpy
	matrizDatosAjustadosT = Transpuesta.transpose().tolist() # Convertir la lista numpy con la transpuesta a formato normal de lista
	print("\nMatriz de datos ajustados transpuesta:")
	#print(matrizDatosAjustadosT)
	MDAT = pd.DataFrame(np.array(matrizDatosAjustadosT)) # Matriz de datos ajustados con pandas
	print(MDAT)

	# Matriz de covarianza (Transpuesta X Datos-Ajustados)
	matrizCovarianza = list()
	MCrow = list()
	suma = 0
	for i in range(len(matrizDatosAjustadosT)): # Filas de la matriz 1
		for j in range(columns): # Columnas de la matriz 2
			for k in range(rows): # Filas de la matriz 2
				suma += (matrizDatosAjustadosT[i][k] * matrizDatosAjustados[k][j])
			covarianza = (suma / (rows-1)) # Cada valor de la matriz de covarianza se debe dividir por n-1 (filas-1)
			MCrow.append(covarianza)
			suma = 0
		matrizCovarianza.append(MCrow)
		MCrow = list()
	print("\nMatriz de covarianza:")
	#print(matrizCovarianza)
	MC = pd.DataFrame(np.array(matrizCovarianza)) # Matriz de covarianza con pandas
	print(MC)

	# Valores propios y vectores propios
	evalues, evectors = la.eig(np.array(matrizCovarianza))
	print("\nValores propios:")
	eigenvalues = evalues.tolist()
	print(eigenvalues)
	print("\nVectores propios:")
	print(evectors)
	eigenvectors = evectors.tolist()
	#print(eigenvectors)

	# Suma de los valores propios
	sumaEigenvalues = 0
	for x in range(len(eigenvalues)):
		sumaEigenvalues += eigenvalues[x]
	print("\nSuma de los valores propios: " + str(sumaEigenvalues))

	# Importancia de las columnas en pocentajes
	porcentaje = list()
	for x in range(len(eigenvalues)):
		result = (eigenvalues[x] / sumaEigenvalues) * 100
		porcentaje.append(result)
	print("\nImportancia de las columnas en porcentaje: " + str(porcentaje))

	# Indices de los porcentajes
	indicesPorcentajes = list()
	for x in range(len(porcentaje)):
		indicesPorcentajes.append(x)

	# Diccionario de indices: porcentajes
	diccionarioPorcentaje = dict(zip(indicesPorcentajes, porcentaje))
	dp = sorted(diccionarioPorcentaje.items(), key=operator.itemgetter(1)) # Ordena el diccionario en tuplas por el valor (0 = clave, 1 = valor)
	print("\nDiccionario columna:porcentaje ordenado por porcentaje (descendente):")
	dp.reverse() # ordena descendente con .reverse()
	print(dp) # Imprime el diccionario en tuplas ordenado por valor

	# Nuevo conjunto de datos (Datos ajustados con la media X Matriz de valores propios)
	nuevoConjuntoDatos = list()
	NCDrow = list()
	suma = 0
	for i in range(len(matrizDatosAjustados)): # Filas de la matriz 1
		for j in range(len(eigenvectors)): # Columnas de la matriz 2
			for k in range(len(eigenvectors)): # Filas de la matriz 2
				suma += (matrizDatosAjustados[i][k] * eigenvectors[k][j])
			NCDrow.append(suma)
			suma = 0
		nuevoConjuntoDatos.append(NCDrow)
		NCDrow = list()
	print("\nMatriz de nuevo conjunto de datos:")
	#print(nuevoConjuntoDatos)
	NCD = pd.DataFrame(np.array(nuevoConjuntoDatos)) # Matriz de nuevo conjunto de datos con pandas
	print(NCD)

	# Guardar componentes en un archivo csv
	NCD.columns = lcaracteristicas
	name = str(input("\nIngrese el nombre de su archivo .csv: "))
	NCD.to_csv('dataset/' + str(name) + '.csv', header=True, sep=',', index=False)
	print("\nSe ha generado un archivo " + str(name) + ".csv con los componentes")

def ACPK():
	print("Algoritmo de ACPK")

	# Cargar el archivo dataset
	dataset = loadDataset()
	#print(dataset)
	#print(dataset.head())

	# Se extraen las columnas del dataset
	dataset.columns
	#print(dataset.columns)

	print("Descripción del dataset: ")
	print(dataset.describe()) # Descripción estadistica de los datos
	columns = len(dataset.columns) # Número total de columnas
	#print(columns)
	rows = len(dataset.index) # Número total de filas
	#print(rows)

	# Agrupando columnas por tipo de datos
	tipos = dataset.columns.to_series().groupby(dataset.dtypes).groups

	# Armando lista de columnas categóricas
	try:
		ctext = tipos[np.dtype('object')]
	except KeyError:
		ctext = list() # lista de columnas vacia en caso de que no haya categóricas
	print("\nNúmero de columnas categoricas: " + str(len(ctext))) # cantidad de columnas con datos categóricos

	# Armando lista de columnas numéricas
	columnas = dataset.columns  # lista total las columnas
	cnum = list(set(columnas) - set(ctext)) # Total de columnas menos columnas no numéricas
	print("Número de columnas numericas: " + str(len(cnum)))

	# Lista de medias
	#print(dataset.mean())
	'''media = list()
	for x in range(len(dataset.mean())):
		media.append(dataset.mean()[x])
	print(media)'''

	media = dataset.mean().values.tolist() # Convertir la media del dataset a una lista
	print("\nLista de medias: " + str(media))
	#print(media)

	# Lista de desviación estandar
	#print(dataset.std())
	'''destandar = list()
	for x in range(len(dataset.std())):
		destandar.append(dataset.std()[x])
	print(destandar)'''

	destandar = dataset.std().values.tolist() # Convertir la desviación estandar del dataset a una lista
	print("Lista de desviación estandar: " + str(destandar))
	#print(destandar)

	# Lista de caracteristicas
	lcaracteristicas = list()
	for x in range(len(cnum)):
		lcaracteristicas.append(columnas[x])
	print("Lista de características: " + str(lcaracteristicas))
	#print(lcaracteristicas)

	# Todas las filas y todas las columnas del dataset en una lista
	matriz = dataset.loc[:,].values
	print("\nMatriz:")
	print(matriz)

	# ===========================================================================================================================
	'''
	# Matriz en una lista
	matrizLista = matriz.tolist() # Convertir la matriz a una lista
	#print(matrizLista)

	# Transpuesta de la matriz de datos ajustados
	Transpuesta = np.array(matrizLista) # Convertir la lista a formato numpy
	matrizListaT = Transpuesta.transpose().tolist() # Convertir la lista numpy con la transpuesta a formato normal de lista
	print("\nMatriz transpuesta:")
	#print(matrizListaT)
	MLT = pd.DataFrame(np.array(matrizListaT)) # Matriz de datos ajustados con pandas
	print(MLT)

	# Matriz de covarianza (Transpuesta X Matriz)
	matrizCovarianza = list()
	MCrow = list()
	suma = 0
	for i in range(len(matrizListaT)): # Filas de la matriz 1
		for j in range(columns): # Columnas de la matriz 2
			for k in range(rows): # Filas de la matriz 2
				suma += (matrizListaT[i][k] * matrizLista[k][j])
			covarianza = (suma / (rows-1)) # Cada valor de la matriz de covarianza se debe dividir por n-1 (filas-1)
			MCrow.append(covarianza)
			suma = 0
		matrizCovarianza.append(MCrow)
		MCrow = list()
	print("\nMatriz de covarianza:")
	#print(matrizCovarianza)
	MC = pd.DataFrame(np.array(matrizCovarianza)) # Matriz de covarianza con pandas
	print(MC)

	# Valores propios y vectores propios
	evalues, evectors = la.eig(np.array(matrizCovarianza))
	print("\nValores propios:")
	eigenvalues = evalues.tolist()
	print(eigenvalues)
	print("\nVectores propios:")
	print(evectors)
	eigenvectors = evectors.tolist()
	#print(eigenvectors)
	'''























	

def KNN():
	print("Algoritmo de KNN")

	# Se establece la selección de la metrica que se quiere
	opcionMetrica = seleccion_metrica()

	# Cargar el archivo dataset
	dataset = loadDataset()
	#print(dataset)
	#print(dataset.head())

	# Se extraen las columnas del dataset
	dataset.columns
	#print(dataset.columns)

	print("Descripción del dataset: ")
	print(dataset.describe()) # Descripción estadistica de los datos
	columns = len(dataset.columns) # Número total de columnas
	#print(columns)
	rows = len(dataset.index) # Número total de filas
	#print(rows)

	# Agrupando columnas por tipo de datos
	tipos = dataset.columns.to_series().groupby(dataset.dtypes).groups

	# Armando lista de columnas categóricas
	try:
		ctext = tipos[np.dtype('object')]
	except KeyError:
		ctext = list() # lista de columnas vacia en caso de que no haya categóricas
	print("\nNúmero de columnas categoricas: " + str(len(ctext))) # cantidad de columnas con datos categóricos

	# Armando lista de columnas numéricas
	columnas = dataset.columns  # lista total las columnas
	cnum = list(set(columnas) - set(ctext)) # Total de columnas menos columnas no numéricas
	print("Número de columnas numericas: " + str(len(cnum)))

	# Lista de medias
	#print(dataset.mean())
	'''media = list()
	for x in range(len(dataset.mean())):
		media.append(dataset.mean()[x])
	print(media)'''

	media = dataset.mean().values.tolist() # Convertir la media del dataset a una lista
	print("\nLista de medias: " + str(media))
	#print(media)

	# Lista de desviación estandar
	#print(dataset.std())
	'''destandar = list()
	for x in range(len(dataset.std())):
		destandar.append(dataset.std()[x])
	print(destandar)'''

	destandar = dataset.std().values.tolist() # Convertir la desviación estandar del dataset a una lista
	print("Lista de desviación estandar: " + str(destandar))
	#print(destandar)

	# Lista de caracteristicas
	lcaracteristicas = list()
	for x in range(len(cnum)):
		lcaracteristicas.append(columnas[x])
	print("Lista de características: " + str(lcaracteristicas))
	#print(lcaracteristicas)

	# Todas las filas y todas las columnas del dataset en una lista
	matriz = dataset.loc[:,].values
	print("\nMatriz:")
	print(matriz)

	# Matriz normalizada
	matrizNormalizada = list()
	MNrow = list()
	for x in range(rows):
		for y in range(columns):
			result = ((matriz[x][y] - media[y]) / destandar[y])
			MNrow.append(result)
		matrizNormalizada.append(MNrow)
		MNrow = list()
	print("\nMatriz normalizada:")
	#print(matrizNormalizada) # Matriz con los valores normalizados de los Clusters
	MN = pd.DataFrame(np.array(matrizNormalizada)) # Matriz normalizada con pandas
	print(MN)

	# Matriz de distancia euclidiana
	individuo_tabla = list()
	lista_metrica = list()
	matriz_distancia = list()
	for n in range(rows):
		individuo = vector_registro_matrizNormalizada(n, matrizNormalizada, columns)
		#print(individuo)
		for v in range(rows):
			for h in range(columns):
				valor = matrizNormalizada[v][h]
				individuo_tabla.append(valor)
			#print(individuo_tabla)
			metrica = seleccion_metrica_distancia(individuo, individuo_tabla, opcionMetrica)
			#print(metrica)
			lista_metrica.append(metrica)
			individuo_tabla = list()
		#print(lista_metrica)
		matriz_distancia.append(lista_metrica)
		lista_metrica = list()
	print("\nMatriz de " + str(lista_metricas[opcionMetrica-1]) + ": ")
	#print(matriz_distancia)
	MD = pd.DataFrame(np.array(matriz_distancia)) # Matriz de distacia con pandas
	print(MD)

	'''
	# Matriz de distancia euclidiana, triangular inferior
	triangular_inferior = copy.deepcopy(matriz_distancia) # Copia de la lista
	TIrow = list()
	for x in range(0,len(triangular_inferior)):
		for y in range(x,len(triangular_inferior)):
			triangular_inferior[x][y] = 0
	print("\nMatriz de distancia euclidiana triangular inferior:")
	#print(triangular_inferior)
	TI = pd.DataFrame(np.array(triangular_inferior)) # Matriz de distacia triangular inferior con pandas
	print(TI)
	'''

	# Selección de las características para el gráfico
	print()
	for x in range(len(lcaracteristicas)):
		print(str(x+1) + ". " + lcaracteristicas[x])
	print("Seleccione dos características para el gráfico de Cluster: ")

	# Lista de las caracteristicas seleccionadas
	listaGrafica = list()
	etiquetasGrafica = list()
	for x in range(2):
		nCluster = int(input("Característica " + str(x+1) + ": "))
		listaGrafica.append(nCluster-1) # Se seleccionan las caracteristicas de la lista de acuerdo al numero ingresado
		etiquetasGrafica.append(lcaracteristicas[nCluster-1]) # Etiquetas de los ejes de la gráfica, las características seleccionadas
	print("Ejes de la gráfica: " + str(listaGrafica)) # Lista de ejes para la gráfica
	print("Etiquetas de la gráfica: " + str(etiquetasGrafica))

	# Todas las filas y dos columnas para la gráfica
	matrizGrafica = MN.ix[:, [listaGrafica[0], listaGrafica[1]]].values
	print("\nMatriz de la gráfica:")
	print(matrizGrafica)

	# =============================================================================================================================

	# Número de vecinos cercanos
	k = int(input("\nIngrese el número de K o vecinos cercanos: "))
	while (k < 1 or k > rows):
		k = int(input("El valor de K es incorrecto, debe estar entre 1" + " y " + str(rows) + ": "))

	# Valor seleccionado aleatoriamente para ser el Cluster
	aleatorio = random.randint(0, rows-1)
	print("Valor seleccionado aleatoriamente: " + str(aleatorio))
	Cluster = matriz_distancia[aleatorio]
	print(Cluster)
	
	# Lista de distancias con respecto al Cluster seleccionado
	valorDistancias = list()
	individuo_tabla_distancia = list()
	for x in range(len(matriz_distancia)):
		for y in range(len(Cluster)):
			valor = matriz_distancia[x][y]
			individuo_tabla_distancia.append(valor)
		distancia = seleccion_metrica_distancia(Cluster, individuo_tabla_distancia, opcionMetrica)
		valorDistancias.append(distancia)
		individuo_tabla_distancia = list()
	print("\nDistancias al punto del Cluster:")
	print(valorDistancias)

	# Lista de indices
	indices = list()
	for x in range(len(valorDistancias)):
		indices.append(x)

	# Diccionario de indices: valorDistancias
	diccionario = dict(zip(indices, valorDistancias))
	d = sorted(diccionario.items(), key=operator.itemgetter(1)) # Ordena el diccionario en tuplas por el valor (0 = clave, 1 = valor)
	print("\nDiccionario vecino:ditancia ordenado por distancia:")
	print(d) # Imprime el diccionario en tuplas ordenado por valor

	# Llaves y valores en listas
	llaves = list()
	valores = list()
	for x in range(len(diccionario)):
		llaves.append(d[x][0])
		valores.append(d[x][1])
	#print(llaves)
	#print(valores)
	
	# k vecinos más cercanos con el nuevo dato, ejes X y Y
	knnDistancia = list()
	knnDatos = list()
	for x in range(k):
		#print(str(llaves[x]) + " " + str(valores[x])) # Vecino (indice) y su distancia
		#print(matrizNormalizada[llaves[x]]) # Datos de los vecinos más cercanos
		knnDistancia.append([llaves[x], valores[x]]) # Lista del vecino más cercano con su distancia
		knnDatos.append(matrizNormalizada[llaves[x]]) # Lista de los datos de los vecinos más cercanos

	print("\nMatriz de vecinos más cercano con su distancia:")
	#print(knnDistancia)
	print(pd.DataFrame(np.array(knnDistancia))) # Matriz de vecinos más cercano con su distancia
	print("\nMatriz de los datos de los vecinos más cercanos:")
	#print(knnDatos)
	print(pd.DataFrame(np.array(knnDatos))) # Matriz de los datos de los vecinos más cercanos
	
	'''
	# Listas de los ejes X y Y de Knn # Otra forma de hacerlo
	kEjeX = list()
	kEjeY = list()
	for x in range(k):
		kEjeX.append(knnDatos[x][0])
		kEjeY.append(knnDatos[x][1])
	print(kEjeX) # Eje X de los datos vecinos más cercanos
	print(kEjeY) # Eje Y de los datos vecinos más cercanos
	
	# Listas de los ejes X y Y # Otra forma de hacerlo
	ejeX = list()
	ejeY = list()
	for x in range(len(matrizNormalizada)):
		ejeX.append(matrizNormalizada[x][0])
		ejeY.append(matrizNormalizada[x][1])
	print(ejeX)
	print(ejeY)
	'''

	# DataFrame con pandas de los ejes X y Y de la matriz normalizada
	ejes = pd.DataFrame(np.array(matrizGrafica))
	ejeX = ejes.ix[:, 0]
	ejeY = ejes.ix[:, 1]
	
	# DataFrame con pandas de los ejes X y Y de la matriz de vecinos más cercanos
	KNND = pd.DataFrame(np.array(knnDatos)) # Se convierte la lista de datos de vecinos más cercanos a DataFrame
	matrizGrafica = KNND.ix[:, [listaGrafica[0], listaGrafica[1]]].values # Lista de los ejes que se escogieron de vecinos más cercanos
	kEjes = pd.DataFrame(np.array(matrizGrafica))
	kEjeX = kEjes.ix[:, 0]
	kEjeY = kEjes.ix[:, 1]
	
	# Radio tomado de la mayor distancia de los vecinos más cercanos
	#radio = knnDistancia[k-1][1]

	# Grafica de los datos normalizados
	plt.plot(ejeX, ejeY, 'ro', marker='o', color='r', label="Valores", alpha=0.5) # Datos de la matriz normalizada en rojo
	plt.plot(Cluster[listaGrafica[0]], Cluster[listaGrafica[1]], 'bo', marker='o', color='b', label="Valor nuevo") # Nuevo dato en azul
	plt.plot(Cluster[listaGrafica[0]], Cluster[listaGrafica[1]], 'bo', marker='o', markersize=100, linewidth=0.5, alpha=0.2) # Área del nuevo dato
	plt.plot(kEjeX, kEjeY, 'go', marker='o', color='g', label="Vecinos cerca", alpha=0.5) # Datos de la matriz de vecinos más cercanos en verde
	plt.xlabel(etiquetasGrafica[0]) # Etiqueda en el eje X
	plt.ylabel(etiquetasGrafica[1]) # Etiqueta en el eje Y
	plt.grid(color='b', alpha=0.2, linestyle='dashed', linewidth=0.5) # Malla o grilla
	plt.title('KNN, k = ' + str(k)) # Titulo de la gráfica
	plt.legend(loc="lower right") # Legenda de la gráfica
	plt.show()

def KMEANS():
	print("Algoritmo de KMEANS")

	# Se establece la selección de la metrica que se quiere
	opcionMetrica = seleccion_metrica()

	# Cargar el archivo dataset
	dataset = loadDataset()
	#print(dataset)
	#print(dataset.head())

	# Se extrane las columnas del dataset
	dataset.columns
	print(dataset.columns)

	print(dataset.describe()) # Descripción estadistica de los datos
	columns = len(dataset.columns) # Número total de columnas
	#print(columns)
	rows = len(dataset.index) # Número total de filas
	#print(rows)

	# Agrupando columnas por tipo de datos
	tipos = dataset.columns.to_series().groupby(dataset.dtypes).groups

	# Armando lista de columnas categóricas
	try:
		ctext = tipos[np.dtype('object')]
	except KeyError:
		ctext = list() # lista de columnas vacia en caso de que no haya categóricas
	print("\nNúmero de columnas categoricas: " + str(len(ctext))) # cantidad de columnas con datos categóricos

	# Armando lista de columnas numéricas
	columnas = dataset.columns  # lista total las columnas
	cnum = list(set(columnas) - set(ctext)) # Total de columnas menos columnas no numéricas
	print("Número de columnas numericas: " + str(len(cnum)))

	# Lista de medias
	#print(dataset.mean())
	'''media = list()
	for x in range(len(dataset.mean())):
		media.append(dataset.mean()[x])
	print(media)'''

	media = dataset.mean().values.tolist() # Convertir la media del dataset a una lista
	print("\nLista de medias: " + str(media))
	#print(media)

	# Lista de desviación estandar
	#print(dataset.std())
	'''destandar = list()
	for x in range(len(dataset.std())):
		destandar.append(dataset.std()[x])
	print(destandar)'''

	destandar = dataset.std().values.tolist() # Convertir la desviación estandar del dataset a una lista
	print("Lista de desviación estandar: " + str(destandar))
	#print(destandar)

	# Lista de caracteristicas
	lcaracteristicas = list()
	for x in range(len(cnum)):
		lcaracteristicas.append(columnas[x])
	print("Lista de características: " + str(lcaracteristicas))
	#print(lcaracteristicas)

	# Todas las filas y todas las columnas del dataset en una lista
	matriz = dataset.loc[:,].values
	print("\nMatriz:")
	print(matriz)

	# Matriz normalizada
	matrizNormalizada = list()
	MNrow = list()
	for x in range(rows):
		for y in range(columns):
			result = ((matriz[x][y] - media[y]) / destandar[y])
			MNrow.append(result)
		matrizNormalizada.append(MNrow)
		MNrow = list()
	print("\nMatriz normalizada:")
	#print(matrizNormalizada) # Matriz con los valores normalizados de los Clusters
	MN = pd.DataFrame(np.array(matrizNormalizada)) # Matriz normalizada con pandas
	print(MN)

	# Matriz de distancia euclidiana
	individuo_tabla = list()
	lista_metrica = list()
	matriz_distancia = list()
	for n in range(rows):
		individuo = vector_registro_matrizNormalizada(n, matrizNormalizada, columns)
		#print(individuo)
		for v in range(rows):
			for h in range(columns):
				valor = matrizNormalizada[v][h]
				individuo_tabla.append(valor)
			#print(individuo_tabla)
			metrica = seleccion_metrica_distancia(individuo, individuo_tabla, opcionMetrica)
			#print(metrica)
			lista_metrica.append(metrica)
			individuo_tabla = list()
		#print(lista_metrica)
		matriz_distancia.append(lista_metrica)
		lista_metrica = list()
	print("\nMatriz de " + str(lista_metricas[opcionMetrica-1]) + ": ")
	#print(matriz_distancia)
	MD = pd.DataFrame(np.array(matriz_distancia)) # Matriz de distacia con pandas
	print(MD)

	'''
	# Matriz de distancia euclidiana, triangular inferior
	triangular_inferior = copy.deepcopy(matriz_distancia) # Copia de la lista
	TIrow = list()
	for x in range(0,len(triangular_inferior)):
		for y in range(x,len(triangular_inferior)):
			triangular_inferior[x][y] = 0
	print("\nMatriz de distancia euclidiana triangular inferior:")
	#print(triangular_inferior)
	TI = pd.DataFrame(np.array(triangular_inferior)) # Matriz de distacia triangular inferior con pandas
	print(TI)
	'''

	# Selección de las características para el gráfico
	print()
	for x in range(len(lcaracteristicas)):
		print(str(x+1) + ". " + lcaracteristicas[x])
	print("Seleccione dos características para el gráfico de Cluster: ")

	# Lista de las caracteristicas seleccionadas
	listaGrafica = list()
	etiquetasGrafica = list()
	for x in range(2):
		nCluster = int(input("Característica " + str(x+1) + ": "))
		listaGrafica.append(nCluster-1) # Se seleccionan las caracteristicas de la lista de acuerdo al numero ingresado
		etiquetasGrafica.append(lcaracteristicas[nCluster-1]) # Etiquetas de los ejes de la gráfica, las características seleccionadas
	print("Ejes de la gráfica: " + str(listaGrafica)) # Lista de ejes para la gráfica
	print("Etiquetas de la gráfica: " + str(etiquetasGrafica))

	# Todas las filas y dos columnas para la gráfica
	matrizGrafica = MN.ix[:, [listaGrafica[0], listaGrafica[1]]].values
	print("\nMatriz de la gráfica:")
	print(matrizGrafica)
	
	# =========================================================================================================================
	
	
	# Número de cluster
	nCluster = int(input("\nIngrese el número de N Clusters: "))
	while (nCluster < 1 or nCluster > rows):
		nCluster = int(input("El valor de N es incorrecto, debe estar entre 1" + " y " + str(rows) + ": "))
	
	# Selección de los centroides de los Clusters aleatorios y sus indices hasta completar el número de Clusters
	ClustersCentroides = list()
	ClustersIndices = list()
	x = 0
	while (x < nCluster):
		nRandom = random.randint(0, len(matrizNormalizada)-1)
		if nRandom not in ClustersIndices: # Si el nRandom no esta en la lista ClustersIndices no entra
			x += 1
			ClustersIndices.append(nRandom)
			valor = matrizNormalizada[nRandom]
			ClustersCentroides.append(valor)
	print("\nIndices de los Clusters: " + str(ClustersIndices)) # Indices de los Clusters
	print("Centroides de los Clusters: ") # Clusters
	print(pd.DataFrame(np.array(ClustersCentroides)))

	# Distancia de los Clusters
	ClustersDistancias = list()
	for x in range(nCluster):
		indice = ClustersIndices[x]
		ClustersDistancias.append(matriz_distancia[indice])
	print("\nDistancia de los Clusters: ")
	#print(ClustersDistancias)
	CD = pd.DataFrame(np.array(ClustersDistancias)) # Matriz de distacia de los Clusters
	print(CD)

	#print(CD.describe())
	#print(CD.min())

	# Lista de mínimos
	ClustersMin = CD.min().values.tolist() # Convertir los mínimos de ClustersDistancias a una lista
	print("\nLista de mínimos: " + str(ClustersMin))
	#print(ClustersMin)

	# Lista de Clusters
	Clusters = list()
	listaMin = list()
	for x in range(len(matriz_distancia)):
		for y in range(nCluster):
			listaMin.append(ClustersDistancias[y][x])
		Clusters.append(minimo(listaMin))
		listaMin = list()
	print("\nLista de Clusters: " + str(Clusters))

	# Lista de indices de los Clusters
	indices = list()
	for x in range(len(Clusters)):
		indices.append(x)

	# Diccionario de indices: Clusters
	ClustersDiccionario = dict(zip(indices, Clusters))
	cd = sorted(ClustersDiccionario.items(), key=operator.itemgetter(1)) # Ordena el diccionario en tuplas por el valor (0 = clave, 1 = valor)
	print("\nDiccionario valor:Cluster ordenado por Cluster:")
	print(cd) # Imprime el diccionario en tuplas ordenado por valor

	# Llaves y valores en listas
	llaves = list()
	valores = list()
	for x in range(len(ClustersDiccionario)):
		llaves.append(cd[x][0])
		valores.append(cd[x][1])
	#print(llaves)
	#print(valores)

	# Diccionario con las claves pero sin valores Clusters:Valores
	ClustersDiccionario = {}
	for x in range(nCluster):
		ClustersDiccionario.setdefault('Cluster'+str(x),)
	#print(ClustersDiccionario) # Diccionario vacio
	
	# Diccionario con los Clusters completos
	listaClusters = list()
	for x in range(nCluster):
		for y in range(len(Clusters)):
			if valores[y] == x:
				listaClusters.append(matrizNormalizada[llaves[y]])
		ClustersDiccionario['Cluster'+str(x)] = listaClusters
		listaClusters = list()
	#print(ClustersDiccionario)
	
	# Diccionario de cada Cluster
	for x in range(nCluster):
		print("\nCluster " + str(x) + ": " + str(ClustersDiccionario['Cluster'+str(x)]))


	# ============================================================================================================================
	
	
	# SEGUNDA ITERACIÓN
	iteraciones = 1
	respuesta = False
	while (respuesta == False):

		# Nuevos centroides del Cluster calculados por la media
		mediaClustersNuevosCentroides = list()
		for x in range(nCluster):
			ClusterNuevo = pd.DataFrame(np.array(ClustersDiccionario['Cluster'+str(x)]))
			mcn = ClusterNuevo.mean().values.tolist() # Convertir la media del ClusterX a una lista
			mediaClustersNuevosCentroides.append(mcn)
		print("\nLista de medias de los Cluster nuevos: ")
		print(pd.DataFrame(np.array(mediaClustersNuevosCentroides)))

		# Distancia de nuevos centroides
		individuo_tabla = list()
		lista_metrica = list()
		matriz_distancia_cluster = list()
		for n in range(nCluster):
			
			for v in range(rows):
				for h in range(columns):
					valor = matrizNormalizada[v][h]
					individuo_tabla.append(valor)
				#print(individuo_tabla)
				metrica = seleccion_metrica_distancia(mediaClustersNuevosCentroides[n], individuo_tabla, opcionMetrica)
				#print(metrica)
				lista_metrica.append(metrica)
				individuo_tabla = list()
			#print(lista_metrica)
			matriz_distancia_cluster.append(lista_metrica)
			lista_metrica = list()
		print("\nMatriz de " + str(lista_metricas[opcionMetrica-1]) + " de los Cluster nuevos:")
		#print(matriz_distancia_cluster)
		MDC = pd.DataFrame(np.array(matriz_distancia_cluster)) # Matriz de distacia con pandas
		print(MDC)
		
		# Lista de mínimos nuevos
		ClustersMinNuevos = MDC.min().values.tolist() # Convertir los mínimos de ClustersMinNuevos a una lista
		print("\nLista de mínimos nuevos: " + str(ClustersMinNuevos))
		#print(ClustersMin)

		# Lista de Clusters nuevos
		ClustersNuevos = list()
		listaMinNuevos = list()
		for x in range(len(matriz_distancia)):
			for y in range(nCluster):
				listaMinNuevos.append(matriz_distancia_cluster[y][x])
			ClustersNuevos.append(minimo(listaMinNuevos))
			listaMinNuevos = list()
		print("\nLista de Clusters pasada: " + str(Clusters))
		print("\nLista de Clusters nuevos: " + str(ClustersNuevos))

		# Condición que compara la lista inicial de los Clusters con la lista nueva de los Clusters
		if (Clusters == ClustersNuevos):
			respuesta = True
		else:
			respuesta = False
		#print(respuesta)
		
		# Lista de indices de los Clusters nuevos
		indices = list()
		for x in range(len(ClustersNuevos)):
			indices.append(x)

		# Diccionario de indices: Clusters nuevos
		ClustersDiccionarioNuevo = dict(zip(indices, ClustersNuevos))
		cdn = sorted(ClustersDiccionarioNuevo.items(), key=operator.itemgetter(1)) # Ordena el diccionario en tuplas por el valor (0 = clave, 1 = valor)
		print("\nDiccionario valor:ClusterNuevo ordenado por Cluster:")
		print(cdn) # Imprime el diccionario en tuplas ordenado por valor

		# Llaves y valores en listas
		llaves = list()
		valores = list()
		for x in range(len(ClustersDiccionarioNuevo)):
			llaves.append(cdn[x][0])
			valores.append(cdn[x][1])
		#print(llaves)
		#print(valores)

		# Diccionario con las claves pero sin valores Clusters:Valores
		ClustersDiccionarioNuevo = {}
		for x in range(nCluster):
			ClustersDiccionarioNuevo.setdefault('Cluster'+str(x),)
		#print(ClustersDiccionarioNuevo) # Diccionario vacio

		# Diccionario con los Clusters completos
		listaClustersNuevo = list()
		for x in range(nCluster):
			for y in range(len(Clusters)):
				if valores[y] == x:
					listaClustersNuevo.append(matrizNormalizada[llaves[y]])
			ClustersDiccionarioNuevo['Cluster'+str(x)] = listaClustersNuevo
			listaClustersNuevo = list()
		#print(ClustersDiccionarioNuevo)
		
		# Diccionario de cada Cluster
		for x in range(nCluster):
			print("\nCluster " + str(x) + ": " + str(ClustersDiccionarioNuevo['Cluster'+str(x)]))

		iteraciones += 1
		Clusters = copy.deepcopy(ClustersNuevos)

	# Número de iteraciones despues de la segunda iteración
	print("\nNúmero de iteraciones: " + str(iteraciones))

	# DataFrame con pandas de los ejes X y Y de los centroides de los Clusters seleccionados
	ejesCentroides = pd.DataFrame(np.array(mediaClustersNuevosCentroides))
	ejeXCentroides = ejesCentroides.ix[:, listaGrafica[0]]
	ejeYCentroides = ejesCentroides.ix[:, listaGrafica[1]]

	# DataFrame con pandas de los ejes X y Y de Clusters X
	colors = "grcmykwb" # Lista de colores de los Cluster
	for x in range(nCluster):
		ejesCluster = pd.DataFrame(np.array(ClustersDiccionarioNuevo['Cluster'+str(x)]))
		print("\nCluster " + str(x) + ":")
		print(ejesCluster)
		ejeXCluster = ejesCluster.ix[:, listaGrafica[0]]
		ejeYCluster = ejesCluster.ix[:, listaGrafica[1]]
		# Datos (coordenadas) de cada Cluster X 
		# Se asignan estas lineas en este bucle para no crear ejes, ejex y ejey por cada grupo de Clusters
		color = colors[x] # Color de cada Cluster
		plt.plot(ejeXCluster, ejeYCluster, color+str('o'), marker='o', color=color, label='Cluster'+str(x), alpha=0.5) # Datos del Cluster 0
	
	# Grafica de los datos normalizados
	plt.plot(ejeXCentroides, ejeYCentroides, 'bo', marker='o', color='b', label="Centroides", alpha=0.3) # Datos de los centroides en rojo
	# Área de cada centroide X
	for x in range(nCluster):
		plt.plot(ejeXCentroides[x], ejeYCentroides[x], 'bo', marker='o', markersize=100, linewidth=0.5, alpha=0.1) # Área de los centroides X

	plt.xlabel(etiquetasGrafica[0]) # Etiqueda en el eje X
	plt.ylabel(etiquetasGrafica[1]) # Etiqueta en el eje Y
	plt.grid(color='b', alpha=0.2, linestyle='dashed', linewidth=0.5) # Malla o grilla
	plt.title('KMEANS, N Clusters = ' + str(nCluster) + ', Iteraciones = ' + str(iteraciones)) # Titulo de la gráfica
	plt.legend(loc="lower right") # Legenda de la gráfica
	plt.show()
	
def EM():
	print("Algoritmo de EM")


def menu():
	bucle = True
	while (bucle == True):
		print("\n========================================================================") 
		print("PROGRAMA DE INPLEMENTACIÓN DE ALGORITMOS DE REDUCCIÓN DE DIMENSIONALIDAD") 
		print("E IMPLEMENTACIÓN DE ALGORITMOS DE ClUSTERING")
		print("========================================================================\n")
		print("SELECCIONE UN ALGOTIRMO:")
		print("------------------------------------------------------------------------")
		print("* Análisis de componentes principales (ACP) ....................1")
		print("* Análisis de componentes principales por kernel (ACPK) ........2")
		print("* KNN (K-Vecinos más Cercanos) .................................3")
		print("* KMEANS (Método de agrupamiento) ..............................4")
		print("* SALIR ........................................................0")
		print("------------------------------------------------------------------------")
		opcion = input("Opción: ")

		if (opcion != "1" and opcion != "2" and opcion != "3" and opcion != "4" and opcion != "0"):
			print("La opción seleccionada no es valida")

		while (opcion < "0" or opcion > "4"):
			opcion = input("Ingrese nuevamente la opcion: ")

		if (opcion == "1"):
			ACP()
		elif (opcion == "2"):
			ACPK()
		elif (opcion == "3"):
			KNN()
		elif (opcion == "4"):
			KMEANS()
		elif (opcion == "0"):
			bucle = False
			exit()

menu()