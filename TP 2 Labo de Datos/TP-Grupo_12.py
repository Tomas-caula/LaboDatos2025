import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 1 - Analisis Exploratorio del Dataset
data_df = pd.read_csv("Fashion-MNIST.csv", index_col=0)

print(data_df.shape) # Numero de filas y columnas
print(data_df.head())  # Primeras filas del dataset

print(data_df['label'].value_counts()) # Cantidad de imagenes por clase
#print(data_df.isnull().sum())  # Verificar si hay valores nulos

# Asignar X e Y correctamente
y = data_df['label'].astype(int)
X = data_df.drop('label', axis=1)

# Normalizar
X = X / 255.0

# Calcular desviacion estandar por pixel
std_por_pixel = np.std(X, axis=0).values  # .values para convertir de pandas Series a NumPy array

# Reshape (784 → 28x28)
std_image = std_por_pixel.reshape(28, 28)

# Visualizar
plt.figure(figsize=(6, 6))
sns.heatmap(std_image, cmap='viridis', cbar=True)
plt.title('Desviación estándar por píxel (dataset completo)')
plt.axis('off')
plt.show()

# Hay 10 clases en Fashion-MNIST (etiquetas de 0 a 9)
for clase in range(10):
    # Filtrar las filas que corresponden a esta clase
    X_clase = X[y == clase]
    
    # Calcular la desviacion estandar de cada pixel para esta clase
    std_pixel_clase = np.std(X_clase, axis=0).values.reshape(28, 28)
    
    # Guardar en el diccionario
    #stds_por_clase[clase] = std_pixel_clase
    
    # Visualizacion
    plt.figure(figsize=(4, 4))
    sns.heatmap(std_pixel_clase, cmap='viridis', cbar=True)
    plt.title(f'Desviación estándar - Clase {clase}')
    plt.axis('off')
    plt.show()

# Diccionario para guardar imágenes promedio por clase
promedios_por_clase = {}

for clase in range(10):
    # Filtrar imágenes de la clase actual
    X_clase = X[y == clase]

    # Calcular el promedio por píxel (valor medio en cada posición)
    promedio_pixel_clase = np.mean(X_clase, axis=0).values.reshape(28, 28)

    # Guardar en el diccionario
    promedios_por_clase[clase] = promedio_pixel_clase

    # Visualización
    plt.figure(figsize=(4, 4))
    sns.heatmap(promedio_pixel_clase, cmap='gray', cbar=False)
    plt.title(f'Imagen promedio - Clase {clase}')
    plt.axis('off')
    plt.show()

# Crear matriz 10x10 para guardar distancias entre imágenes promedio
dist_matrix = np.zeros((10, 10))

for i in range(10):
    # Aplanar la imagen promedio de clase i a vector (784,)
    img_i = promedios_por_clase[i].flatten()
    
    for j in range(10):
        img_j = promedios_por_clase[j].flatten()
        
        # Calcular distancia euclidiana entre los promedios
        dist = np.linalg.norm(img_i - img_j)
        dist_matrix[i, j] = dist

# Visualizar la matriz de distancias
plt.figure(figsize=(8, 6))
sns.heatmap(dist_matrix, annot=True, fmt=".2f", cmap="magma", xticklabels=range(10), yticklabels=range(10))
plt.title("Distancia Euclidiana entre imágenes promedio de cada clase")
plt.xlabel("Clase")
plt.ylabel("Clase")
plt.show()


#Ejercicio 2 
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Cargar el dataset
data_df = pd.read_csv("Fashion-MNIST.csv", index_col=0)

# Crear subconjunto con solo las clases 0 y 8
data_df_0_8 = data_df[data_df['label'].isin([0, 8])]

# Análisis del balance de clases

    #conteo_clases = data_df_0_8['label'].value_counts()
    #print(conteo_clases)

    #porcentaje_clases = (conteo_clases / len(data_df_0_8)) * 100
    #print(porcentaje_clases)

#Aca pudiomos observar que las clases estan balanceadas, por lo que no es necesario balancearlas


# Visualizar el balance de clases
    #plt.figure()
    #sns.countplot(data=data_df_0_8, x='label')
    #plt.title('Distribución de clases 0 y 8')
    #plt.xlabel('Clase')
    #plt.ylabel('Cantidad de imágenes')
#plt.show()

# Separar características (X) y etiquetas (y)
X = data_df_0_8.drop('label', axis=1)
y = data_df_0_8['label']

# Crear conjunto de entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Función para evaluar diferentes subconjuntos de atributos
def evaluar_subconjuntos(X_train, X_test, y_train, y_test, subconjuntos, k=3):
    resultados = {}
    
    for nombre, indices in subconjuntos.items():
        # Seleccionar subconjunto de atributos
        X_train_subset = X_train.iloc[:, indices]
        X_test_subset = X_test.iloc[:, indices]
        
        # Entrenar modelo kNN
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_subset, y_train)
        
        # Evaluar modelo
        y_pred = knn.predict(X_test_subset)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        
        resultados[nombre] = accuracy
    
    return resultados

# Definimos todos los subconjuntos de atributos que vamos a usar -> 
subconjuntos_3_atributos = {
    'Subconjunto 1': [0, 1, 2],  # Primeros 3 píxeles
    'Subconjunto 2': [200, 300, 400],  # Píxeles del medio
    'Subconjunto 3': [781, 782, 783],  # Últimos 3 píxeles
}

subconjuntos_5_atributos = {
    'Subconjunto 1': [0, 1, 2, 3, 4],
    'Subconjunto 2': [100, 200, 300, 400, 500],
    'Subconjunto 3': [779, 780, 781, 782, 783],
}

# <- Definimos todos los subconjuntos de atributos que vamos a usar

# Evaluar subconjuntos de 3 atributos
resultados_3 = evaluar_subconjuntos(X_train, X_test, y_train, y_test, subconjuntos_3_atributos)
print("\nResultados para subconjuntos de 3 atributos:")
for nombre, accuracy in resultados_3.items():
    print(f"{nombre}: {accuracy:.4f}")

# Evaluar subconjuntos de 5 atributos
resultados_5 = evaluar_subconjuntos(X_train, X_test, y_train, y_test, subconjuntos_5_atributos)
print("\nResultados para subconjuntos de 5 atributos:")
for nombre, accuracy in resultados_5.items():
    print(f"{nombre}: {accuracy:.4f}")

# Visualizar resultados
plt.figure(figsize=(12, 6))

# Gráfico para 3 atributos
plt.subplot(1, 2, 1)
plt.bar(resultados_3.keys(), resultados_3.values())
plt.title('Precisión con 3 atributos')
plt.xticks(rotation=45)
plt.ylabel('Precisión')

# Gráfico para 5 atributos
plt.subplot(1, 2, 2)
plt.bar(resultados_5.keys(), resultados_5.values())
plt.title('Precisión con 5 atributos')
plt.xticks(rotation=45)
plt.ylabel('Precisión')



#Probando con k diferentes

const_k = [1, 3, 5, 7, 9]
# Almacenar resultados para graficar
resultados_3_atributos = {k: {} for k in const_k}
resultados_5_atributos = {k: {} for k in const_k}

# Evaluar y almacenar resultados para 3 atributos
for k in const_k:
    resultados_k = evaluar_subconjuntos(X_train, X_test, y_train, y_test, subconjuntos_3_atributos, k)
    resultados_3_atributos[k] = resultados_k

# Evaluar y almacenar resultados para 5 atributos
for k in const_k:
    resultados_k = evaluar_subconjuntos(X_train, X_test, y_train, y_test, subconjuntos_5_atributos, k)
    resultados_5_atributos[k] = resultados_k

# Crear gráfico
plt.figure(figsize=(15, 6))

# Gráfico para 3 atributos
plt.subplot(1, 2, 1)
for subconjunto in subconjuntos_3_atributos.keys():
    valores = [resultados_3_atributos[k][subconjunto] for k in const_k]
    plt.plot(const_k, valores, marker='o', label=subconjunto)
plt.title('Precisión vs k para 3 atributos')
plt.xlabel('Valor de k')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)


# Gráfico para 5 atributos
plt.subplot(1, 2, 2)
for subconjunto in subconjuntos_5_atributos.keys():
    valores = [resultados_5_atributos[k][subconjunto] for k in const_k]
    plt.plot(const_k, valores, marker='o', label=subconjunto)
plt.title('Precisión vs k para 5 atributos')
plt.xlabel('Valor de k')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#Caso pruebo con todos los atributos con k= 3


# %%