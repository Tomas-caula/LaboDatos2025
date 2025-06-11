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
