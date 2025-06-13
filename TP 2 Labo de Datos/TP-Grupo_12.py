import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score

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

# Crear subconjunto with solo las clases 0 y 8
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold # Importar StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Paso 0: Cargar y preparar el dataset ---

# Cargar el dataset completo (todas las clases de Fashion-MNIST)
# Asegúrate de que el archivo 'Fashion-MNIST.csv' esté en el mismo directorio que tu script.
try:
    data_df = pd.read_csv("Fashion-MNIST.csv", index_col=0)
except FileNotFoundError:
    print("Error: 'Fashion-MNIST.csv' no encontrado. Asegúrate de que el archivo esté en el directorio correcto.")
    exit() # Sale del script si el archivo no se encuentra

# Separar características (X) y etiquetas (y)
# Normalizar los valores de los píxeles de 0-255 a 0-1.0 para mejorar el rendimiento del modelo.
X = data_df.drop('label', axis=1) / 255.0
y = data_df['label'].astype(int) # Asegurar que las etiquetas sean enteros

# Definir los nombres de las clases para una mejor interpretación de los resultados
# ¡IMPORTANTE! Revisa la documentación de Fashion-MNIST para confirmar estos nombres si es necesario.
class_names = [
    "T-shirt/top",  # Clase 0
    "Trouser",      # Clase 1
    "Pullover",     # Clase 2
    "Dress",        # Clase 3
    "Coat",         # Clase 4
    "Sandal",       # Clase 5
    "Shirt",        # Clase 6
    "Sneaker",      # Clase 7
    "Bag",          # Clase 8
    "Ankle boot"    # Clase 9
]

# --- Paso 1: División en conjunto de desarrollo y held-out ---

# Dividir los datos en un conjunto de desarrollo (para entrenamiento y validación cruzada)
# y un conjunto held-out (para evaluación final imparcial).
# test_size=0.2 significa que el 20% de los datos se usará para held-out, 80% para desarrollo.
# random_state asegura que la división sea reproducible.
# stratify=y asegura que la proporción de clases sea la misma en ambos conjuntos (muestreo estratificado).
X_dev, X_heldout, y_dev, y_heldout = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Tamaño del conjunto de desarrollo: {X_dev.shape[0]} muestras")
print(f"Tamaño del conjunto held-out: {X_heldout.shape[0]} muestras")

# --- Paso 2: Selección de hiperparámetro max_depth usando validación cruzada ---

# Definir la cuadrícula de hiperparámetros a probar para 'max_depth'.
# Se probarán valores de 1 a 10 para la profundidad máxima del árbol.
param_grid = {'max_depth': range(1, 11)}

# Configurar la validación cruzada Estratificada K-Fold.
# n_splits=5 significa que los datos de desarrollo se dividirán en 5 'folds'.
# shuffle=True aleatoriza los datos antes de dividirlos en folds.
# random_state asegura que los folds sean los mismos cada vez que ejecutes el código.
# StratifiedKFold asegura que la proporción de clases sea la misma en cada fold.
# El enunciado no pide que sea estratificado pero por las dudas y para un mejor rendimiento se estratifica
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Inicializar el modelo DecisionTreeClassifier.
# random_state también para la reproducibilidad de la construcción del árbol.
dt = DecisionTreeClassifier(random_state=42)

# Configurar GridSearchCV para realizar la búsqueda del mejor hiperparámetro.
# dt: el modelo a optimizar.
# param_grid: los hiperparámetros y sus valores a probar.
# cv: la estrategia de validación cruzada.
# scoring='accuracy': la métrica utilizada para evaluar cada combinación de hiperparámetros.
# n_jobs=-1: usa todos los núcleos de la CPU para acelerar el proceso.
grid_search = GridSearchCV(dt, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)

# Entrenar GridSearchCV en el conjunto de desarrollo.
# Esto realizará la validación cruzada internamente para cada max_depth.
grid_search.fit(X_dev, y_dev)

# Imprimir los resultados del mejor hiperparámetro encontrado
print("\n--- Resultados de la Selección de Hiperparámetros ---")
print("Mejor max_depth:", grid_search.best_params_['max_depth'])
print(f"Mejor exactitud promedio (validación cruzada): {grid_search.best_score_:.4f}")

# Visualizar la exactitud promedio de la validación cruzada para cada max_depth
results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(10, 6))
plt.plot(results['param_max_depth'], results['mean_test_score'], marker='o', linestyle='-')
plt.title('Exactitud Promedio de Validación Cruzada vs. Max Depth', fontsize=16)
plt.xlabel('Max Depth (Profundidad Máxima del Árbol)', fontsize=12)
plt.ylabel('Exactitud Promedio (Validación Cruzada)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(1, 11)) # Asegura que todos los valores de 1 a 10 estén en el eje X
plt.tight_layout()
plt.show()

# --- Paso 3: Entrenar el modelo final con el mejor max_depth en todo el conjunto de desarrollo ---

# Obtener el mejor valor de max_depth determinado por GridSearchCV.
mejor_max_depth = grid_search.best_params_['max_depth']

# Inicializar y entrenar el modelo de Árbol de Decisión final.
# Este modelo se entrena con TODOS los datos del conjunto de desarrollo (X_dev, y_dev).
# Esto asegura que el modelo final aprenda de la mayor cantidad de datos posible antes de la evaluación final.
final_model = DecisionTreeClassifier(max_depth=mejor_max_depth, random_state=42)
final_model.fit(X_dev, y_dev)

print(f"\nModelo final entrenado con max_depth = {mejor_max_depth} en el conjunto de desarrollo completo.")

# --- Paso 4: Evaluar el modelo en el conjunto held-out ---

# Realizar predicciones en el conjunto held-out.
# Esta es la evaluación imparcial del rendimiento del modelo en datos no vistos.
y_pred_heldout = final_model.predict(X_heldout)

# Calcular la exactitud en el conjunto held-out.
accuracy_heldout = accuracy_score(y_heldout, y_pred_heldout)
print(f"Exactitud final en el conjunto held-out: {accuracy_heldout:.4f}")

# --- Paso 5: Matriz de confusión y reporte de clasificación ---

# Calcular la matriz de confusión.
conf_matrix = confusion_matrix(y_heldout, y_pred_heldout)

# Visualizar la matriz de confusión con nombres de clases.
plt.figure(figsize=(10, 8)) # Aumentar el tamaño para mejor visualización de etiquetas
sns.heatmap(
    conf_matrix,
    annot=True,     # Mostrar los valores en las celdas
    fmt="d",        # Formato de los números como enteros
    cmap="Blues",   # Mapa de color
    xticklabels=class_names, # Etiquetas del eje X (predicciones)
    yticklabels=class_names  # Etiquetas del eje Y (valores reales)
)
plt.title("Matriz de Confusión - Árbol de Decisión (Conjunto Held-out)", fontsize=16)
plt.xlabel("Clase Predicha", fontsize=12)
plt.ylabel("Clase Real", fontsize=12)
plt.tight_layout() # Ajusta el layout para que las etiquetas no se corten
plt.show()

# Análisis de la matriz de confusión
print("\n--- Análisis de la Matriz de Confusión ---")
print("Observa los valores fuera de la diagonal principal. Estos indican las clases que el modelo confunde.")
print("Por ejemplo, un valor alto en la fila 'X' y columna 'Y' significa que muchas instancias de la clase 'X' real")
print("fueron incorrectamente predichas como clase 'Y'.")
print("\nRelaciona estos resultados con la similitud visual observada en el Ejercicio 1 (Análisis Exploratorio).")
print("¿Las clases que el modelo confunde más son aquellas que visualmente son más similares?")

# Reporte de Clasificación detallado (precisión, recall, f1-score por clase)
print("\n--- Reporte de Clasificación en el conjunto held-out ---")
print(classification_report(y_heldout, y_pred_heldout, target_names=class_names))

print("\nEste reporte proporciona métricas detalladas para cada clase, lo cual es útil para")
print("entender el rendimiento del modelo en clases específicas, especialmente si alguna estuviera desbalanceada.")
# %%
