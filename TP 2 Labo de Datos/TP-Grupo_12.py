#Grupo TP02 - 12
#Francisco Catri
#Ioel Failenbogen
#Tomas Benjamin Caula
#Codigo completo del TP 2 de Laboratorio de Datos

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

# Ejercicio 1
data_df = pd.read_csv("Fashion-MNIST.csv", index_col=0)

print(data_df['label'].value_counts()) # Cantidad de imagenes por clase
#print(data_df.isnull().sum())  # Verificar si hay valores nulos

# Asignar X e Y
y = data_df['label'].astype(int)
X = data_df.drop('label', axis=1)

# Normalizar
X = X / 255.0

# Calcular desviacion estandar por pixel
std_por_pixel = np.std(X, axis=0).values  # .values para convertir de pandas Series a NumPy array

# Reshape
std_image = std_por_pixel.reshape(28, 28)

# Visualizar
plt.figure(figsize=(6, 6))
sns.heatmap(std_image, cmap='viridis', cbar=True)
plt.title('Desviacion estandar por pixel (dataset completo)')
plt.axis('off')
plt.show()

# Hay 10 clases en Fashion-MNIST (etiquetas de 0 a 9)
for clase in range(10):
    # Filtrar las filas que corresponden a esta clase
    X_clase = X[y == clase]
    
    # Calcular la desviacion estandar de cada pixel para esta clase
    std_pixel_clase = np.std(X_clase, axis=0).values.reshape(28, 28)
    
    

    #if clase in [0, 8]:
        # Obtener los índices de los 5 valores más altos y más bajos
    #    indices_max = np.argsort(std_pixel_clase.flatten())[-5:]
    #    indices_min = np.argsort(std_pixel_clase.flatten())[:5]
        
    #    print(f"\nClase {clase}:")
    #    print("5 atributos con mayor desviación estándar:")
    #    for idx in indices_max:
    #        print(f"Atributo {idx}: {std_pixel_clase.flatten()[idx]:.4f}")
            
    #    print("\n5 atributos con menor desviación estándar:")
    #    for idx in indices_min:
    #        print(f"Atributo {idx}: {std_pixel_clase.flatten()[idx]:.4f}")
    # Visualizacion
    plt.figure(figsize=(4, 4))
    sns.heatmap(std_pixel_clase, cmap='viridis', cbar=True)
    plt.title(f'Desviacion estandar - Clase {clase}')
    plt.axis('off')
    plt.show()

# Diccionario para guardar imagenes promedio por clase
promedios_por_clase = {}

for clase in range(10):
    # Filtrar imagenes de la clase actual
    X_clase = X[y == clase]

    # Calcular el promedio por pixel (valor medio en cada posicion)
    promedio_pixel_clase = np.mean(X_clase, axis=0).values.reshape(28, 28)

    # Guardar en el diccionario
    promedios_por_clase[clase] = promedio_pixel_clase

    # Visualizacion
    plt.figure(figsize=(4, 4))
    sns.heatmap(promedio_pixel_clase, cmap='gray', cbar=False)
    plt.title(f'Imagen promedio - Clase {clase}')
    plt.axis('off')
    plt.show()

# Crear matriz 10x10 para guardar distancias entre imagenes promedio
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
plt.title("Distancia Euclidiana entre imagenes promedio de cada clase")
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
# Cargar el dataset
data_df = pd.read_csv("Fashion-MNIST.csv", index_col=0)

# Crear subconjunto with solo las clases 0 y 8
data_df_0_8 = data_df[data_df['label'].isin([0, 8])]

# Analisis del balance de clases

conteo_clases = data_df_0_8['label'].value_counts()
print(conteo_clases)

porcentaje_clases = (conteo_clases / len(data_df_0_8)) * 100
print(porcentaje_clases)

#Aca pudiomos observar que las clases estan balanceadas, por lo que no es necesario balancearlas

#Visualizar el balance de clases
 
plt.figure()
sns.countplot(data=data_df_0_8, x='label')
plt.title('Distribucion de clases 0 y 8')
plt.xlabel('Clase')
plt.ylabel('Cantidad de imagenes')
plt.show()

# Separar caracteristicas (X) y etiquetas (y)
X = data_df_0_8.drop('label', axis=1)
y = data_df_0_8['label']

# Crear conjunto de entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Funcion para evaluar diferentes subconjuntos de atributos
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
    'Subconjunto 1': [329, 302, 314],  # Primeros 3 pixeles
    'Subconjunto 2': [0, 756, 28],  # Pixeles del medio
    'Subconjunto 3': [741, 740, 737],  # Ultimos 3 pixeles
    'Subconjunto 4': [0,1,27],  # Todos los pixeles
}

subconjuntos_6_atributos = {
   'Subconjunto 1': [329, 302, 314, 741, 740, 737],  # Mas importantes
   'Subconjunto 2': [0, 756, 28,0,1,27],  # Menos importantes

}

# <- Definimos todos los subconjuntos de atributos que vamos a usar

# Evaluamos subconjuntos de 3 atributos
resultados_3 = evaluar_subconjuntos(X_train, X_test, y_train, y_test, subconjuntos_3_atributos)
print("\nResultados para subconjuntos de 3 atributos:")
for nombre, accuracy in resultados_3.items():
    print(f"{nombre}: {accuracy:.4f}")

# Evaluamos subconjuntos de 5 atributos
resultados_5 = evaluar_subconjuntos(X_train, X_test, y_train, y_test, subconjuntos_6_atributos)
print("\nResultados para subconjuntos de 5 atributos:")
for nombre, accuracy in resultados_5.items():
    print(f"{nombre}: {accuracy:.4f}")

# Visualizamos resultados
plt.figure(figsize=(12, 6))

# Grafico para 3 atributos
plt.subplot(1, 2, 1)
plt.bar(resultados_3.keys(), resultados_3.values())
plt.title('Precision con 3 atributos')
plt.xticks(rotation=45)
plt.ylabel('Precision')

# Grafico para 5 atributos
plt.subplot(1, 2, 2)
plt.bar(resultados_5.keys(), resultados_5.values())
plt.title('Precision con 5 atributos')
plt.xticks(rotation=45)
plt.ylabel('Precision')



#Probando con k diferentes

const_k = [1, 3, 5, 7, 9]
# Almacenar resultados para graficar
resultados_3_atributos = {k: {} for k in const_k}
resultados_6_atributos = {k: {} for k in const_k}

# Evaluamos y almacenamos resultados para 3 atributos
for k in const_k:
    resultados_k = evaluar_subconjuntos(X_train, X_test, y_train, y_test, subconjuntos_3_atributos, k)
    resultados_3_atributos[k] = resultados_k

# Evaluamos y almacenamos resultados para 5 atributos
for k in const_k:
    resultados_k = evaluar_subconjuntos(X_train, X_test, y_train, y_test, subconjuntos_6_atributos, k)
    resultados_6_atributos[k] = resultados_k

# Crear grafico
plt.figure(figsize=(15, 6))

# Grafico para 3 atributos
plt.subplot(1, 2, 1)
for subconjunto in subconjuntos_3_atributos.keys():
    valores = [resultados_3_atributos[k][subconjunto] for k in const_k]
    plt.plot(const_k, valores, marker='o', label=subconjunto)
plt.title('Precision vs k para 3 atributos')
plt.xlabel('Valor de k')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)


# Grafico para 5 atributos
plt.subplot(1, 2, 2)
for subconjunto in subconjuntos_6_atributos.keys():
    valores = [resultados_6_atributos[k][subconjunto] for k in const_k]
    plt.plot(const_k, valores, marker='o', label=subconjunto)
plt.title('Precision vs k para 5 atributos')
plt.xlabel('Valor de k')
plt.ylabel('Precision')
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
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Paso 0: Cargar y preparar el dataset ---

data_df = pd.read_csv("Fashion-MNIST.csv", index_col=0)


# Separar caracteristicas (X) y etiquetas (y)
# Normalizar los valores de los pixeles de 0-255 a 0-1.0 para mejorar el rendimiento del modelo.
X = data_df.drop('label', axis=1) / 255.0
y = data_df['label'].astype(int) # Asegurar que las etiquetas sean enteros

# Definir los nombres de las clases para una mejor interpretacion de los resultados
class_names = [
    "T-shirt/top",  
    "Trouser",      
    "Pullover",    
    "Dress",
    "Coat",        
    "Sandal",      
    "Shirt",     
    "Sneaker",      
    "Bag",      
    "Ankle boot"   
]

# Paso 1: Division en conjunto de desarrollo y held-out ->

# Dividir los datos en un conjunto de desarrollo (para entrenamiento y validacion cruzada)
# y un conjunto held-out (para evaluacion final imparcial).
# stratify=y asegura que la proporcion de clases sea la misma en ambos conjuntos (muestreo estratificado).
X_dev, X_heldout, y_dev, y_heldout = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Tamaño del conjunto de desarrollo: {X_dev.shape[0]} muestras")
print(f"Tamaño del conjunto held-out: {X_heldout.shape[0]} muestras")

# <- Paso 1: Division en conjunto de desarrollo y held-out 

#  Paso 2: Seleccion de hiperparametro max_depth usando validacion cruzada ->

param_grid = {'max_depth': range(1, 11)}

# Los datos de desarrollo los dividiremos en 5 folds.
# Con StratifiedKFold aseguramos que la proporcion de clases sea la misma en cada fold.
# El enunciado no pide que sea estratificado pero por las dudas y para un mejor rendimiento se      
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Iinicializamos el modelo DecisionTreeClassifier.
dt = DecisionTreeClassifier(random_state=42)

grid_search = GridSearchCV(dt, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_dev, y_dev)

#print("Mejor max_depth:", grid_search.best_params_['max_depth'])
#print(f"Mejor exactitud promedio (validacion cruzada): {grid_search.best_score_:.4f}")

# Visualizar la exactitud promedio de la validacion cruzada para cada max_depth
results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(10, 6))
plt.plot(results['param_max_depth'], results['mean_test_score'], marker='o', linestyle='-')
plt.title('Exactitud Promedio de Validacion Cruzada vs. Max Depth', fontsize=16)
plt.xlabel('Max Depth (Profundidad Maxima del arbol)', fontsize=12)
plt.ylabel('Exactitud Promedio (Validacion Cruzada)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(1, 11)) # Asegura que todos los valores de 1 a 10 esten en el eje X
plt.tight_layout()
plt.show()


#<- Paso 2: Seleccion de hiperparametro max_depth usando validacion cruzada

#Paso 3: Entrenar el modelo final con el mejor max_depth en todo el conjunto de desarrollo ->

# Obtener el mejor valor de max_depth determinado por GridSearchCV.
mejor_max_depth = grid_search.best_params_['max_depth']

# Inicializar y entrenar el modelo de arbol de Decision final.
# este modelo lo entrenamo con TODOS los datos del conjunto de desarrollo (X_dev, y_dev).
# PAra  asegurarnos que el modelo final aprenda de la mayor cantidad de datos posible antes de la evaluacion final.
final_model = DecisionTreeClassifier(max_depth=mejor_max_depth, random_state=42)
final_model.fit(X_dev, y_dev)

print(f"\nModelo final entrenado con max_depth = {mejor_max_depth} en el conjunto de desarrollo completo.")

#<- Paso 3: Entrenar el modelo final con el mejor max_depth en todo el conjunto de desarrollo 

# Paso 4: Evaluar el modelo en el conjunto held-out ->

# Realizar predicciones en el conjunto held-out.
# Esta es la evaluacion imparcial del rendimiento del modelo en datos no vistos.
y_pred_heldout = final_model.predict(X_heldout)

# Calcular la exactitud en el conjunto held-out.
accuracy_heldout = accuracy_score(y_heldout, y_pred_heldout)
#print(f"Exactitud held-out: {accuracy_heldout:.4f}")

#<- Paso 4: Evaluar el modelo en el conjunto held-out

# Paso 5: Matriz de confusion y reporte de clasificacion ->

# Calcular la matriz de confusion.
conf_matrix = confusion_matrix(y_heldout, y_pred_heldout)

# Visualizar la matriz de confusion con nombres de clases.
plt.figure(figsize=(10, 8)) # Aumentar el tamaño para mejor visualizacion de etiquetas
sns.heatmap(
    conf_matrix,
    annot=True,     # Mostrar los valores en las celdas
    fmt="d",   
    cmap="Blues", 
    xticklabels=class_names, # Etiquetas del eje X (predicciones)
    yticklabels=class_names  # Etiquetas del eje Y (valores reales)
)
plt.title("Matriz de Confusion - arbol de Decision (Conjunto Held-out)", fontsize=16)
plt.xlabel("Clase Predicha", fontsize=12)
plt.ylabel("Clase Real", fontsize=12)
plt.tight_layout() # Ajusta el layout para que las etiquetas no se corten
plt.show()

print(classification_report(y_heldout, y_pred_heldout, target_names=class_names))
# %%
