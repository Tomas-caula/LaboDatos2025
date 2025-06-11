import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Cargar el dataset
data_df = pd.read_csv("Fashion-MNIST.csv", index_col=0)

# Crear subconjunto con solo las clases 0 y 8
data_df_0_8 = data_df[data_df['label'].isin([0, 8])]

# Análisis del balance de clases
conteo_clases = data_df_0_8['label'].value_counts()
print("\nCantidad de muestras por clase:")
print(conteo_clases)

# Calcular el porcentaje de cada clase
porcentaje_clases = (conteo_clases / len(data_df_0_8)) * 100
print("\nPorcentaje de muestras por clase:")
print(porcentaje_clases)

# Visualizar el balance de clases
plt.figure(figsize=(8, 6))
sns.countplot(data=data_df_0_8, x='label')
plt.title('Distribución de clases 0 y 8')
plt.xlabel('Clase')
plt.ylabel('Cantidad de imágenes')
plt.show()

# Determinar si está balanceado
diferencia_porcentual = abs(porcentaje_clases[0] - porcentaje_clases[8])
print(f"\nDiferencia porcentual entre clases: {diferencia_porcentual:.2f}%")

if diferencia_porcentual < 5:
    print("El dataset está balanceado (diferencia menor al 5%)")
else:
    print("El dataset está desbalanceado (diferencia mayor al 5%)")

# Separar características (X) y etiquetas (y)
X = data_df_0_8.drop('label', axis=1)
y = data_df_0_8['label']

# Crear conjunto de entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Mostrar información sobre los conjuntos creados
print("\nInformación de los conjuntos de datos:")
print(f"Tamaño del conjunto original: {len(data_df_0_8)}")
print(f"Tamaño del conjunto de entrenamiento: {len(X_train)} ({len(X_train)/len(data_df_0_8)*100:.1f}%)")
print(f"Tamaño del conjunto de prueba: {len(X_test)} ({len(X_test)/len(data_df_0_8)*100:.1f}%)")

# Verificar el balance en el conjunto de entrenamiento
conteo_train = y_train.value_counts()
print("\nDistribución de clases en el conjunto de entrenamiento:")
print(conteo_train)
print("\nPorcentajes en el conjunto de entrenamiento:")
print((conteo_train / len(y_train) * 100).round(2))

# Comparar rendimiento usando todos los píxeles vs 3 píxeles seleccionados
k_values = [3, 5, 7]
resultados_todos_pixeles = []
resultados_3_pixeles = []

for k in k_values:
    # Modelo con todos los píxeles
    knn_todos = KNeighborsClassifier(n_neighbors=k)
    knn_todos.fit(X_train, y_train)
    y_pred_todos = knn_todos.predict(X_test)
    acc_todos = metrics.accuracy_score(y_test, y_pred_todos)
    resultados_todos_pixeles.append(acc_todos)
    
    # Modelo con 3 píxeles seleccionados
    knn_3 = KNeighborsClassifier(n_neighbors=k)
    knn_3.fit(X_train.iloc[:, [0, 392, 783]], y_train)
    y_pred_3 = knn_3.predict(X_test.iloc[:, [0, 392, 783]])
    acc_3 = metrics.accuracy_score(y_test, y_pred_3)
    resultados_3_pixeles.append(acc_3)

# Visualizar comparación
plt.figure(figsize=(10, 6))
x = np.arange(len(k_values))
width = 0.35

plt.bar(x - width/2, resultados_todos_pixeles, width, label='Todos los píxeles')
plt.bar(x + width/2, resultados_3_pixeles, width, label='3 píxeles seleccionados')

plt.xlabel('Valor de k')
plt.ylabel('Precisión')
plt.title('Comparación de precisión: Todos los píxeles vs 3 píxeles')
plt.xticks(x, k_values)
plt.legend()
plt.show()

# Imprimir resultados numéricos
print("\nResultados de precisión:")
print("\nTodos los píxeles:")
for k, acc in zip(k_values, resultados_todos_pixeles):
    print(f"k={k}: {acc:.4f}")

print("\n3 píxeles seleccionados:")
for k, acc in zip(k_values, resultados_3_pixeles):
    print(f"k={k}: {acc:.4f}") 