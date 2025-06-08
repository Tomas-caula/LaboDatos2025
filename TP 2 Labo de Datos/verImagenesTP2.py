#!/usr/bin/env python
# coding: utf-8

# Visualizar imágenes


# %% Import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns

# %% Load dataset

data_df = pd.read_csv("Fashion-MNIST.csv", index_col=0)
print(data_df.head())


# %% Select single image and convert to 28x28 array

img_nbr = 3

# keep label out
img = np.array(data_df.iloc[img_nbr, :-1]).reshape(28, 28)


# %% Plot image

plt.imshow(img, cmap="gray")


# %%
X = data_df.loc[:23333, :"pixel783"]
Y = data_df.loc[:23333, "label":]


H = data_df.loc[23333:33333, :"pixel783"]
J = data_df.loc[23333:33333, "label":]
print(Y)
# %%
model = KNeighborsClassifier(n_neighbors=4)  # modelo en abstracto
model.fit(X, Y)  # entreno el modelo con los datos X e Y

# %%
Y_pred = model.predict(H)  # me fijo qué clases les asigna el modelo a mis datos
metrics.accuracy_score(J, Y_pred)
# metrics.confusion_matrix(J, Y_pred)

# %%
cm = confusion_matrix(J, Y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.show()

# %%
