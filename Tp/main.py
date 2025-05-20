import pandas as pd 
import matplotlib.pyplot as plt 
import phonenumbers
from phonenumbers import geocoder, carrier
from inline_sql import sql, sql_val
import duckdb

basePathTomas = "C:/Users/tomas/OneDrive/Documents/GitHub/LaboDatos2025/Tp/"
basePathIoel = "D:/Users/Usuario/Documents/GitHub/LaboDatos2025/Tp"
basePathFran = "C:/Users/franc/Documents/GitHub/LaboDatos2025/Tp"

basePath = basePathFran
# basePath = basePathTomas  
# basePath = basePathIoel

establecimientos = pd.read_csv(basePath + "/estableciminetos.csv", encoding="latin1", low_memory=False)
bibliotecas = pd.read_csv(basePath + "/bibliotecas.csv")
poblacion = pd.read_csv(basePath + "/poblacion.csv")

def tabla_relaciones_EE():
    
    establecimientosEducativosComunes = "EEcomunes.csv"

    filtrados = []
    filtrados.append(establecimientos.iloc[4, [4, 8, 9, 16, 17, 18, 19]])
    for i in range(5, len(estableciminetos)):
        if establecimientos.iloc[i, 13] == "1":
            filtrados.append(establecimientos.iloc[i, [4, 8, 9, 16, 17, 18, 19]])
        
    filtrados_df = pd.DataFrame(filtrados)
    filtrados_df.to_csv(establecimientosEducativosComunes, index=False, encoding="utf-8")

def tabla_relaciones_BP():
    
    bibliotecasPopulares = "BP.csv"
    filtrados = []
    for i in range(1, len(bibliotecas)):
      filtrados.append(bibliotecas.iloc[i, [2, 9, 10, 15, 22]])
    filtrados_df = pd.DataFrame(filtrados)
    filtrados_df.to_csv(bibliotecasPopulares, index=False, encoding="utf-8")

def tabla_departamentos():
    departamentos = "departamentos.csv"
    tabla_0 = ["id_departamento", "nombre", "poblacion_0_6", "poblacion_6_12", "poblacion_12_18"]
    filtrados = []
    poblacion = poblacion.iloc[12 :, 1: 3].dropna(thresh=1)
    columnas = ["edad", "casos"]
    poblacion = poblacion.rename(columns=lambda c: columnas[int(c.split(" ")[1]) - 1])
    poblacion = poblacion[poblacion["edad"] != "Edad"]
    poblacion = poblacion[poblacion["edad"] != "Total"].reset_index(drop=True)
    
    while poblacion.iloc[i].astype(str).str.startswith("AREA") == False:
        tabla_por_departamento = [area, nombre, poblacion_0_6, poblacion_6_12, ppoblacion_12_18]
        i += 1
        tabla_por_departamento = []
        if bibliotecas.iloc[i, 10] == "1":
            filtrados.append(bibliotecas.iloc[i, [2, 9, 10, 15, 22]])
    area = poblacion.iloc[i, 2].split("       ")[2]
    filtrados_df = pd.DataFrame(filtrados)
    filtrados_df.to_csv(departamentos, index=False, encoding="utf-8")
    
# def analizarNumeros (): 
#     validos = 0
#     index = 5
#     while(index < 64716):
#         j = 0 
#         cond = False
#         while(j <= 2 ):
#             condicion = establecimientos.iloc[index, 13 + j]
#             j+= 1
#             if(condicion == "1"): 
#                 cond = True
#         if cond: 
#             validos += 1
#         cond = False
#         index += 1


#     print(validos)


def analizarMail ():
    df = bibliotecas["mail"]
    print("null mail", len(bibliotecas) - df.count())
    print("mails", df.count())





#hacer una tabla provincia departamente cantidad de jardines, poblacion de edad de jardin, primaria 

#print(consultaSQL.head(8))
print(poblacion)

#print(consultaSQL)

#dataframeResultado = sql^ consultaSQL

#print(dataframeResultado)






