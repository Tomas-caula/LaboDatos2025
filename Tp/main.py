import pandas as pd 
import matplotlib.pyplot as plt 
import phonenumbers
from phonenumbers import geocoder, carrier
from inline_sql import sql, sql_val
import duckdb




basePathTomas = "C:/Users/tomas/OneDrive/Documents/GitHub/LaboDatos2025/Tp/"


basePath = basePathTomas


establecimientos = pd.read_csv(basePath + "estableciminetos.csv", encoding="latin1", low_memory=False)
bibliotecas = pd.read_csv(basePath + "/bibliotecas.csv")
poblacion = pd.read_csv(basePath + "/poblacion.csv")

def analizarNumeros (): 
    validos = 0
    index = 5
    while(index < 64716):
        j = 0 
        cond = False
        while(j <= 2 ):
            condicion = establecimientos.iloc[index, 13 + j]
            j+= 1
            if(condicion == "1"): 
                cond = True
        if cond: 
            validos += 1
        cond = False
        index += 1


    print(validos)

def analizarMail ():
    df = bibliotecas["mail"]
    print("null mail", len(bibliotecas) - df.count())
    print("mails", df.count())

poblacion = poblacion.iloc[12 :, 1: 5].dropna(thresh=1)
columnas = ["edad", "casos", "porcentaje", "acumulado"]
poblacion = poblacion.rename(columns=lambda c: columnas[int(c.split(" ")[1]) - 1])
poblacion = poblacion[poblacion["edad"] != "Edad"]
poblacion = poblacion[poblacion["edad"] != "Total"].reset_index(drop=True)


#hacer una tabla provincia departamente cantidad de jardines, poblacion de edad de jardin, primaria 

#print(consultaSQL.head(8))


print(consultaSQL)

#dataframeResultado = sql^ consultaSQL

#print(dataframeResultado)







