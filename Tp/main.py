import pandas as pd 
import matplotlib.pyplot as plt 



def analizarNumeros (): 
    tables = pd.read_csv("C:/Users/tomas/OneDrive/Documents/GitHub/LaboDatos2025/Tp/estableciminetos.csv", encoding="latin1", low_memory=False)

    numerosValidos = 0
    numerosInvalidos = 0 
    index = 5


    while(index < 64716):
        numer = tables.iloc[index, 11]
        #en realidad numer == "011 S/D" or numer == "1 1" or numer == "0 0" es ridculo porque la condicion dejada implica esto
        if (len(numer.strip()) < 19):
            numerosInvalidos += 1
        else: 
            numerosValidos +=1
        index += 1
    

    print(numerosInvalidos)
    print(numerosValidos)

def analizarMail ():

    tables = pd.read_csv("C:/Users/tomas/OneDrive/Documents/GitHub/LaboDatos2025/Tp/bibliotecas-populares.csv")
    df = tables["mail"]
    print("null mail", len(tables) - df.count())
    print("mails", df.count())
analizarMail()


