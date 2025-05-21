import pandas as pd 
import matplotlib.pyplot as plt 
import phonenumbers
from phonenumbers import geocoder, carrier
from inline_sql import sql, sql_val
import duckdb

basePath = "./"
invalidos = []
establecimientos = pd.read_csv(basePath + "/estableciminetos.csv", encoding="latin1", low_memory=False)
bibliotecas = pd.read_csv(basePath + "/bibliotecas.csv")

def tabla_relaciones_EE():
    establecimientosEducativosComunes = "EEcomunes.csv"
    filtrados = []
    filtrados.append(establecimientos.iloc[4, [4, 8, 9, 17, 18, 19]])
    for i in range(5, len(establecimientos)):
        if establecimientos.iloc[i, 13] == "1":
            filtrados.append(establecimientos.iloc[i, [4, 8, 9, 17, 18, 19]])
    filtrados_df = pd.DataFrame(filtrados)
    columnas = {filtrados_df.columns[0]: "id_departamento", filtrados_df.columns[1]:"nombre", filtrados_df.columns[2]:"domicilio", filtrados_df.columns[3]:"jardin_infantes", filtrados_df.columns[4]: "primario", filtrados_df.columns[5]:"secundario"}
    filtrados_df = filtrados_df.rename(columns=columnas)
    filtrados_df = filtrados_df[filtrados_df["secundario"] != "Secundario"]
    filtrados_df.to_csv(establecimientosEducativosComunes, encoding="utf-8", index=True, index_label="id_establecimiento")

def tabla_relaciones_BP():
    bibliotecasPopulares = "BP.csv"
    filtrados = []
    for i in range(1, len(bibliotecas)):
        fila = bibliotecas.iloc[i]
        if isinstance(fila.iloc[15], str) and "@" in fila.iloc[15]:
            filtrados.append((fila.iloc[2], fila.iloc[15].split('@')[1].split('.')[0], fila.iloc[22]))
        else:
            filtrados.append((fila.iloc[2], fila.iloc[15], fila.iloc[22]))

    filtrados_df = pd.DataFrame(filtrados)
    columnas = {filtrados_df.columns[0]: "nombre", filtrados_df.columns[1]: "mail", filtrados_df.columns[2]: "fecha_fundacion"}
    filtrados_df = filtrados_df.rename(columns=columnas)
    filtrados_df.to_csv(bibliotecasPopulares, encoding="utf-8", index=True, index_label="id_biblioteca")
    return filtrados_df

def obtener_provincia(codigo):
    localEstablecimiento =  establecimientos.iloc[5 :, 0: 5].dropna(thresh=1)
    columnas = {localEstablecimiento.columns[0]:"jurisdiccion", localEstablecimiento.columns[1]:"sector", localEstablecimiento.columns[2]:"ambito", localEstablecimiento.columns[3]:"departamento", localEstablecimiento.columns[4]:"codigo_area"}
    localEstablecimiento =  localEstablecimiento.rename(columns=columnas)
    coincidencias = localEstablecimiento[localEstablecimiento["codigo_area"] == str(codigo)]
    codigosCaba = ['02007', '02014', '02021', '02028', '02035', '02042', '02049', '02056', '02063', '02070', '02077', '02084', '02091', '02098']
    if not coincidencias.empty:
        return coincidencias["jurisdiccion"].iloc[0]
    elif codigo == "94008":
        invalidos.append(codigo)
        return "Tierra del Fuego"
    elif codigo in codigosCaba:
        return "Ciudad de Buenos Aires"
    else :
        return "desconocida"

def tabla_departamentos():
    poblacion = pd.read_csv(basePath + "/poblacion.csv")
    departamentos = "departamentos.csv"
    col_dep = ["id_departamento", "nombre", "poblacion jardin", "poblacion primario", "poblacion secundario"]
    df_departamentos = pd.DataFrame(columns=col_dep)
    poblacion = poblacion.iloc[12 :, 1: 3].dropna(thresh=1)
    columnas = ["edad", "casos"]
    poblacion = poblacion.rename(columns=lambda c: columnas[int(c.split(" ")[1]) - 1])
    poblacion = poblacion[poblacion["edad"] != "Edad"]
    poblacion = poblacion[poblacion["edad"] != "RESUMEN"]
    poblacion = poblacion[poblacion["edad"] != "Total"].reset_index(drop=True)
    index = 1
    area = poblacion["edad"][0].split(" ")[2]
    nombre = poblacion["casos"][0]
    cuenta_jardin = 0 
    cuenta_primario = 0 
    cuenta_secundario = 0
    while index < len(poblacion):
        if "#" in poblacion["edad"][index]:
            df_departamentos.loc[len(df_departamentos)] = [area, nombre, cuenta_jardin, cuenta_primario, cuenta_secundario]
            area = poblacion["edad"][index].split(" ")[2]
            nombre = poblacion["casos"][index]
            cuenta_jardin = 0 
            cuenta_primario = 0 
            cuenta_secundario = 0
        elif 3 <= int(poblacion["edad"][index]) <= 5  :
            valor = str(poblacion["casos"][index]).replace(" ", "").strip()
            cuenta_jardin += int(valor)
        elif 5 <=int(poblacion["edad"][index]) <= 12: 
            valor = str(poblacion["casos"][index]).replace(" ", "").strip()
            cuenta_primario += int(valor)
        elif 12 <= int(poblacion["edad"][index]) <= 18: 
            valor = str(poblacion["casos"][index]).replace(" ", "").strip()
            cuenta_secundario += int(valor)
        index += 1
    df_departamentos['provincia'] = df_departamentos['id_departamento'].apply(obtener_provincia)
    df_departamentos.to_csv(departamentos, index=False, encoding="utf-8")
    
    # while poblacion.iloc[i].astype(str).str.startswith("AREA") == False:
    #     tabla_por_departamento = [area, nombre, poblacion_0_6, poblacion_6_12, ppoblacion_12_18]
    #     i += 1
    #     tabla_por_departamento = []
    #     if bibliotecas.iloc[i, 10] == "1":
    #         filtrados.append(bibliotecas.iloc[i, [2, 9, 10, 15, 22]])
    # area = poblacion.iloc[i, 2].split("       ")[2]
    # filtrados_df = pd.DataFrame(filtrados)
    # filtrados_df.to_csv(departamentos, index=False, encoding="utf-8")
    
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

#hacer una tabla provincia departameento cantidad de jardines, poblacion de edad de jardin, primaria 

#print(consultaSQL.head(8))
#print(consultaSQL)

#dataframeResultado = sql^ consultaSQL

#print(dataframeResultado)

#tabla_relaciones_EE()
#tabla_relaciones_BP()
#tabla_departamentos()

# Leer el archivo BP.csv
bp = pd.read_csv("BP.csv")

# Registrar el DataFrame como tabla en DuckDB
# Leer los archivos
ee = pd.read_csv("EEcomunes.csv")
deptos = pd.read_csv("departamentos.csv")

# Conectar y registrar las tablas
con = duckdb.connect()
con.register('bp', bp)
con.register('ee', ee)
con.register('deptos', deptos)

consultai = duckdb.query("""
SELECT
    deptos.provincia AS Provincia,
    deptos.nombre AS Departamento,
    COUNT(CASE WHEN ee.jardin_infantes = '1' THEN 1 END) AS Jardines,
    deptos."poblacion jardin" AS "Población Jardin",
    COUNT(CASE WHEN ee.primario = '1' THEN 1 END) AS Primarias,
    deptos."poblacion primario" AS "Población Primaria",
    COUNT(CASE WHEN ee.secundario = '1' THEN 1 END) AS Secundarios,
    deptos."poblacion secundario" AS "Población Secundaria"
FROM deptos
LEFT OUTER JOIN ee ON deptos.id_departamento = ee.id_departamento
GROUP BY deptos.provincia, deptos.nombre, deptos."poblacion jardin", deptos."poblacion primario", deptos."poblacion secundario"
ORDER BY deptos.provincia ASC, Primarias DESC
""").df()

consulta1 = "consulta1.csv"
consultai.to_csv(consulta1, encoding="utf-8", index=False)

consultaii = duckdb.query("""
SELECT
    deptos.provincia AS Provincia,
    deptos.nombre AS Departamento,
    COUNT(CASE WHEN bp.fecha_fundacion >= '1950-01-01' THEN 1 END) AS "Cantidad de BP fundadas desde 1950"
FROM deptos
LEFT OUTER JOIN bp ON deptos.id_departamento = bp.nombre
GROUP BY deptos.provincia, deptos.nombre
ORDER BY deptos.provincia ASC, "Cantidad de BP fundadas desde 1950" DESC
""").df()

consulta2 = "consulta2.csv"
consultaii.to_csv("consulta2.csv", encoding="utf-8", index=False)

consultaiii = duckdb.query("""
SELECT
    deptos.provincia AS Provincia,
    deptos.nombre AS Departamento,
    COUNT(DISTINCT ee.id_establecimiento) AS Cant_EE,
    COUNT(DISTINCT bp.id_biblioteca) AS Cant_BP,
    (deptos."poblacion jardin" + deptos."poblacion primario" + deptos."poblacion secundario") AS Población
FROM deptos
LEFT OUTER JOIN ee ON deptos.id_departamento = ee.id_departamento
LEFT OUTER JOIN bp ON deptos.id_departamento = bp.nombre
GROUP BY deptos.provincia, deptos.nombre, deptos."poblacion jardin", deptos."poblacion primario", deptos."poblacion secundario"
ORDER BY Cant_EE DESC, Cant_BP DESC, Provincia ASC, Departamento ASC
""").df()

consulta3 = "consulta3.csv"
consultaiii.to_csv(consulta3, encoding="utf-8", index=False)

consultaiv = duckdb.query("""
SELECT
    t1.provincia AS Provincia,
    t1.nombre AS Departamento,
    t1.mail AS "Dominio más frecuente en BP"
FROM (
    SELECT
        deptos.provincia,
        deptos.nombre,
        bp.mail,
        COUNT() AS cantidad
    FROM deptos
    LEFT OUTER JOIN bp ON deptos.id_departamento = bp.nombre
    WHERE bp.mail IS NOT NULL AND bp.mail <> ''
    GROUP BY deptos.provincia, deptos.nombre, bp.mail
) t1
INNER JOIN (
    SELECT
        provincia,
        nombre,
        MAX(cantidad) AS max_cantidad
    FROM (
        SELECT
            deptos.provincia AS provincia,
            deptos.nombre AS nombre,
            bp.mail,
            COUNT() AS cantidad
        FROM deptos
        LEFT OUTER JOIN bp ON deptos.id_departamento = bp.nombre
        WHERE bp.mail IS NOT NULL AND bp.mail <> ''
        GROUP BY deptos.provincia, deptos.nombre, bp.mail
    ) t2
    GROUP BY provincia, nombre
) t3
ON t1.provincia = t3.provincia AND t1.nombre = t3.nombre AND t1.cantidad = t3.max_cantidad
ORDER BY Provincia ASC, Departamento ASC
""").df()

consulta4 = "consulta4.csv"
consultaiv.to_csv(consulta4, encoding="utf-8", index=False)

bp_segun_provincia = {}
deptos = pd.read_csv("departamentos.csv", dtype={"id_departamento": str})

for i in range(len(bp)):
    id_dep = str(bp.nombre[i])
    filtro = deptos[deptos["id_departamento"] == id_dep]
    if not filtro.empty:
        provincia = filtro["provincia"].values[0]
        if provincia in bp_segun_provincia:
            bp_segun_provincia[provincia] += 1
        else:
            bp_segun_provincia[provincia] = 1
    otroFiltro = deptos[deptos["id_departamento"] == "0" + id_dep]
    if not otroFiltro.empty:
        provincia = otroFiltro["provincia"].values[0]
        if provincia in bp_segun_provincia:
            bp_segun_provincia[provincia] += 1
        else:
            bp_segun_provincia[provincia] = 1
# NO APARECEN DE CABA NO APARECEN DE CABA NO APARECEN DE CABA NO APARECEN DE CABA NO APARECEN DE CABA

bp_segun_provincia = sorted(bp_segun_provincia.items(), key=lambda x: x[1], reverse=True)
print(bp_segun_provincia)

fig, ax = plt.subplots()
ax.bar(data=bp_segun_provincia, x=[x[0] for x in bp_segun_provincia], height=[x[1] for x in bp_segun_provincia])
ax.set_xlabel("Provincias")
ax.set_ylabel("Cantidad de Bibliotecas Populares")
ax.set_title("Cantidad de Bibliotecas Populares por Provincia")
plt.tight_layout()
plt.show()