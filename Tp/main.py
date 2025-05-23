import pandas as pd
import matplotlib.pyplot as plt
import duckdb
import seaborn as sns

basePath = "./"
invalidos = []
establecimientos = pd.read_csv(
    basePath + "/estableciminetos.csv", encoding="latin1", low_memory=False
)
bibliotecas = pd.read_csv(basePath + "/bibliotecas.csv")


diccionario_caba = {
    "02007": "02101",
    "02014": "02102",
    "02021": "02103",
    "02028": "02104",
    "02035": "02105",
    "02042": "02106",
    "02049": "02107",
    "02056": "02108",
    "02063": "02109",
    "02070": "02110",
    "02077": "02111",
    "02084": "02112",
    "02091": "02113",
    "02098": "02114",
    "02105": "02105",
}


codigos_postales = [
    "02103",
    "02106",
    "02110",
    "02108",
    "02104",
    "02108",
    "02101",
    "02112",
    "02107",
    "02101",
    "02107",
    "02111",
    "02114",
    "02112",
    "02115",
    "02115",
    "02112",
    "02109",
    "02111",
    "02115",
    "02103",
    "02105",
    "02111",
    "02106",
    "02109",
    "02115",
    "02105",
    "02109",
    "02104",
    "02109",
    "02107",
    "02101",
    "02108",
    "02104",
    "02102",
    "02111",
    "02106",
    "02106",
    "02112",
    "02115",
    "02112",
    "02112",
    "02109",
]


def tabla_relaciones_EE():
    establecimientosEducativosComunes = "EEcomunes.csv"
    filtrados = []
    filtrados.append(establecimientos.iloc[4, [4, 9, 17, 18, 19]])
    for i in range(5, len(establecimientos)):
        if establecimientos.iloc[i, 13] == "1":
            filtrados.append(establecimientos.iloc[i, [4, 9, 17, 18, 19]])
    filtrados_df = pd.DataFrame(filtrados)
    columnas = {
        filtrados_df.columns[0]: "id_departamento",
        # filtrados_df.columns[1]: "nombre",
        filtrados_df.columns[1]: "domicilio",
        filtrados_df.columns[2]: "jardin_infantes",
        filtrados_df.columns[3]: "primario",
        filtrados_df.columns[4]: "secundario",
    }
    filtrados_df = filtrados_df.rename(columns=columnas)
    filtrados_df = filtrados_df[filtrados_df["secundario"] != "Secundario"]
    # filtrados_df.loc[
    #     filtrados_df["id_departamento"].isin(diccionario_caba.values()),
    #     "id_departamento",
    # ] = "02000"
    filtrados_df.to_csv(
        establecimientosEducativosComunes,
        encoding="utf-8",
        index=True,
        index_label="id_establecimiento",
    )


def tabla_relaciones_BP():
    bibliotecasPopulares = "BP.csv"
    filtrados = []
    for i in range(1, len(bibliotecas)):
        fila = bibliotecas.iloc[i]
        if isinstance(fila.iloc[15], str) and "@" in fila.iloc[15]:
            filtrados.append(
                (fila.iloc[2], fila.iloc[15].split("@")[1].split(".")[0], fila.iloc[22])
            )
        else:
            filtrados.append((fila.iloc[2], fila.iloc[15], fila.iloc[22]))

    filtrados_df = pd.DataFrame(filtrados)
    columnas = {
        filtrados_df.columns[0]: "nombre",
        filtrados_df.columns[1]: "mail",
        filtrados_df.columns[2]: "fecha_fundacion",
    }
    filtrados_df = filtrados_df.rename(columns=columnas)

    for index in range(len(codigos_postales)):
        filtrados_df.loc[649 + index, filtrados_df.columns[0]] = codigos_postales[index]

    filtrados_df.to_csv(
        bibliotecasPopulares, encoding="utf-8", index=True, index_label="id_biblioteca"
    )
    return filtrados_df


def obtener_provincia(codigo):
    localEstablecimiento = establecimientos.iloc[5:, 0:5].dropna(thresh=1)
    columnas = {
        localEstablecimiento.columns[0]: "jurisdiccion",
        localEstablecimiento.columns[1]: "sector",
        localEstablecimiento.columns[2]: "ambito",
        localEstablecimiento.columns[3]: "departamento",
        localEstablecimiento.columns[4]: "codigo_area",
    }
    localEstablecimiento = localEstablecimiento.rename(columns=columnas)
    coincidencias = localEstablecimiento[
        localEstablecimiento["codigo_area"] == str(codigo)
    ]

    if not coincidencias.empty:
        return coincidencias["jurisdiccion"].iloc[0]
    elif codigo == "94008":
        invalidos.append(codigo)
        return "Tierra del Fuego"
    elif codigo in diccionario_caba.keys():
        return "Ciudad de Buenos Aires"
    else:
        return "desconocida"


def tabla_departamentos():
    poblacion = pd.read_csv(basePath + "/poblacion.csv")
    departamentos = "departamentos.csv"
    col_dep = [
        "id_departamento",
        "nombre",
        "poblacion jardin",
        "poblacion primario",
        "poblacion secundario",
        "total poblacion",
    ]
    df_departamentos = pd.DataFrame(columns=col_dep)
    poblacion = poblacion.iloc[12:, 1:3].dropna(thresh=1)
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
    total_poblacion = 0
    while index < len(poblacion):
        if "#" in poblacion["edad"][index]:
            df_departamentos.loc[len(df_departamentos)] = [
                area,
                nombre,
                cuenta_jardin,
                cuenta_primario,
                cuenta_secundario,
                total_poblacion,
            ]
            area = poblacion["edad"][index].split(" ")[2]
            nombre = poblacion["casos"][index]
            cuenta_jardin = 0
            cuenta_primario = 0
            cuenta_secundario = 0
            total_poblacion = 0
        elif 3 <= int(poblacion["edad"][index]) <= 5:
            valor = str(poblacion["casos"][index]).replace(" ", "").strip()
            cuenta_jardin += int(valor)
            total_poblacion += int(valor)
        elif 5 <= int(poblacion["edad"][index]) <= 12:
            valor = str(poblacion["casos"][index]).replace(" ", "").strip()
            cuenta_primario += int(valor)
            total_poblacion += int(valor)
        elif 12 <= int(poblacion["edad"][index]) <= 18:
            valor = str(poblacion["casos"][index]).replace(" ", "").strip()
            total_poblacion += int(valor)
            cuenta_secundario += int(valor)
        else:
            valor = str(poblacion["casos"][index]).replace(" ", "").strip()
            total_poblacion += int(valor)
        index += 1
    df_departamentos["provincia"] = df_departamentos["id_departamento"].apply(
        obtener_provincia
    )
    df_departamentos["id_departamento"] = df_departamentos["id_departamento"].replace(
        diccionario_caba
    )
    # df_departamentos = unir_comunas(df_departamentos) Antes pensabamos que era una buena unir todo el departamento de caba, claramente mala idea
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


tabla_relaciones_EE()
tabla_relaciones_BP()
tabla_departamentos()

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


def analizarMail():
    df = bibliotecas["mail"]
    print("null mail", len(bibliotecas) - df.count())
    print("mails", df.count())


# hacer una tabla provincia departameento cantidad de jardines, poblacion de edad de jardin, primaria

# print(consultaSQL.head(8))
# print(consultaSQL)

# dataframeResultado = sql^ consultaSQL

# print(dataframeResultado)

# tabla_relaciones_EE()
# tabla_relaciones_BP()
# tabla_departamentos()

# Leer el archivo BP.csv
bp = pd.read_csv("BP.csv")

# Registrar el DataFrame como tabla en DuckDB
# Leer los archivos
ee = pd.read_csv("EEcomunes.csv")
deptos = pd.read_csv("departamentos.csv")

# # Conectar y registrar las tablas
# con = duckdb.connect()
# con.register("bp", bp)
# con.register("ee", ee)
# con.register("deptos", deptos)

consultai = duckdb.query(
    """
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
"""
).df()

consulta1 = "consulta1.csv"
consultai.to_csv(consulta1, encoding="utf-8", index=False)

consultaii = duckdb.query(
    """
SELECT
    deptos.provincia AS Provincia,
    deptos.nombre AS Departamento,
    COUNT(CASE WHEN bp.fecha_fundacion >= '1950-01-01' THEN 1 END) AS "Cantidad de BP fundadas desde 1950"
FROM deptos
LEFT OUTER JOIN bp ON deptos.id_departamento = bp.nombre
GROUP BY deptos.provincia, deptos.nombre
ORDER BY deptos.provincia ASC, "Cantidad de BP fundadas desde 1950" DESC
"""
).df()

consulta2 = "consulta2.csv"
consultaii.to_csv("consulta2.csv", encoding="utf-8", index=False)

consultaiii = duckdb.query(
    """
SELECT
    deptos.provincia AS Provincia,
    deptos.nombre AS Departamento,
    COUNT(DISTINCT ee.id_establecimiento) AS Cant_EE,
    COUNT(DISTINCT bp.id_biblioteca) AS Cant_BP,
    deptos."total poblacion" AS Población,
FROM deptos
LEFT OUTER JOIN ee ON deptos.id_departamento = ee.id_departamento
LEFT OUTER JOIN bp ON deptos.id_departamento = bp.nombre
GROUP BY deptos.provincia, deptos.nombre, deptos."poblacion jardin", deptos."poblacion primario", deptos."poblacion secundario", deptos."total poblacion"
ORDER BY Cant_EE DESC, Cant_BP DESC, Provincia ASC, Departamento ASC
"""
).df()

consulta3 = "consulta3.csv"
consultaiii.to_csv(consulta3, encoding="utf-8", index=False)

consultaiv = duckdb.query(
    """
SELECT
    Cantidad_por_dominio.Provincia AS Provincia,
    Cantidad_por_dominio.Departamento AS Departamento,
    Cantidad_por_dominio.Dominio AS Dominio
FROM (
    SELECT
        deptos.provincia AS Provincia,
        deptos.nombre AS Departamento,
        deptos.id_departamento AS ID_depto,
        bp.mail AS Dominio,
        COUNT(*) AS Cantidad
    FROM bp
    LEFT OUTER JOIN deptos ON deptos.id_departamento = bp.nombre
    WHERE bp.mail IS NOT NULL AND bp.mail <> ''
    GROUP BY deptos.provincia, deptos.nombre, deptos.id_departamento, bp.mail
) Cantidad_por_dominio
INNER JOIN (
    SELECT
        ID_depto,
        MAX(Cantidad) AS max_cantidad
    FROM (
        SELECT
            bp.nombre AS ID_depto,
            bp.mail AS Dominio,
            COUNT(*) AS Cantidad
        FROM bp
        WHERE bp.mail IS NOT NULL AND bp.mail <> ''
        GROUP BY bp.nombre, bp.mail
    ) Dominios
    GROUP BY ID_depto
) Dominio_maximo
ON Cantidad_por_dominio.ID_depto = Dominio_maximo.ID_depto AND Cantidad_por_dominio.Cantidad = Dominio_maximo.max_cantidad
ORDER BY Cantidad_por_dominio.Provincia ASC, Cantidad_por_dominio.Departamento ASC
"""
).df()
consulta4 = "consulta4.csv"
consultaiv.to_csv(consulta4, encoding="utf-8", index=False)


def grafico_I():
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

    bp_segun_provincia = sorted(
        bp_segun_provincia.items(), key=lambda x: x[1], reverse=True
    )
    # print(bp_segun_provincia)

    fig, ax = plt.subplots()
    ax.bar(
        data=bp_segun_provincia,
        x=[x[0] for x in bp_segun_provincia],
        height=[x[1] for x in bp_segun_provincia],
    )
    ax.set_xlabel("Provincias")
    ax.set_ylabel("Cantidad de Bibliotecas Populares")
    ax.set_title("Cantidad de Bibliotecas Populares por Provincia")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# Ejercicio 2
# Vamos a hacer tres graficos de puntos por cada Nivel educativo, cada burbuja va a representar un departamento, el eje x va a ser la poblacion y el eje y la cantidad de establecimientos educativos.


def grafico_II(df):
    plt.scatter(
        df["Población Jardin"], df["Jardines"], color="green", alpha=0.7, label="Jardín"
    )
    plt.scatter(
        df["Población Primaria"],
        df["Primarias"],
        color="blue",
        alpha=0.7,
        label="Primaria",
    )
    plt.scatter(
        df["Población Secundaria"],
        df["Secundarios"],
        color="red",
        alpha=0.7,
        label="Secundaria",
    )
    plt.xlabel(f"Población")
    plt.ylabel(f"Cantidad de EE")
    plt.title(f"Cantidad de EE vs Población")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))  # Leyenda a la derecha
    plt.tight_layout()
    plt.show()


# graficar_puntos_nivel(pd.read_csv("consulta1.csv"))


# def encontrarCantEE(df):
#     cuenta = 0
#     for i in range(len(df)):
#         cuenta += int(df.loc[i,"Total_EE"])
#     return cuenta


def grafico_III():
    df = pd.read_csv("consulta1.csv")
    # nuevoDf = df.drop_duplicates(subset =["Provincia"]).reset_index(drop=True)
    df["Total_EE"] = df["Primarias"] + df["Secundarios"] + df["Jardines"]

    # for i in range(len(nuevoDf)):
    #     nuevoDf.loc[i, "Total_EE"] = encontrarCantEE(df[df["Provincia"] == nuevoDf.loc[i, "Provincia"]].reset_index(drop=True))
    # print(nuevoDf)
    sns.boxplot(
        data=df,
        x="Provincia",
        y="Total_EE",
    )
    plt.title("Distribución de EE por provincia")
    plt.xticks(rotation=90)
    plt.show()


def grafico_IV():
    df = pd.read_csv("consulta3.csv")
    df["BP_por_mil"] = (df["Cant_BP"] / df["Población"]) * 1000
    df["EE_por_mil"] = (df["Cant_EE"] / df["Población"]) * 1000

    sns.scatterplot(
        data=df,
        x="BP_por_mil",
        y="EE_por_mil",
        hue="Provincia",  # Opcional: colorear por provincia
    )

    # Etiquetas y título
    plt.title("Relación entre BP y EE por cada mil habitantes (por departamento)")
    plt.xlabel("BP por mil habitantes")
    plt.ylabel("EE por mil habitantes")
    plt.show()


grafico_I()
grafico_II(pd.read_csv("consulta1.csv"))
grafico_III()
grafico_IV()
