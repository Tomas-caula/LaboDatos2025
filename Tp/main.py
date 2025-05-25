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

# Este diccionario fue creado para vincular codigos de areas equivalentes.
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

# la variable codigo_postales reprensenta en orden los codigos de area de todas aquellas bibliotecas ubicadas en capital feder
# en los datos crudos no hay separecion por comuna.
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
    filtrados.append(establecimientos.iloc[4, [4, 8, 17, 18, 19]])
    for i in range(5, len(establecimientos)):
        if establecimientos.iloc[i, 13] == "1":
            filtrados.append(establecimientos.iloc[i, [4, 8, 17, 18, 19]])
    filtrados_df = pd.DataFrame(filtrados)

    # creamos un dataFrame limpio sacando todas las cosas filas, columnas o textos inecesarios
    columnas = {
        filtrados_df.columns[0]: "id_departamento",
        filtrados_df.columns[1]: "nombre",
        filtrados_df.columns[2]: "jardin_infantes",
        filtrados_df.columns[3]: "primario",
        filtrados_df.columns[4]: "secundario",
    }

    filtrados_df = filtrados_df.rename(columns=columnas)

    # Renombramos las columnas correctamente

    filtrados_df = filtrados_df[filtrados_df["secundario"] != "Secundario"]

    # eliminamos aquella fila donde el secundario sea Secundario para ahora si tenerla lo mas limpia posible
    filtrados_df = filtrados_df.drop_duplicates(subset="nombre")
    filtrados_df = filtrados_df.drop(columns="nombre")

    # evitamos que en la columna de nombre haya repetidos. Esto lo hacemos porque nos dimos cuenta que cada entrada de colegio esta anotada implicando asi
    # que colegios como el San Cayetano, Instituto San Roman, Ort, etc esten anotados multiples veces porque tiene mas de una puerta

    filtrados_df.to_csv(
        establecimientosEducativosComunes,
        encoding="utf-8",
        index=True,
        index_label="id_establecimiento",
    )

    # guardamos todo dentro de un archivo csv con el nombre EEcomunes


def tabla_relaciones_BP():
    # Eliminamos todo lo no necesario de los datos en crudo y renombramos las columnas para que este todo mas limpio y ordenado ->
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

    # <- Eliminamos todo lo no necesario de los datos en crudo y renombramos las columnas para que este todo mas limpio y ordenado

    # Encontramos que las bibiliotecas populares tienen asignado el codigo de capital federal como si fuesen todos los mismo, sin separar por comuna ->
    for index in range(len(codigos_postales)):
        filtrados_df.loc[649 + index, filtrados_df.columns[0]] = codigos_postales[index]

    # <- Encontramos que las bibiliotecas populares tienen asignado el codigo de capital federal como si fuesen todos los mismo, sin separar por comuna

    filtrados_df.to_csv(
        bibliotecasPopulares, encoding="utf-8", index=True, index_label="id_biblioteca"
    )
    # Lo guardamos en el csv


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

tabla_relaciones_EE()
tabla_relaciones_BP()
tabla_departamentos()



def analizarMail():
    df = bibliotecas["mail"]
    print("null mail", len(bibliotecas) - df.count())
    print("mails", df.count())
    # Contamos cuales son todos los mailes vacios, df.count discrimina los que estan vacios.


def analizarBibliotecas():
    df = pd.read_csv("BP.csv")
    df = df[df["fecha_fundacion"].isna()]
    # print("Bibliotecas sin fecha de fundacion", df)
    print("cantidad de biblitecas sin fecha: ", len(df))

    # Analizamos todas las biblitecas con la fecha de fundacion no existente


def analizarEstablecimientos():
    df = pd.read_csv("EEcomunes.csv")
    # Agarramos los Establecimientos ya filtrados por la funcion tabla_relaciones_EE()
    df = df[
        df["jardin_infantes"].isna() & df["primario"].isna() & df["secundario"].isna()
    ]

    print("Escuelas sin jardin, primario ni secundario", len(df))
    print("Cantidad de instituciones sin nivel:", len(df))

    # Anilizamos todos los establecimientos educativos que no tienen ni primario ni secondario ni primario.


analizarEstablecimientos()
analizarBibliotecas()


# hacer una tabla provincia departameento cantidad de jardines, poblacion de edad de jardin, primaria
# Registrar el DataFrame como tabla en DuckDB
# Leer los archivos
ee = pd.read_csv("EEcomunes.csv")
deptos = pd.read_csv("departamentos.csv")
bp = pd.read_csv("BP.csv")
#Consulta 1
# Unimos las tablas de Departamentos y Establecimientos Educativos según el id_departamento
# Las reagrupamos por departamento y provincia para que luego en la seleccion de los datos podamos contar la cantidad de jardines, primarias y secundarias
# Seleccionamos las tablas necesarias para la consulta
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
# Consulta 2
# Unimos las tablas de Departamentos y Bibliotecas Populares según el id_departamento
# Las reagrupamos por departamento y provincia para que luego en la seleccion de los datos podamos contar la cantidad de bibliotecas fundadas desde 1950 por departamento
# Seleccionamos las tablas necesarias para la consulta
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

# Consulta 3
# Unimos las tablas de Departamentos, Establecimientos Educativos y Bibliotecas Populares según el id_departamento
# Las reagrupamos por departamento y provincia para que luego en la seleccion de los datos podamos contar la cantidad de bibliotecas y establecimientos educativos por departamento
# Seleccionamos las tablas necesarias para la consulta
# Ordenamos por cantidad por cantidad EE descendente, cantidad BP descendente, nombre de provincia ascendente y nombre de departamento ascendente
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

# Consulta 4
# Creamos una tabla que nos permita ver los cantidad de veces que aparece cada dominio por departamento
# Creamos una tabla que indica las veces que aparece el dominio mas frecuente por departamento a partir de crear de vuelta la tabla anterior
# Unimos ambas tablas por la cantidad de veces que aparece el domino y seleccionamos las columnas necesarias
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
    # Leemos la tabla de departamentos creada y filtrado por la funcion tablaDepartamentos() previamente y establecemos la columna id_departamento como strin
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

def grafico_III():
    df = pd.read_csv("consulta1.csv")
    df["Total_EE"] = df["Primarias"] + df["Secundarios"] + df["Jardines"]
    orden_provincias = df.groupby("Provincia")["Total_EE"].median().sort_values().index
    # Establecemos un oreden de las provincias por el valor medio de los Totales

    sns.boxplot(data=df, x="Provincia", y="Total_EE", order=orden_provincias)
    # Creamos el boxplot por provincia y en bace al Total de instituciones ordenado
    plt.title("Distribución de EE por provincia")
    # Le ponemops un titulo al grafico
    plt.xticks(rotation=90)
    # Rotamos el nombre de las provincias para que no se vea superapuesto.
    plt.show()


def grafico_IV():
    df = pd.read_csv("consulta3.csv")
    df["BP_por_mil"] = (df["Cant_BP"] / df["Población"]) * 1000
    df["EE_por_mil"] = (df["Cant_EE"] / df["Población"]) * 1000
    # Aarramos y creamos las columnas de BP_por_mil y EE_por_mil teniendo por cada mil habitantes
    sns.scatterplot(
        data=df,
        x="BP_por_mil",
        y="EE_por_mil",
        hue="Provincia",
    )

    # Colocamos los puntos en un scatterplot teniendo en el eje x las Bibliotecas polulas y los Establecimientos en el eje Y. Separamos por provincias tambien

    # Etiquetas y título
    plt.title("Relación entre BP y EE por cada mil habitantes (por departamento)")
    plt.xlabel("BP por mil habitantes")
    plt.ylabel("EE por mil habitantes")
    # Le ponemos un titulo al grafico y el nombre de los ejes
    plt.show()

# Mostramos todos los graficos ->
grafico_I()
grafico_II(pd.read_csv("consulta1.csv"))
grafico_III()
grafico_IV()
# <- Mostramos todos los graficos