import pandas as pd
from pathlib import Path


# Directorio donde están los CSV preprocesados de ML
DIRECTORIO = Path("dataset/preprocesados_ML")


# Especificar el orden y los campos requeridos para el grupo 1
CAMPOS_GRUPO_1 = [
    'body',            # cuerpo del email
    'date',            # fecha
    'label',           # etiqueta
    'receiver',        # receptor
    'sender',          # emisor
    'subject',         # asunto
    'texto_completo',  # texto completo original
    'texto_preprocesado', # texto tras preprocesado
    'urls'             # existencia de URLs
]

# Campos requeridos para el grupo 2
CAMPOS_GRUPO_2 = [
    'body',
    'label',
    'subject',
    'texto_completo',
    'texto_preprocesado'
]

# Listas de archivos para cada grupo
archivos_grupo_1 = [
    "CEAS_08.csv", "SpamAssasin.csv", "Nazario.csv",
    "Nazario_5.csv", "Nigerian_5.csv", "Nigerian_Fraud.csv",
    "TREC_05.csv", "TREC_06.csv", "TREC_07.csv"
]

archivos_grupo_2 = [
    "CEAS_08.csv", "Enron.csv", "Ling.csv", "SpamAssasin.csv",
    "Nazario.csv", "Nazario_5.csv", "Nigerian_5.csv", 
    "Nigerian_Fraud.csv", "TREC_05.csv", "TREC_06.csv", "TREC_07.csv"
]


def combinar_y_guardar(archivos, campos, nombre_salida):
    """
    Combina los CSV indicados en 'archivos' según el esquema de 'campos',
    y guarda el resultado en un archivo llamado 'nombre_salida'.
    """
    dfs = []  # lista para acumular DataFrames procesados

    for archivo in archivos:
        ruta = DIRECTORIO / archivo
        try:
            # Leer el CSV completo como texto para evitar warnings de tipo mixto
            df = pd.read_csv(
                ruta,
                dtype=str,        # forzar todas las columnas como string
                low_memory=False  # desactiva procesamiento en trozos
            )

            # Reindexar: ajusta las columnas a la lista 'campos'
            # - Si faltan columnas, se crean con NaN.
            # - Si sobran columnas, se descartan.
            df = df.reindex(columns=campos)

            dfs.append(df)
            print(
                f"✅ {archivo} procesado "
                f"(registros: {len(df)}, columnas: {list(df.columns)})"
            )
        except Exception as e:
            # Captura cualquier error de lectura o procesamiento
            print(f"❌ Error con {archivo}: {e}")

    if dfs:
        # Concatenar todos los DataFrames homogéneos
        combinado = pd.concat(dfs, ignore_index=True)

        # Guardar el CSV combinado
        salida = DIRECTORIO / nombre_salida
        combinado.to_csv(salida, index=False)
        print(
            f"\nArchivo combinado guardado: {salida} "
            f"(total registros: {len(combinado)})\n"
        )
    else:
        print("⚠️ No se han podido combinar archivos. Ningún DataFrame válido.")

# Ejecución: combinar ambos grupos
combinar_y_guardar(
    archivos_grupo_1,
    CAMPOS_GRUPO_1,
    "grupo_combinado_1.csv"
)
combinar_y_guardar(
    archivos_grupo_2,
    CAMPOS_GRUPO_2,
    "grupo_combinado_2.csv"
)
