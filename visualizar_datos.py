import pandas as pd
from pathlib import Path

# Configuración de directorios
DIRECTORIO_DATASETS = Path("dataset/solo_ingles")
ARCHIVOS_CSV = [
    "CEAS_08.csv", "Enron.csv", "Ling.csv", "SpamAssasin.csv",
    "Nazario.csv", "Nazario_5.csv", "Nigerian_5.csv", "Nigerian_Fraud.csv",
    "TREC_05.csv", "TREC_06.csv", "TREC_07.csv", "grupo_combinado_1.csv", "grupo_combinado_2.csv"
]

def mostrar_ejemplos(dataset, n=3, max_chars=50):
    """
    Muestra ejemplos legibles de un dataset con texto truncado,
    y el porcentaje de distribución de las etiquetas.
    """
    try:
        # Leer el archivo
        df = pd.read_csv(DIRECTORIO_DATASETS / dataset)
        
        # 1. Mostrar distribución de etiquetas si existe la columna
        if 'label' in df.columns:
            total = len(df)
            distribucion = df['label'].value_counts(normalize=True).mul(100).round(2)
            
            print(f"\n📊 {dataset}: Distribución de etiquetas")
            print(f"  - Total registros: {total}")
            for etiqueta, porcentaje in distribucion.items():
                conteo = df['label'].value_counts()[etiqueta]
                print(f"  - Label {etiqueta}: {porcentaje}% ({conteo} registros)")
        else:
            print(f"\n⚠️ {dataset}: No tiene columna 'label'")
        
        # 2. Mostrar muestra de registros
        orden_columnas = []
        if 'texto_completo' in df.columns:
            orden_columnas.append('texto_completo')
        if 'texto_preprocesado' in df.columns:
            orden_columnas.append('texto_preprocesado')
        if 'label' in df.columns:
            orden_columnas.append('label')
        
        if not orden_columnas:
            print(f"⚠️ {dataset}: No se encontraron columnas de texto relevantes")
            return
        
        # Muestrear y formatear
        muestra = df[orden_columnas].sample(n=n, random_state=42)
        
        # Truncar textos largos
        for col in ['texto_completo', 'texto_preprocesado']:
            if col in muestra.columns:
                muestra[col] = muestra[col].astype(str).str[:max_chars] + '...'
        
        print(f"\n📁 Muestra de {n} registros:")
        print(muestra.to_string(index=False, justify='left'))
        
    except Exception as e:
        print(f"\n❌ Error procesando {dataset}: {str(e)}")

# Configurar pandas para mejor visualización
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)

# Procesar todos los datasets
print("=== Análisis de datasets preprocesados ===")
for dataset in ARCHIVOS_CSV:
    mostrar_ejemplos(dataset)

print("\n=== Análisis completado ===")
