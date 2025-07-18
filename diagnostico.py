import pandas as pd
from pathlib import Path
import numpy as np
import csv
import sys
import warnings
from collections import defaultdict

csv.field_size_limit(100000000)

# Lista de directorios donde buscar las diferentes versiones de los datasets
# Añadir aquí nuevas versiones que vayas generando
DIRECTORIOS_COMPARAR = [
    Path("dataset/iniciales"),               # Original
    Path("dataset/preprocesamiento_general"),  # Después de normalización de etiquetas
    Path("dataset/solo_ingles"),  # Después de dejar solo los correos en inglés
    # Añade aquí más rutas
]

# Lista de archivos a comparar (deben estar en todos los directorios)
ARCHIVOS_CSV = [
    "CEAS_08.csv", "Enron.csv", "Ling.csv", "SpamAssasin.csv",
    "Nazario.csv", "Nazario_5.csv", "Nigerian_5.csv", "Nigerian_Fraud.csv",
    "TREC_05.csv", "TREC_06.csv", "TREC_07.csv"
]

# ======== CLASE PRINCIPAL PARA EL ANÁLISIS ========
class AnalizadorDataset:
    """
    Clase que analiza múltiples versiones de datasets y almacena métricas comparativas.
    """
    def __init__(self):
        # Estructura: {nombre_archivo: {version: métricas}}
        self.registros = defaultdict(dict)
        # Estructura para totales: {version: métricas_totales}
        self.totales = defaultdict(dict)
    
    def analizar_archivo(self, ruta_archivo, version):
        """
        Procesa un archivo CSV y calcula todas las métricas requeridas.
        """
        try:
            # Leer el archivo con tolerancia a errores
            df = pd.read_csv(ruta_archivo, engine='python', on_bad_lines='warn')
            
            # Detectar nombres reales de columnas (case-insensitive)
            subject_col = next((col for col in df.columns if col.lower() == 'subject'), None)
            body_col = next((col for col in df.columns if col.lower() == 'body'), None)
            
            # Calcular todas las métricas
            stats = {
                'total_registros': len(df),
                'duplicados_totales': self._calcular_duplicados(df, keep='first'),
                'duplicados_subject_body': self._calcular_duplicados(df, subset=[subject_col, body_col]) 
                                          if subject_col and body_col else 'N/A',
                'duplicados_body': self._calcular_duplicados(df, subset=[body_col]) 
                                 if body_col else 'N/A',
                'campos_faltantes': self._calcular_faltantes(df),
                'faltantes_subject_body': self._calcular_faltantes_combinados(df),
                'distribucion_labels': self._analizar_labels(df)
            }
            
            # Almacenar resultados
            self.registros[ruta_archivo.name][version] = stats
            
        except Exception as e:
            print(f"Error analizando {ruta_archivo.name} ({version}): {str(e)}")
    
    def calcular_totales(self):
        """
        Calcula los totales sumando las métricas de todos los datasets por versión.
        """
        for archivo, versiones in self.registros.items():
            for version, stats in versiones.items():
                # Normalizar nombre de versión
                version_normalizada = self._normalizar_nombre_version(version)
                
                if version_normalizada not in self.totales:
                    self.totales[version_normalizada] = {
                        'total_registros': 0,
                        'duplicados_totales': 0,
                        'duplicados_subject_body': 0,
                        'duplicados_body': 0,
                        'total_campos_faltantes': 0,
                        'archivos_procesados': 0,
                        'distribucion_labels_consolidada': defaultdict(int)
                    }
                
                # Sumar métricas numéricas
                self.totales[version_normalizada]['total_registros'] += stats['total_registros']
                self.totales[version_normalizada]['duplicados_totales'] += stats['duplicados_totales']
                
                # Manejar valores que pueden ser 'N/A'
                if stats['duplicados_subject_body'] != 'N/A':
                    self.totales[version_normalizada]['duplicados_subject_body'] += stats['duplicados_subject_body']
                
                if stats['duplicados_body'] != 'N/A':
                    self.totales[version_normalizada]['duplicados_body'] += stats['duplicados_body']
                
                # Sumar campos faltantes totales
                total_faltantes = sum(stats['campos_faltantes'].values())
                self.totales[version_normalizada]['total_campos_faltantes'] += total_faltantes
                
                self.totales[version_normalizada]['archivos_procesados'] += 1
                
                # Consolidar distribución de labels
                if stats['distribucion_labels']:
                    # Convertir porcentajes a conteos absolutos para sumar correctamente
                    for label, porcentaje in stats['distribucion_labels'].items():
                        conteo_absoluto = int((porcentaje / 100) * stats['total_registros'])
                        self.totales[version_normalizada]['distribucion_labels_consolidada'][label] += conteo_absoluto
        
        # Convertir conteos consolidados de vuelta a porcentajes
        for version in self.totales:
            total_registros = self.totales[version]['total_registros']
            if total_registros > 0:
                distribucion_final = {}
                for label, conteo in self.totales[version]['distribucion_labels_consolidada'].items():
                    porcentaje = (conteo / total_registros) * 100
                    distribucion_final[label] = round(porcentaje, 2)
                self.totales[version]['distribucion_labels_final'] = distribucion_final
            else:
                self.totales[version]['distribucion_labels_final'] = {}
    
    def _normalizar_nombre_version(self, version):
        """
        Normaliza nombres de versiones para mantener consistencia.
        """
        if 'normalizacion_etiquetas' in version or 'etiquetas_normalizadas' in version:
            return 'preprocesamiento_general'
        return version
    
    def _calcular_duplicados(self, df, subset=None, keep='first'):
        """Calcula duplicados según el subconjunto de columnas especificado."""
        if subset is None:  # Duplicados completos
            return df.duplicated(keep=keep).sum()
        try:
            return df.duplicated(subset=subset, keep=keep).sum()
        except KeyError:  # Si alguna columna no existe
            return 0
    
    def _calcular_faltantes(self, df):
        """Calcula valores faltantes por columna."""
        return {col: df[col].isna().sum() for col in df.columns}
    
    def _calcular_faltantes_combinados(self, df):
        """Calcula registros donde ambos campos (subject y body) están vacíos."""
        subject_col = next((col for col in df.columns if col.lower() == 'subject'), None)
        body_col = next((col for col in df.columns if col.lower() == 'body'), None)
        if subject_col and body_col:
            return df[[subject_col, body_col]].isna().all(axis=1).sum()
        return 0
    
    def _analizar_labels(self, df):
        """Calcula distribución porcentual de etiquetas."""
        if 'label' in df.columns:
            return df['label'].value_counts(normalize=True).mul(100).round(2).to_dict()
        return None

def generar_reporte_comparativo(analizador):
    """
    Genera un reporte HTML con comparativas detalladas entre versiones.
    """
    html = """
    <html>
    <head>
        <title>Reporte Comparativo de Procesamiento</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .dataset { border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; }
            .resumen { border: 2px solid #007acc; padding: 20px; margin-bottom: 30px; background-color: #f8f9fa; }
            table { width: 100%; border-collapse: collapse; margin: 10px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            .resaltado { background-color: #f0f8ff; }
            .totales { background-color: #e8f5e8; }
            .titulo-resumen { color: #007acc; margin-bottom: 15px; }
        </style>
    </head>
    <body>
        <h1>Reporte Comparativo de Procesamiento de Datasets</h1>
    """
    
    # Generar sección de resumen total
    html += generar_seccion_resumen(analizador)
    
    # Generar sección para cada archivo
    for archivo in ARCHIVOS_CSV:
        datos = analizador.registros.get(archivo, {})
        if not datos:
            continue
            
        html += f"""
        <div class='dataset'>
            <h2>📁 {archivo}</h2>
            <h3>📊 Métricas Principales</h3>
            {generar_tabla_metricas(datos)}
            <h3>🔍 Evolución de Campos Faltantes</h3>
            {generar_tabla_faltantes(datos)}
        </div>
        """
    
    html += "</body></html>"
    return html

def generar_seccion_resumen(analizador):
    """
    Genera la sección de resumen con totales de todos los datasets.
    """
    html = """
    <div class='resumen'>
        <h2 class='titulo-resumen'>📈 RESUMEN TOTAL - Comparativa General</h2>
        <p><strong>Resumen consolidado de todos los datasets procesados</strong></p>
        {tabla_totales}
    </div>
    """.format(tabla_totales=generar_tabla_totales(analizador.totales))
    
    return html

def generar_tabla_totales(totales):
    """
    Genera tabla HTML con los totales consolidados por versión.
    """
    html = """
    <table class='totales'>
        <tr>
            <th>Versión</th>
            <th>Archivos Procesados</th>
            <th>Total Registros</th>
            <th>Distribución Labels</th>
            <th>Total Duplicados</th>
            <th>Duplicados (Subject+Body)</th>
            <th>Duplicados (Body)</th>
            <th>Total Campos Faltantes</th>
        </tr>
    """
    
    for version, stats in totales.items():
        html += f"""
        <tr>
            <td><strong>{version}</strong></td>
            <td>{stats['archivos_procesados']}</td>
            <td><strong>{stats['total_registros']:,}</strong></td>
            <td>{formatear_labels(stats.get('distribucion_labels_final', {}))}</td>
            <td>{stats['duplicados_totales']:,}</td>
            <td>{stats['duplicados_subject_body']:,}</td>
            <td>{stats['duplicados_body']:,}</td>
            <td>{stats['total_campos_faltantes']:,}</td>
        </tr>
        """
    
    html += "</table>"
    return html

def generar_tabla_metricas(datos):
    """
    Genera tabla HTML con las métricas principales excluyendo Faltantes (Subject+Body).
    """
    html = """
    <table class='resaltado'>
        <tr>
            <th>Versión</th>
            <th>Registros</th>
            <th>Duplicados Totales</th>
            <th>Duplicados (Subject+Body)</th>
            <th>Duplicados (Body)</th>
            <th>Distribución Labels</th>
        </tr>
    """
    
    for version, stats in datos.items():
        html += f"""
        <tr>
            <td>{version}</td>
            <td>{stats['total_registros']}</td>
            <td>{stats['duplicados_totales']}</td>
            <td>{stats['duplicados_subject_body']}</td>
            <td>{stats['duplicados_body']}</td>
            <td>{formatear_labels(stats['distribucion_labels'])}</td>
        </tr>
        """
    
    html += "</table>"
    return html

def generar_tabla_faltantes(datos):
    """
    Genera tabla HTML con la evolución de campos faltantes incluyendo Subject+Body.
    """
    html = "<table><tr><th>Campo</th>"
    
    # Encabezados de versiones
    for version in datos.keys():
        html += f"<th>{version}</th>"
    html += "</tr>"
    
    # Campos comunes + Subject+Body
    campos = set()
    for stats in datos.values():
        campos.update(stats['campos_faltantes'].keys())
    campos.add('subject+body')  # Añadir campo especial
    
    # Filas de datos
    for campo in campos:
        html += f"<tr><td>{campo}</td>"
        for version, stats in datos.items():
            if campo == 'subject+body':
                count = stats['faltantes_subject_body']
            else:
                count = stats['campos_faltantes'].get(campo, 0)
            html += f"<td>{count}</td>"
        html += "</tr>"
    
    html += "</table>"
    return html

def formatear_labels(distribucion):
    """Formatea la distribución de labels para visualización."""
    if not distribucion:
        return "N/A"
    return ", ".join([f"{k}: {v}%" for k, v in distribucion.items()])

if __name__ == "__main__":
    analizador = AnalizadorDataset()
    
    # Procesar todas las versiones
    for directorio in DIRECTORIOS_COMPARAR:
        for archivo in ARCHIVOS_CSV:
            ruta = directorio / archivo
            if ruta.exists():
                analizador.analizar_archivo(ruta, directorio.name)
    
    # Calcular totales consolidados
    analizador.calcular_totales()
    
    # Generar y guardar reporte
    with open("diagnostico/reporte_comparativo.html", "w", encoding="utf-8") as f:
        f.write(generar_reporte_comparativo(analizador))
    
    print("✅ Reporte generado: reporte_comparativo.html")
    print("\n📊 Resumen de totales por versión:")
    for version, stats in analizador.totales.items():
        distribucion = formatear_labels(stats.get('distribucion_labels_final', {}))
        print(f"  {version}: {stats['total_registros']:,} registros totales, {stats['archivos_procesados']} archivos")
        print(f"    Distribución: {distribucion}")
