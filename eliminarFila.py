#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import glob

def crear_carpeta(ruta):
    """
    Crea una carpeta si no existe.
    """
    os.makedirs(ruta, exist_ok=True)

def eliminar_label_de_csv(entrada, salida):
    """
    Elimina la columna 'Label' de un archivo CSV.
    """
    try:
        with open(entrada, "r", newline="", encoding="utf-8") as f_in, \
             open(salida, "w", newline="", encoding="utf-8") as f_out:
            
            lector = csv.reader(f_in)
            escritor = csv.writer(f_out)
            
            # 1. Leer la primera fila (cabecera).
            cabecera = next(lector)
            
            # 2. Verificar si 'Label' está en la cabecera.
            if "Label" not in cabecera:
                print(f"Advertencia: La columna 'Label' no existe en {entrada}. Archivo omitido.")
                return
            
            # 3. Obtener el índice de la columna 'Label'.
            indice_label = cabecera.index("Label")
            
            # 4. Eliminar la columna 'Label' de la cabecera.
            cabecera.pop(indice_label)
            escritor.writerow(cabecera)
            
            # 5. Eliminar la columna 'Label' de cada fila y escribir la nueva fila.
            for fila in lector:
                if len(fila) <= indice_label:
                    print(f"Advertencia: La fila tiene menos columnas de las esperadas en {entrada}.")
                    continue
                fila.pop(indice_label)
                escritor.writerow(fila)
                
        print(f"Archivo procesado y guardado sin 'Label': {salida}")
        
    except Exception as e:
        print(f"Error al procesar el archivo {entrada}: {e}")

def main():
    # Directorio de entrada donde están los archivos CSV con 'Label'
    directorio_entrada = r"C:\Users\Jm\Desktop\MASTER MII\SEGUNDO\E-HEALTH\PROYECTO-eHEALTH\datos"
    
    # Directorio de salida donde se guardarán los CSV sin 'Label'
    directorio_salida = os.path.join(os.getcwd(), 'datos_sin_label')
    crear_carpeta(directorio_salida)
    
    # Patrón de búsqueda para archivos CSV
    patron_busqueda = os.path.join(directorio_entrada, '*.csv')
    archivos_csv = glob.glob(patron_busqueda)
    
    if not archivos_csv:
        print(f"No se encontraron archivos CSV en la carpeta: {directorio_entrada}")
        return
    
    print(f"Encontrados {len(archivos_csv)} archivo(s) para procesar.\n")
    
    # Procesar cada archivo CSV
    for archivo in archivos_csv:
        nombre_archivo = os.path.basename(archivo)
        ruta_salida = os.path.join(directorio_salida, nombre_archivo)
        
        print(f"Procesando archivo: {nombre_archivo}")
        eliminar_label_de_csv(archivo, ruta_salida)
    
    print("\nProceso completado para todos los archivos.")

if __name__ == "__main__":
    main()
