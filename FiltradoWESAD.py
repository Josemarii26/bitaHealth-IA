
import os
import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import Counter

def crear_carpeta(ruta):
    os.makedirs(ruta, exist_ok=True)

def procesar_archivo(ruta_txt, ruta_pkl, salida_csv, ventana_segundos=30, sampling_rate=700):

    try:
        datos_raw = pd.read_csv(
            ruta_txt,
            sep='\t',
            comment='#',
            header=0,
            usecols=range(10)
        )
    except Exception as e:
        print(f'Error al cargar el archivo TXT ({ruta_txt}): {e}')
        return

    datos_raw.columns = [
        'nSeq', 'DI', 'CH1_ECG', 'CH2_EDA', 'CH3_EMG', 'CH4_TEMP',
        'CH5_ACC_X', 'CH6_ACC_Y', 'CH7_ACC_Z', 'CH8_RESPIRATION'
    ]

    vcc = 3
    chan_bit = 2**16  

    datos_raw['ECG_mV'] = ((datos_raw['CH1_ECG'] / chan_bit - 0.5) * vcc)
    datos_raw['EDA_uS'] = (((datos_raw['CH2_EDA'] / chan_bit) * vcc) / 0.12)

    try:
        with open(ruta_pkl, 'rb') as file:
            datos_pkl = pickle.load(file, encoding='latin1')
    except Exception as e:
        print(f'Error al cargar el archivo PKL ({ruta_pkl}): {e}')
        return

    etiquetas = datos_pkl.get('label', [])
    
    if etiquetas is None or len(etiquetas) == 0:
        print(f'Advertencia: No se encontraron etiquetas en el archivo PKL ({ruta_pkl}).')
        return

    mapa_etiquetas = {
        0: 'No Definido/Transitorio',
        1: 'Baseline',
        2: 'Estrés',
        3: 'Diversión',
        4: 'Meditación'
    }

    etiquetas_mapeadas = [mapa_etiquetas.get(etiqueta, 'Ignorar') for etiqueta in etiquetas]

    contador_etiquetas_filtradas = Counter(etiquetas_mapeadas)
    print(f'\nProcesando archivos:\nTXT: {ruta_txt}\nPKL: {ruta_pkl}')
    print(f'Conteo de etiquetas filtradas: {contador_etiquetas_filtradas}')

    indices_validos = [i for i, etiqueta in enumerate(etiquetas_mapeadas) if etiqueta != 'Ignorar']
    etiquetas_filtradas = [etiquetas_mapeadas[i] for i in indices_validos]
    datos_filtrados = datos_raw.iloc[indices_validos].reset_index(drop=True)

    print(f'Total de muestras válidas: {len(etiquetas_filtradas)}')
    print(f'Número de muestras en señales filtradas: {len(datos_filtrados)}')

    ventana_muestras = ventana_segundos * sampling_rate  

    total_ventanas = len(datos_filtrados) // ventana_muestras
    print(f'Total de ventanas posibles: {total_ventanas}')

    lista_features = []

    for i in range(total_ventanas):
        inicio = i * ventana_muestras
        fin = inicio + ventana_muestras

        if fin > len(datos_filtrados):
            print(f'Ventana {i} excede el número de muestras disponibles.')
            break

        ventana = datos_filtrados.iloc[inicio:fin]
        ecg_signal = ventana['ECG_mV'].values

        etiqueta_ventana = max(set(etiquetas_filtradas[inicio:fin]), key=etiquetas_filtradas[inicio:fin].count)

        features = {}
        features['Label'] = etiqueta_ventana
        features['Ventana_Inicio'] = inicio
        features['Ventana_Fin'] = fin

        print(f'\nProcesando Ventana {i}: {etiqueta_ventana}')
        print(f'Número de muestras en la ventana: {len(ventana)}')

        # Procesamiento de ECG
        try:
            # Limpiar la señal ECG
            ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method="neurokit")

            # Detectar picos R
            signals_ecg, info_ecg = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
            print(f'ECG Processed para ventana {i}')

            hrv_time = nk.hrv_time(signals_ecg, sampling_rate=sampling_rate, show=False)
            hrv_freq = nk.hrv_frequency(signals_ecg, sampling_rate=sampling_rate, show=False)
            hrv_features = {**hrv_time, **hrv_freq}

            for key, value in hrv_features.items():
                if isinstance(value, pd.Series):
                    scalar_value = value.iloc[0]
                    if isinstance(scalar_value, (int, float, np.number)):
                        features[key] = scalar_value
                    else:
                        print(f'Advertencia: La característica "{key}" no es escalar. Valor asignado: {scalar_value}')
                        features[key] = np.nan
                elif isinstance(value, (int, float, np.number)):
                    features[key] = value
                else:
                    print(f'Advertencia: La característica "{key}" no es escalar. Valor asignado: {value}')
                    features[key] = np.nan

            # Obtener picos R
            r_peaks = info_ecg.get("ECG_R_Peaks", [])
            features['ECG_R_Peaks_Count'] = len(r_peaks)

        except Exception as e:
            print(f'Error al procesar ECG en ventana {i}: {e}')
            # Asignar NaN a las características HRV y ECG relacionadas
            for key in [
                'HRV_MeanNN', 'HRV_SDNN', 'HRV_SDANN1', 'HRV_SDNNI1', 'HRV_SDANN2',
                'HRV_SDNNI2', 'HRV_SDANN5', 'HRV_SDNNI5', 'HRV_RMSSD',
                'HRV_SDSD', 'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN',
                'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN', 'HRV_SDRMSSD',
                'HRV_Prc20NN', 'HRV_Prc80NN', 'HRV_pNN50', 'HRV_pNN20',
                'HRV_MinNN', 'HRV_MaxNN', 'HRV_HTI', 'HRV_TINN',
                'HRV_HFn', 'HRV_LnHF', 'ECG_R_Peaks_Count'
            ]:
                features[key] = np.nan
            lista_features.append(features)
            continue  

        
        lista_features.append(features)

    # Crear un DataFrame con las características extraídas
    df_features = pd.DataFrame(lista_features)

    # Seleccionar solo las columnas relevantes que no tienen NaN
    columnas_relevantes = [
        'HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_SDSD',
        'HRV_pNN50', 'HRV_pNN20', 'HRV_IQRNN',
        'HRV_HTI', 'HRV_TINN', 'HRV_HFn', 'HRV_LnHF',
        'ECG_R_Peaks_Count', 'Label'
    ]

    # Verificar que todas las columnas relevantes existen
    columnas_existentes = [c for c in columnas_relevantes if c in df_features.columns]
    df_features_relevantes = df_features[columnas_existentes].dropna()

    print(df_features_relevantes.head())
    print(df_features_relevantes.info())

    # Visualización de Distribuciones de Características HRV
    caracteristicas_hrv = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_SDSD']

    for caracteristica in caracteristicas_hrv:
        if caracteristica in df_features_relevantes.columns:
            plt.figure(figsize=(8, 4))
            sns.histplot(df_features_relevantes[caracteristica].dropna(), kde=True)
            plt.title(f'Distribución de {caracteristica}')
            plt.xlabel(caracteristica)
            plt.ylabel('Frecuencia')
            plt.tight_layout()
            plt.show()
        else:
            print(f'Advertencia: La característica "{caracteristica}" no está presente en el DataFrame.')

    if 'HRV_SDNN' in df_features_relevantes.columns and 'Label' in df_features_relevantes.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Label', y='HRV_SDNN', data=df_features_relevantes)
        plt.title('HRV_SDNN por Etiqueta')
        plt.xlabel('Etiqueta')
        plt.ylabel('HRV_SDNN')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print('Advertencia: No se encontraron las columnas necesarias para el boxplot.')

    crear_carpeta(os.path.dirname(salida_csv))
    df_features_relevantes.to_csv(salida_csv, index=False)
    print(f"Dataset guardado exitosamente como '{salida_csv}'")

def main():
    directorio_s = r'C:\Users\Jm\Desktop\MASTER MII\SEGUNDO\E-HEALTH\WESAD'

    carpeta_salida = os.path.join(os.getcwd(), 'datos')
    crear_carpeta(carpeta_salida)

    carpetas = [f for f in os.listdir(directorio_s) if os.path.isdir(os.path.join(directorio_s, f)) and f.upper().startswith('S') and f[1:].isdigit()]

    if not carpetas:
        print(f'No se encontraron carpetas que empiecen con "S" seguido de un número en {directorio_s}')
        return

    print(f'Encontradas {len(carpetas)} carpeta(s) para procesar.')

    for carpeta in carpetas:
        ruta_carpeta = os.path.join(directorio_s, carpeta)
        nombre_sin_prefijo = carpeta.upper()  # 'S1', 'S2', etc.
        ruta_txt = os.path.join(ruta_carpeta, f'{nombre_sin_prefijo}_respiban.txt')
        ruta_pkl = os.path.join(ruta_carpeta, f'{nombre_sin_prefijo}.pkl')

        if not os.path.exists(ruta_txt):
            print(f'Advertencia: No se encontró el archivo TXT en {ruta_carpeta}')
            continue
        if not os.path.exists(ruta_pkl):
            print(f'Advertencia: No se encontró el archivo PKL en {ruta_carpeta}')
            continue

        numero_archivo = carpeta[1:]  # Extraer el número después de 'S', por ejemplo 'S3' -> '3'
        salida_csv = os.path.join(carpeta_salida, f'caracteristicas_ECG_procesadas_sin_normalizacion{numero_archivo}.csv')

        procesar_archivo(ruta_txt, ruta_pkl, salida_csv)

    print("\nProcesamiento completado para todas las carpetas existentes.")

if __name__ == "__main__":
    main()
