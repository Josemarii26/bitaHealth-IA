# BitaHealth

**BitaHealth** es una herramienta integral de análisis de salud basada en señales biométricas. Utilizando técnicas de procesamiento de señales (como la detección de picos ECG y el análisis de la Variabilidad de la Frecuencia Cardíaca - HRV) junto con modelos de clasificación entrenados con IA, BitaHealth permite obtener información relevante sobre el estado de salud (por ejemplo, detectar estrés, meditación, baseline, etc.) a partir de datos recogidos con dispositivos BITalino.

## Características

- **Procesamiento de señales ECG**: Limpieza, detección de picos y cálculo de métricas HRV.
- **Segmentación por ventanas**: Permite tratar señales largas dividiéndolas en segmentos de tiempo (por ejemplo, ventanas de 30 segundos).
- **Predicción con IA**: Modelos entrenados (k-NN, SVM, Decision Tree, RandomForest, etc.) que clasifican las muestras según diferentes estados (Meditación, Estrés, No Definido/Transitorio, Baseline, Diversión).
- **Generación de gráficos**: Se generan gráficos de la señal limpia, intervalos RR, Poincaré plot, histograma, espectro FFT y un segmento ampliado con pico destacado.
- **Descarga de informes PDF**: Los resultados y gráficos se pueden exportar en un informe PDF multi-página.

## Estructura del Repositorio

- **dataset1-10/**  
  Contiene los CSV con los datasets empleados para entrenar a la IA.  
- **datos/**  
  Carpeta con CSV de datos de diferentes pacientes (con etiquetas).  
- **datos_sin_label/**  
  Versión de los CSV anteriores pero sin la columna de etiquetas.
- **Resultados_HRV/**  
  Carpeta donde se organizan los resultados de la IA clasificados por dataset.  
  - **Artifacts/**: Objetos de preprocesado (imputer, scaler, PCA, labelencoder) para Dataset 4.  
  - **Models/**: Modelos entrenados (k-NN, SVM, etc.) para cada dataset.  
  - **Plots/**: Gráficas generadas durante el estudio de los resultados.  
  - **Reports/**: Informes y resúmenes generados (por ejemplo, matrices de confusión, curvas ROC, etc.).

- **scripts/** (opcional)  
  Aquí se encuentran scripts adicionales, por ejemplo:  
  - `Eliminar fila.py`: Script para eliminar la columna "Label" de un CSV.
  - `filtradoWESAD.py`: Script para el procesamiento y filtrado de datos del conjunto WESAD.
  
- **flaskServer.py**  
  Contiene el servidor Flask que:  
  - Recibe el archivo TXT de BITalino.  
  - Segmenta la señal en ventanas, calcula métricas HRV y genera gráficos (codificados en Base64).  
  - Ejecuta el pipeline de la IA y devuelve los resultados en formato JSON, incluyendo la opción de descargar un informe PDF.



## Requisitos e Instalación

### Dependencias del Servidor (Flask)
- Python 3.x  
- [NeuroKit2](https://github.com/neuropsychology/NeuroKit)  
- pandas, numpy  
- joblib  
- Flask, flask-cors  
- matplotlib, seaborn  
- jsPDF y html2canvas (si se utiliza para generar el PDF en el servidor o desde el front)

Para instalar las dependencias, puedes usar `pip`:

`pip install neurokit2 pandas numpy joblib flask flask-cors matplotlib seaborn`

## Uso y ejecución del Servidor Flask
Desde la carpeta raíz del proyecto (o donde se encuentre flaskServer.py), ejecuta:

`python flaskServer.py`


